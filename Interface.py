"""
Cocktail RAG Command-Line Interface

A simple CLI that takes a question, embeds it, finds the most relevant 
document chunks from the vector store, and passes them along with the 
question to an LLM to generate an answer.
"""

import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
import sys

# Configuration
CSV_PATH = "hotaling_cocktails - Cocktails.csv"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OPENROUTER_API_KEY = "YOUR OPENROUTER_API_KEY"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
BEST_MODEL = "google/gemini-3-pro-preview"  # Best performing model from evaluation

# Text splitting configuration (matching cocktail_rag.py)
ENABLE_TEXT_SPLITTING = True
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def split_text(text, chunk_size, chunk_overlap):
    """Splits text into chunks of specified size with overlap."""
    if not isinstance(text, str) or not text.strip():
        return []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundaries
        if end < len(text):
            for i in range(end, max(start + chunk_size - 100, start), -1):
                if text[i] in '.!?\n':
                    end = i + 1
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = end - chunk_overlap
        if start >= len(text):
            break
    
    return chunks


def row_to_doc(row):
    """Convert a CSV row into a readable text block for retrieval."""
    parts = [
        f"Name: {row.get('Cocktail Name', '')}",
        f"Bartender/Bar: {row.get('Bartender', '')} | {row.get('Bar/Company', '')} | {row.get('Location', '')}",
        f"Ingredients: {row.get('Ingredients', '')}",
        f"Garnish: {row.get('Garnish', '')}",
        f"Glassware: {row.get('Glassware', '')}",
        f"Preparation: {row.get('Preparation', '')}",
        f"Notes: {row.get('Notes', '')}",
    ]
    return "\n".join([p for p in parts if isinstance(p, str) and p.strip()])


def build_index(df, model, use_splitting=ENABLE_TEXT_SPLITTING):
    """Build the document list and vector index."""
    all_docs = []
    
    print("Processing documents and splitting text...")
    for idx, row in df.iterrows():
        base_doc = row_to_doc(row)
        cocktail_name = row.get('Cocktail Name', 'Unknown')
        
        if use_splitting and len(base_doc) > CHUNK_SIZE:
            chunks = split_text(base_doc, CHUNK_SIZE, CHUNK_OVERLAP)
            for i, chunk in enumerate(chunks):
                chunk_with_meta = f"[Cocktail: {cocktail_name} | Part {i+1}/{len(chunks)}]\n{chunk}"
                all_docs.append(chunk_with_meta)
        else:
            all_docs.append(base_doc)
    
    print(f"Total document chunks: {len(all_docs)} (from {len(df)} original recipes)")
    
    # Generate embeddings
    print("Generating embeddings...")
    emb = model.encode(all_docs, convert_to_numpy=True, show_progress_bar=True)
    faiss.normalize_L2(emb)
    
    # Build FAISS index
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    
    print(f"Index built with {index.ntotal} vectors")
    return index, all_docs


def search(query, model, index, docs, k=5):
    """Vector search, returning (score, text) pairs."""
    q = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q)
    scores, idx = index.search(q, k)
    return [(float(scores[0][i]), docs[idx[0][i]]) for i in range(k)]


def generate_answer(query, retrieved_docs, client):
    """Generate an answer using the LLM with retrieved context."""
    # Build context from retrieved documents
    context = "\n\n---\n\n".join([f"Recipe {i+1}:\n{doc}" for i, doc in enumerate(retrieved_docs)])
    
    prompt = f"""You are a cocktail expert assistant. The user has already seen the detailed recipes below. Your job is to provide ANALYSIS and INSIGHTS, not to repeat the recipe details.

Retrieved Recipes (already shown to user):
{context}

User Question: {query}

Please provide a CONCISE response that:
1. **Summarizes** which recipe(s) best match the request and why (focus on key ingredients, style, or characteristics)
2. **Compares** the recipes if multiple are relevant (what makes each unique)
3. **Suggests** modifications, variations, or alternatives if applicable
4. **Provides** any additional context or tips (e.g., difficulty level, best occasions, flavor profile)

IMPORTANT: Do NOT repeat the full recipe details. The user can already see them above. Focus on analysis, recommendations, and insights."""

    try:
        response = client.chat.completions.create(
            model=BEST_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful cocktail expert assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"


def initialize_rag_system():
    """Initialize the RAG system by loading data and building index."""
    print("=" * 60)
    print("Initializing Cocktail RAG System")
    print("=" * 60)
    
    # Load data
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV file not found: {CSV_PATH}")
        sys.exit(1)
    
    print(f"Loading data from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} cocktail recipes")
    
    # Load embedding model
    print(f"Loading embedding model: {EMB_MODEL}...")
    embedding_model = SentenceTransformer(EMB_MODEL)
    
    # Build index
    index, docs = build_index(df, embedding_model)
    
    # Initialize OpenRouter client
    if not OPENROUTER_API_KEY:
        print("Error: OPENROUTER_API_KEY not set!")
        sys.exit(1)
    
    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
    )
    
    print("\n" + "=" * 60)
    print("RAG System Ready!")
    print(f"Using LLM: {BEST_MODEL}")
    print("=" * 60 + "\n")
    
    return embedding_model, index, docs, client


def main():
    """Main CLI loop."""
    # Initialize RAG system
    embedding_model, index, docs, client = initialize_rag_system()
    
    print("Cocktail RAG Interface")
    print("Type your questions about cocktails. Type 'quit' or 'exit' to exit.\n")
    
    while True:
        try:
            # Get user question
            query = input("Question: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            # Step 1: Embed the question
            print("\n[Step 1] Embedding question...")
            
            # Step 2: Find most relevant document chunks
            print("[Step 2] Searching vector store for relevant documents...")
            retrieved = search(query, embedding_model, index, docs, k=5)
            print(f"Found {len(retrieved)} relevant document chunks")
            
            # Show retrieved documents
            print("\n--- Retrieved Documents ---")
            retrieved_docs = []
            for i, (score, doc) in enumerate(retrieved, 1):
                print(f"\n[Document {i}] (Similarity: {score:.3f})")
                # Show first 200 characters
                preview = doc[:200] + "..." if len(doc) > 200 else doc
                print(preview)
                retrieved_docs.append(doc)
            print("\n" + "-" * 60)
            
            # Step 3: Pass to LLM to generate answer
            print(f"\n[Step 3] Generating answer using {BEST_MODEL}...")
            answer = generate_answer(query, retrieved_docs, client)
            
            # Display answer
            print("\n" + "=" * 60)
            print("ANSWER:")
            print("=" * 60)
            print(answer)
            print("=" * 60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}\n")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

