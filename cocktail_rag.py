import pandas as pd
import faiss
import os
import time
import json
import shutil
from datetime import datetime
from sentence_transformers import SentenceTransformer
from openai import OpenAI

def get_terminal_width():
    """Get terminal width with safe defaults."""
    try:
        return min(shutil.get_terminal_size().columns - 2, 120)
    except:
        return 80

CSV_PATH = "hotaling_cocktails - Cocktails.csv"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# OpenRouter configuration
OPENROUTER_API_KEY = " YOUR OPENROUTER_API_KEY"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Model configurations for evaluation
MODELS_TO_EVALUATE = [
    {
        "name": "Gemini-3-Pro",
        "model_id": "google/gemini-3-pro-preview",
        "description": "Google Gemini 3 pro preview"
    },
    {
        "name": "GPT-5.0",
        "model_id": "openai/gpt-5",
        "description": "OpenAI GPT-5.0"
    },
    {
        "name": "Qwen-2.5-72B",
        "model_id": "qwen/qwen-2.5-72b-instruct",
        "description": "Qwen 2.5 72B Instruct"
    },
    {
        "name": "Llama-3.1-70B",
        "model_id": "meta-llama/llama-3.1-70b-instruct",
        "description": "Meta Llama 3.1 70B Instruct"
    }
]

# Default model (for regular use)
OPENROUTER_MODEL = "openai/gpt-4o-mini"  

# Text splitting configuration
# Enable text splitting to break long documents into smaller chunks for better retrieval
ENABLE_TEXT_SPLITTING = True  # Set to False to disable splitting (use full documents)
CHUNK_SIZE = 500  # Maximum characters per chunk
CHUNK_OVERLAP = 100  # Overlap between chunks to maintain context


def split_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Split text into chunks with overlap for better context preservation.
    
    Args:
        text: Text to split
        chunk_size: Maximum characters per chunk
        chunk_overlap: Number of characters to overlap between chunks
    
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundary if possible
        if end < len(text):
            # Look for sentence endings near the chunk boundary
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
    """Build the document list and vector index with optional text splitting.
    
    Args:
        df: DataFrame containing cocktail data
        model: SentenceTransformer model for embeddings
        use_splitting: If True, split long documents into chunks
    
    Returns:
        index: FAISS vector index
        docs: List of document chunks (may be more than original rows if splitting is used)
    """
    all_docs = []
    
    print("Processing documents and splitting text...")
    for idx, row in df.iterrows():
        # Create base document with metadata
        base_doc = row_to_doc(row)
        cocktail_name = row.get('Cocktail Name', 'Unknown')
        
        if use_splitting and len(base_doc) > CHUNK_SIZE:
            # Split long documents into chunks
            chunks = split_text(base_doc, CHUNK_SIZE, CHUNK_OVERLAP)
            
            # Add metadata to each chunk
            for i, chunk in enumerate(chunks):
                chunk_with_meta = f"[Cocktail: {cocktail_name} | Part {i+1}/{len(chunks)}]\n{chunk}"
                all_docs.append(chunk_with_meta)
        else:
            # Keep short documents as-is
            all_docs.append(base_doc)
    
    print(f"Total document chunks: {len(all_docs)} (from {len(df)} original recipes)")
    
    # Generate embeddings
    print("Generating embeddings...")
    emb = model.encode(all_docs, convert_to_numpy=True, show_progress_bar=True)
    faiss.normalize_L2(emb)  # normalize so inner product is cosine similarity
    
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


def generate_response(query, retrieved_docs, client, model_name=OPENROUTER_MODEL, track_metrics=False):
    """Generate a response using OpenRouter LLM with retrieved context.
    
    Args:
        query: User query
        retrieved_docs: List of retrieved document texts
        client: OpenAI client configured for OpenRouter
        model_name: Model identifier
        track_metrics: If True, returns metrics along with response
    
    Returns:
        If track_metrics=False: response text
        If track_metrics=True: dict with 'response', 'response_time', 'tokens_used', etc.
    """
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
        start_time = time.time()
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a knowledgeable cocktail expert who helps users find and understand cocktail recipes."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        response_time = time.time() - start_time
        
        response_text = response.choices[0].message.content
        
        if track_metrics:
            # Extract token usage if available
            usage = getattr(response, 'usage', None)
            tokens_used = {
                'prompt_tokens': usage.prompt_tokens if usage else None,
                'completion_tokens': usage.completion_tokens if usage else None,
                'total_tokens': usage.total_tokens if usage else None
            }
            
            return {
                'response': response_text,
                'response_time': response_time,
                'tokens_used': tokens_used,
                'response_length': len(response_text),
                'word_count': len(response_text.split())
            }
        else:
            return response_text
    except Exception as e:
        error_msg = f"Error calling OpenRouter API: {str(e)}"
        if track_metrics:
            return {
                'response': error_msg,
                'response_time': None,
                'tokens_used': None,
                'response_length': 0,
                'word_count': 0,
                'error': str(e)
            }
        return error_msg


def calculate_relevance_score(response, query, retrieved_docs, embedding_model):
    """Calculate relevance score between response and query using embeddings."""
    try:
        query_emb = embedding_model.encode([query], convert_to_numpy=True)
        response_emb = embedding_model.encode([response], convert_to_numpy=True)
        
        # Cosine similarity
        from numpy import dot
        from numpy.linalg import norm
        similarity = dot(query_emb[0], response_emb[0]) / (norm(query_emb[0]) * norm(response_emb[0]))
        return float(similarity)
    except:
        return 0.0


def check_avoids_repetition(response, retrieved_docs):
    """Check if response avoids repeating recipe details (lower score = better, avoids repetition)."""
    # Simple heuristic: count how many recipe details appear verbatim in response
    repetition_count = 0
    response_lower = response.lower()
    
    for doc in retrieved_docs:
        # Extract key parts (ingredients, preparation steps)
        doc_lines = doc.lower().split('\n')
        for line in doc_lines:
            if len(line) > 20:  # Only check substantial lines
                if line in response_lower:
                    repetition_count += 1
    
    # Normalize: 0 = no repetition (best), higher = more repetition
    return max(0, 1.0 - (repetition_count / max(len(retrieved_docs) * 2, 1)))


def check_completeness(response, query):
    """Check if response addresses key aspects of the query."""
    query_lower = query.lower()
    response_lower = response.lower()
    
    # Check for key question words/phrases
    completeness_score = 0.0
    total_checks = 0
    
    # Check for recommendation/analysis
    if any(word in response_lower for word in ['recommend', 'suggest', 'best', 'match', 'suitable']):
        completeness_score += 1.0
    total_checks += 1
    
    # Check for explanation/comparison
    if any(word in response_lower for word in ['because', 'why', 'compare', 'different', 'unique']):
        completeness_score += 1.0
    total_checks += 1
    
    # Check for suggestions/modifications
    if any(word in response_lower for word in ['modify', 'variation', 'alternative', 'adjust', 'tip']):
        completeness_score += 1.0
    total_checks += 1
    
    return completeness_score / total_checks if total_checks > 0 else 0.0


def evaluate_model_performance(model_config, test_queries, embedding_model, index, docs, client):
    """Evaluate a single model's performance on test queries."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_config['name']} ({model_config['model_id']})")
    print(f"{'='*60}")
    
    results = []
    successful = 0
    failed = 0
    
    print(f"\nStarting to process {len(test_queries)} queries for {model_config['name']}...")
    print(f"Total queries to process: {len(test_queries)}")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"[Query {i}/{len(test_queries)}] Processing...")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        try:
            # Perform retrieval
            print(f"  Step 1/3: Performing vector search...")
            search_results = search(query, embedding_model, index, docs, k=5)
            retrieved_docs = [doc for _, doc in search_results]
            print(f"  Step 1/3: Found {len(retrieved_docs)} relevant recipes")
            
            # Generate response with metrics
            print(f"  Step 2/3: Calling LLM API ({model_config['name']})...")
            result = generate_response(query, retrieved_docs, client, model_config['model_id'], track_metrics=True)
            print(f"  Step 2/3: LLM response received")
            
            print(f"  Step 3/3: Calculating metrics...")
            
            if 'error' in result:
                print(f"  [ERROR] Error: {result['error']}")
                results.append({
                    'query': query,
                    'model': model_config['name'],
                    'error': result['error'],
                    'metrics': None
                })
                failed += 1
                continue
            
            # Calculate evaluation metrics
            relevance = calculate_relevance_score(result['response'], query, retrieved_docs, embedding_model)
            avoids_repetition = check_avoids_repetition(result['response'], retrieved_docs)
            completeness = check_completeness(result['response'], query)
            
            metrics = {
                'response_time': result['response_time'],
                'response_length': result['response_length'],
                'word_count': result['word_count'],
                'tokens_used': result['tokens_used'],
                'relevance_score': relevance,
                'avoids_repetition_score': avoids_repetition,
                'completeness_score': completeness,
                'response': result['response']
            }
            
            # Calculate overall score (weighted average)
            overall_score = (
                relevance * 0.4 +           # 40% weight on relevance
                avoids_repetition * 0.3 +   # 30% weight on avoiding repetition
                completeness * 0.3          # 30% weight on completeness
            )
            metrics['overall_score'] = overall_score
            
            results.append({
                'query': query,
                'model': model_config['name'],
                'metrics': metrics
            })
            
            successful += 1
            print(f"  [OK] Response time: {result['response_time']:.2f}s")
            print(f"  [OK] Overall score: {overall_score:.3f}")
            
        except Exception as e:
            print(f"  [ERROR] Exception: {str(e)}")
            results.append({
                'query': query,
                'model': model_config['name'],
                'error': str(e),
                'metrics': None
            })
            failed += 1
            continue
    
    print(f"\n{'-'*60}")
    print(f"Model {model_config['name']} completed:")
    print(f"  Total queries: {len(test_queries)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Results collected: {len(results)}")
    
    # Verify we processed all queries
    if len(results) != len(test_queries):
        print(f"  [WARNING] Mismatch! Expected {len(test_queries)} results, got {len(results)}")
    else:
        print(f"  [OK] All {len(test_queries)} queries processed")
    print(f"{'-'*60}")
    
    return results


def generate_evaluation_report(all_results, output_file="model_evaluation_report.json"):
    """Generate a comprehensive evaluation report."""
    # Aggregate metrics by model
    model_stats = {}
    all_model_names = set()  # Track all models, including failed ones
    
    # First pass: collect all model names
    for result in all_results:
        model_name = result['model']
        all_model_names.add(model_name)
    
    # Initialize stats for all models
    for model_name in all_model_names:
        model_stats[model_name] = {
            'response_times': [],
            'overall_scores': [],
            'relevance_scores': [],
            'avoids_repetition_scores': [],
            'completeness_scores': [],
            'response_lengths': [],
            'word_counts': [],
            'total_tokens': [],
            'query_count': 0,
            'failed_count': 0
        }
    
    # Second pass: collect metrics for successful queries
    for result in all_results:
        model_name = result['model']
        
        if result['metrics'] is None:
            # Track failed queries
            model_stats[model_name]['failed_count'] += 1
            continue
        
        metrics = result['metrics']
        model_stats[model_name]['response_times'].append(metrics['response_time'])
        model_stats[model_name]['overall_scores'].append(metrics['overall_score'])
        model_stats[model_name]['relevance_scores'].append(metrics['relevance_score'])
        model_stats[model_name]['avoids_repetition_scores'].append(metrics['avoids_repetition_score'])
        model_stats[model_name]['completeness_scores'].append(metrics['completeness_score'])
        model_stats[model_name]['response_lengths'].append(metrics['response_length'])
        model_stats[model_name]['word_counts'].append(metrics['word_count'])
        if metrics['tokens_used'] and metrics['tokens_used']['total_tokens']:
            model_stats[model_name]['total_tokens'].append(metrics['tokens_used']['total_tokens'])
        model_stats[model_name]['query_count'] += 1
    
    # Calculate averages - include all models, even if they failed
    report = {
        'timestamp': datetime.now().isoformat(),
        'models_evaluated': sorted(list(all_model_names)),
        'summary': {},
        'detailed_results': all_results
    }
    
    for model_name, stats in model_stats.items():
        report['summary'][model_name] = {
            'avg_response_time': sum(stats['response_times']) / len(stats['response_times']) if stats['response_times'] else None,
            'avg_overall_score': sum(stats['overall_scores']) / len(stats['overall_scores']) if stats['overall_scores'] else None,
            'avg_relevance_score': sum(stats['relevance_scores']) / len(stats['relevance_scores']) if stats['relevance_scores'] else None,
            'avg_avoids_repetition_score': sum(stats['avoids_repetition_scores']) / len(stats['avoids_repetition_scores']) if stats['avoids_repetition_scores'] else None,
            'avg_completeness_score': sum(stats['completeness_scores']) / len(stats['completeness_scores']) if stats['completeness_scores'] else None,
            'avg_response_length': sum(stats['response_lengths']) / len(stats['response_lengths']) if stats['response_lengths'] else None,
            'avg_word_count': sum(stats['word_counts']) / len(stats['word_counts']) if stats['word_counts'] else None,
            'avg_tokens_used': sum(stats['total_tokens']) / len(stats['total_tokens']) if stats['total_tokens'] else None,
            'total_queries': stats['query_count'],
            'failed_queries': stats['failed_count']
        }
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    return report


def print_evaluation_summary(report):
    """Print a formatted summary of the evaluation."""
    # Get terminal width, default to 80 if unavailable
    safe_width = get_terminal_width()
    
    print("\n" + "="*safe_width)
    print("EVALUATION SUMMARY")
    print("="*safe_width)
    
    summary = report['summary']
    
    if not summary:
        print("[ERROR] No summary data available!")
        return
    
    # Sort models by overall score
    sorted_models = sorted(summary.items(), 
                         key=lambda x: x[1]['avg_overall_score'] if x[1]['avg_overall_score'] is not None else 0, 
                         reverse=True)
    
    # Calculate column widths based on terminal width
    # Model name: flexible, but max 25 chars
    # Other columns: compact but readable
    model_col_width = min(25, safe_width // 6)
    score_col_width = 10
    time_col_width = 9
    
    # Adjust if terminal is narrow
    if safe_width < 90:
        model_col_width = 20
        score_col_width = 8
        time_col_width = 8
    
    # Print header with proper alignment
    header = f"{'Model':<{model_col_width}} {'Overall':<{score_col_width}} {'Relevance':<{score_col_width}} {'NoRepeat':<{score_col_width}} {'Complete':<{score_col_width}} {'Time(s)':<{time_col_width}}"
    print(f"\n{header}")
    print("-" * len(header))
    
    # Print each model's data
    for model_name, stats in sorted_models:
        failed_count = stats.get('failed_queries', 0)
        total_queries = stats.get('total_queries', 0) + failed_count
        
        # If all queries failed, show special indicator
        if total_queries > 0 and failed_count == total_queries:
            display_name = model_name[:model_col_width-2] if len(model_name) > model_col_width-2 else model_name
            row = f"{display_name:<{model_col_width}} {'[FAILED]':<{score_col_width}} {'N/A':<{score_col_width}} {'N/A':<{score_col_width}} {'N/A':<{score_col_width}} {'N/A':<{time_col_width}}"
            print(row)
            continue
        
        overall = stats['avg_overall_score'] if stats['avg_overall_score'] is not None else 0.0
        relevance = stats['avg_relevance_score'] if stats['avg_relevance_score'] is not None else 0.0
        no_repeat = stats['avg_avoids_repetition_score'] if stats['avg_avoids_repetition_score'] is not None else 0.0
        complete = stats['avg_completeness_score'] if stats['avg_completeness_score'] is not None else 0.0
        time_avg = stats['avg_response_time'] if stats['avg_response_time'] is not None else 0.0
        
        # Truncate model name if too long
        display_name = model_name[:model_col_width-2] if len(model_name) > model_col_width-2 else model_name
        
        # Add warning if some queries failed (but keep it short)
        if failed_count > 0:
            display_name = f"{display_name[:model_col_width-8]} ({failed_count}f)"
        
        # Format numbers with appropriate precision
        row = f"{display_name:<{model_col_width}} {overall:<{score_col_width}.3f} {relevance:<{score_col_width}.3f} {no_repeat:<{score_col_width}.3f} {complete:<{score_col_width}.3f} {time_avg:<{time_col_width}.2f}"
        print(row)
    
    # Find best model
    if sorted_models:
        best_model = sorted_models[0][0]
        best_score = sorted_models[0][1]['avg_overall_score']
        if best_score is not None:
            print(f"\n[WINNER] Best Overall Model: {best_model}")
            print(f"   Average Overall Score: {best_score:.3f}")
        else:
            print(f"\n[WARNING] Could not determine best model (no valid scores)")
    else:
        print(f"\n[ERROR] No models to compare!")
    
    print("\n" + "="*safe_width)


def evaluate_models(test_queries=None):
    """Main evaluation function to test multiple models."""
    if test_queries is None:
        # Default test queries covering different aspects
        test_queries = [
            "Looking for a refreshing highball-style cocktail with ginger beer or ginger notes",
            "I want a whiskey-based cocktail that's not too sweet, suitable for winter",
            "Find me a tropical cocktail with pineapple and coconut flavors",
            "What's a good cocktail for a party that can be made in large batches?",
            "I need a low-alcohol cocktail that's still flavorful and interesting",
            "I'm looking for a classic gin cocktail that's elegant and sophisticated",
            "Can you recommend a tequila cocktail with citrus and a spicy kick?",
            "I want something fruity and sweet, perfect for a summer brunch",
            "Find me a complex cocktail with multiple spirits and herbal notes",
            "I need a quick and easy cocktail recipe that requires minimal ingredients"
        ]
    
    print("="*80)
    print("RAG MODEL EVALUATION")
    print("="*80)
    print(f"Testing {len(MODELS_TO_EVALUATE)} models on {len(test_queries)} queries")
    print(f"\nTest queries ({len(test_queries)} total):")
    for i, q in enumerate(test_queries, 1):
        print(f"  {i}. {q}")
    print()
    
    # Load data and build index
    print("\nLoading data and building index...")
    df = pd.read_csv(CSV_PATH)
    embedding_model = SentenceTransformer(EMB_MODEL)
    index, docs = build_index(df, embedding_model)
    
    # Initialize OpenRouter client
    if not OPENROUTER_API_KEY:
        print("Error: OPENROUTER_API_KEY not set!")
        return
    
    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
    )
    
    # Evaluate each model
    all_results = []
    total_models = len(MODELS_TO_EVALUATE)
    
    for model_idx, model_config in enumerate(MODELS_TO_EVALUATE, 1):
        print(f"\n{'#'*80}")
        print(f"MODEL {model_idx}/{total_models}: {model_config['name']}")
        print(f"Model ID: {model_config['model_id']}")
        print(f"{'#'*80}")
        
        try:
            print(f"Starting evaluation of {model_config['name']} on {len(test_queries)} queries...")
            results = evaluate_model_performance(
                model_config, test_queries, embedding_model, index, docs, client
            )
            
            # Verify we got results for all queries
            if len(results) != len(test_queries):
                print(f"[WARNING] Expected {len(test_queries)} results, got {len(results)}")
            
            all_results.extend(results)
            print(f"\n[OK] Successfully evaluated {model_config['name']}: {len(results)} results")
        except Exception as e:
            import traceback
            print(f"[ERROR] Failed to evaluate {model_config['name']}: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            # Still add error results for this model
            for query in test_queries:
                all_results.append({
                    'query': query,
                    'model': model_config['name'],
                    'error': str(e),
                    'metrics': None
                })
            continue
    
    # Generate report
    safe_width = get_terminal_width()
    print("\n" + "="*safe_width)
    print("GENERATING EVALUATION REPORT")
    print("="*safe_width)
    
    if not all_results:
        print("[ERROR] No results to generate report from!")
        return None
    
    # Check which models were evaluated
    models_evaluated = set()
    for result in all_results:
        if result.get('model'):
            models_evaluated.add(result['model'])
    
    print(f"\nModels evaluated: {len(models_evaluated)}/{len(MODELS_TO_EVALUATE)}")
    print(f"Expected models: {[m['name'] for m in MODELS_TO_EVALUATE]}")
    print(f"Evaluated models: {sorted(models_evaluated)}")
    
    if len(models_evaluated) < len(MODELS_TO_EVALUATE):
        missing = [m['name'] for m in MODELS_TO_EVALUATE if m['name'] not in models_evaluated]
        print(f"[WARNING] Missing models: {missing}")
    
    report = generate_evaluation_report(all_results)
    print_evaluation_summary(report)
    
    # Print summary statistics
    safe_width = get_terminal_width()
    print("\n" + "="*safe_width)
    print("DETAILED STATISTICS BY MODEL")
    print("="*safe_width)
    
    summary = report['summary']
    
    # Sort by overall score for consistent display
    sorted_by_score = sorted(summary.items(), 
                           key=lambda x: x[1]['avg_overall_score'] if x[1]['avg_overall_score'] is not None else 0,
                           reverse=True)
    
    for model_name, stats in sorted_by_score:
        print(f"\n{model_name}:")
        failed_count = stats.get('failed_queries', 0)
        total_processed = stats['total_queries']
        total_attempted = total_processed + failed_count
        print(f"  {'Queries processed:':<25} {total_processed}/{len(test_queries)}")
        if failed_count > 0:
            print(f"  {'Failed queries:':<25} {failed_count}")
        if stats['avg_overall_score'] is not None:
            print(f"  {'Average Overall Score:':<25} {stats['avg_overall_score']:.3f}")
        if stats['avg_relevance_score'] is not None:
            print(f"  {'Average Relevance Score:':<25} {stats['avg_relevance_score']:.3f}")
        if stats['avg_avoids_repetition_score'] is not None:
            print(f"  {'Average No-Repetition Score:':<25} {stats['avg_avoids_repetition_score']:.3f}")
        if stats['avg_completeness_score'] is not None:
            print(f"  {'Average Completeness Score:':<25} {stats['avg_completeness_score']:.3f}")
        if stats['avg_response_time'] is not None:
            print(f"  {'Average Response Time:':<25} {stats['avg_response_time']:.2f}s")
        if stats['avg_tokens_used'] is not None:
            print(f"  {'Average Tokens Used:':<25} {stats['avg_tokens_used']:.0f}")
        if stats['avg_word_count'] is not None:
            print(f"  {'Average Word Count:':<25} {stats['avg_word_count']:.0f}")
    
    safe_width = get_terminal_width()
    print(f"\n{'='*safe_width}")
    print(f"Full detailed results saved to: model_evaluation_report.json")
    print(f"{'='*safe_width}")
    
    return report


def main():
    # Load data and build index
    df = pd.read_csv(CSV_PATH)
    model = SentenceTransformer(EMB_MODEL)
    index, docs = build_index(df, model)
    
    # Initialize OpenRouter client
    if not OPENROUTER_API_KEY:
        print("Warning: OPENROUTER_API_KEY not set. Please set it as an environment variable.")
        print("Example: export OPENROUTER_API_KEY='your-api-key'")
        print("\nShowing search results only (without LLM response):\n")
        use_llm = False
    else:
        client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
        )
        use_llm = True

    # Example query: adjust as needed
    query = "Looking for a refreshing highball-style cocktail with ginger beer or ginger notes"
    
    # Perform vector search
    results = search(query, model, index, docs, k=5)
    
    # Display search results
    print("=" * 60)
    print("RETRIEVED RECIPES:")
    print("=" * 60)
    retrieved_docs = []
    for score, doc in results:
        print(f"\n[Score: {score:.3f}]\n{doc}\n" + "-" * 40)
        retrieved_docs.append(doc)
    
    # Generate LLM response if API key is available
    if use_llm:
        print("\n" + "=" * 60)
        print("AI ANALYSIS & RECOMMENDATIONS:")
        print("=" * 60)
        print("(Based on the recipes above, providing insights and suggestions)\n")
        response = generate_response(query, retrieved_docs, client)
        print(response)


if __name__ == "__main__":
    import sys
    
    # Debug: Print command line arguments
    print(f"\n[DEBUG] Command line arguments: {sys.argv}")
    print(f"[DEBUG] Number of arguments: {len(sys.argv)}")
    if len(sys.argv) > 1:
        print(f"[DEBUG] First argument: '{sys.argv[1]}'")
    
    # Check if user wants to run evaluation
    if len(sys.argv) > 1 and sys.argv[1] == "--evaluate":
        print(f"[DEBUG] Evaluation mode flag detected!")
        # Run model evaluation
        print("\n" + "="*80)
        print("STARTING MODEL EVALUATION MODE")
        print("="*80)
        print(f"Will evaluate {len(MODELS_TO_EVALUATE)} models:")
        for i, m in enumerate(MODELS_TO_EVALUATE, 1):
            print(f"  {i}. {m['name']} ({m['model_id']})")
        print("="*80)
        
        if len(sys.argv) > 2:
            # Custom test queries provided
            test_queries = sys.argv[2:]
            print(f"\nUsing {len(test_queries)} custom test queries")
            evaluate_models(test_queries)
        else:
            # Use default test queries
            print(f"\nUsing 10 default test queries")
            evaluate_models()
        
        print("\n" + "="*80)
        print("EVALUATION COMPLETE")
        print("="*80)
    else:
        # Run normal RAG query
        print("\n" + "="*80)
        print("RAG QUERY MODE (Single Query)")
        print("="*80)
        print("[WARNING] You are running in SINGLE QUERY mode!")
        print("[WARNING] This will only process ONE query with ONE model.")
        print("")
        print("If you want to evaluate ALL 4 models on ALL 10 queries,")
        print("you MUST use this command:")
        print("")
        print("   python cocktail_rag.py --evaluate")
        print("")
        print("="*80)
        print("Press Ctrl+C to cancel and run with --evaluate instead")
        print("Or wait 3 seconds to continue with single query mode...")
        print("="*80)
        
        import time
        try:
            time.sleep(3)
        except KeyboardInterrupt:
            print("\nCancelled. Please run: python cocktail_rag.py --evaluate")
            sys.exit(0)
        
        print("\nContinuing with single query mode...\n")
        main()

