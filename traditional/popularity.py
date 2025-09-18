"""
Subreddit-Specific Popularity-Based Baseline for Music Recommendations

This script implements a subreddit-aware popularity-based baseline for music recommendations
using Reddit queries. It ranks candidates based on their frequency within each specific
subreddit for fair comparison with language models.

=== ALGORITHM ===
1. Load Reddit music queries with 100 candidate songs and ground truth recommendations
2. Build subreddit-specific popularity matrices from candidates within each subreddit
3. For each query, rank the 100 candidates by their popularity within that subreddit
4. Generate up to 100 ranked predictions per query
5. Evaluate using comprehensive ranking metrics
"""

import json
from pathlib import Path
import os
import random
import math
from datetime import datetime
from collections import defaultdict, Counter

import numpy as np
import wandb
from tqdm import tqdm

# Import our utilities
from utils import (
    normalize_string, normalize_list_of_lists,
    calculate_comprehensive_metrics, calculate_subreddit_metrics,
    print_metrics_summary, print_subreddit_comparison
)

# ─── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_JSONL   = os.getenv("INPUT_JSONL", "")
OUTPUT_JSONL  = os.getenv("POPULARITY_OUTPUT", "pipeline/language_models/results/FINAL_SUBREDDIT_POPULARITY_BASELINE.jsonl")
MAX_PREDICTIONS = 100
RANDOM_SEED   = 42
# ────────────────────────────────────────────────────────────────────────────────

# Initialize wandb (use WANDB_API_KEY from environment)
wandb.login()

def load_records(path):
    """Load JSONL records"""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records

def build_subreddit_popularity_matrices(records):
    print("Building subreddit-specific popularity matrices...")
    
    # Group records by subreddit
    subreddit_records = defaultdict(list)
    for record in records:
        subreddit = record.get("source_subreddit", "unknown")
        subreddit_records[subreddit].append(record)
    
    subreddit_popularity = {}
    
    for subreddit, sub_records in subreddit_records.items():
        print(f"\nProcessing subreddit: {subreddit} ({len(sub_records)} records)")
        
        candidate_counts = Counter()
        total_candidate_occurrences = 0
        
        # Count frequency of each song in this subreddit's candidate sets
        for record in sub_records:
            candidates = record.get("final_candidates", [])
            for song in candidates:
                candidate_counts[song] += 1
                total_candidate_occurrences += 1
        
        # Convert to popularity scores (normalized frequencies) for this subreddit
        popularity_scores = {}
        for song, count in candidate_counts.items():
            popularity_scores[song] = count / total_candidate_occurrences if total_candidate_occurrences > 0 else 0.0
        
        subreddit_popularity[subreddit] = popularity_scores
        
        print(f"  - {len(popularity_scores)} unique songs")
        print(f"  - {total_candidate_occurrences} total candidate occurrences")
        
        # Print some statistics for this subreddit
        if popularity_scores:
            scores = list(popularity_scores.values())
            print(f"  - Max popularity: {max(scores):.6f}")
            print(f"  - Mean popularity: {np.mean(scores):.6f}")
            
            # Show top 5 most popular songs for this subreddit
            top_songs = candidate_counts.most_common(5)
            print(f"  - Top 5 songs: {[song for song, _ in top_songs]}")
    
    return subreddit_popularity

def popularity_rerank(candidates, subreddit_popularity_scores, max_predictions=100):
    if not candidates:
        return []
    
    if not subreddit_popularity_scores:
        # Fallback: return candidates in original order
        return candidates[:max_predictions]
    
    # Score each candidate using subreddit-specific popularity
    candidate_scores = {}
    for song in candidates:
        candidate_scores[song] = subreddit_popularity_scores.get(song, 0.0)
    
    # Sort by popularity score (descending), then by original position for tie-breaking
    ranked_candidates = sorted(
        candidates,
        key=lambda x: (candidate_scores.get(x, 0.0), -candidates.index(x)),
        reverse=True
    )
    
    return ranked_candidates[:max_predictions]

def main():
    # Set random seeds for reproducibility
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Initialize wandb
    wandb.init(
        project="reddit-music-subreddit-popularity-baseline",
        entity="musiCRS",
        name=f"subreddit_popularity_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "max_predictions": MAX_PREDICTIONS,
            "random_seed": RANDOM_SEED,
            "mode": "subreddit_popularity_reranking"
        },
        tags=["subreddit-popularity-baseline", "music-recommendation", "reranking"]
    )
    
    # Initialize results collection
    results = {
        "all_predictions": [],
        "all_ground_truth": [],
        "all_subreddits": []
    }
    
    # 1. Load data
    print("Loading data...")
    records = load_records(INPUT_JSONL)
    print(f"Loaded {len(records)} records")
    
    # 2. Build subreddit-specific popularity matrices
    subreddit_popularity = build_subreddit_popularity_matrices(records)
    
    # 3. Generate predictions for each record using subreddit-specific popularity
    print("Generating subreddit-specific popularity-based predictions...")
    for record in tqdm(records, desc="Processing queries"):
        candidates = record.get("final_candidates", [])
        subreddit = record.get("source_subreddit", "unknown")
        
        # Get popularity scores for this subreddit
        subreddit_scores = subreddit_popularity.get(subreddit, {})
        
        # Generate popularity-based ranking using subreddit-specific scores
        predicted = popularity_rerank(
            candidates=candidates,
            subreddit_popularity_scores=subreddit_scores,
            max_predictions=MAX_PREDICTIONS
        )
        
        # Update record with predictions
        record["predicted"] = predicted
        
        # Collect results for evaluation
        results["all_predictions"].append(predicted)
        
        # Get ground truth (following same logic as NBCRS)
        ground_truth = record.get("limited_gt", 
                                 record.get("combined_ground_truth", 
                                           record.get("ground_truth", [])))
        results["all_ground_truth"].append(ground_truth)
        results["all_subreddits"].append(record.get("source_subreddit", "unknown"))
    
    # 4. Write out updated JSONL
    out_path = Path(OUTPUT_JSONL)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as wf:
        for rec in records:
            wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
    
    print(f"✅ Saved {len(records)} records with up to {MAX_PREDICTIONS} predictions each to {out_path}")
    
    # 5. Calculate comprehensive evaluation metrics
    print("Calculating evaluation metrics...")
    final_eval_metrics = calculate_comprehensive_metrics(
        results["all_predictions"],
        results["all_ground_truth"]
    )
    
    # Calculate per-subreddit metrics
    subreddit_metrics = calculate_subreddit_metrics(
        results["all_predictions"],
        results["all_ground_truth"], 
        results["all_subreddits"]
    )
    
    # 6. Log metrics to wandb
    final_log_dict = {}
    for metric_name, metric_value in final_eval_metrics.items():
        final_log_dict[f"FINAL_{metric_name}"] = metric_value
    
    # Log per-subreddit metrics to wandb
    for subreddit, sub_metrics in subreddit_metrics.items():
        for metric_name, metric_value in sub_metrics.items():
            final_log_dict[f"SUBREDDIT_{subreddit}_{metric_name}"] = metric_value
    
    wandb.log(final_log_dict)
    
    # 7. Print evaluation results
    if final_eval_metrics:
        print_metrics_summary(final_eval_metrics, "FINAL Evaluation Results for Subreddit-Specific Popularity Baseline")
        
        # Print per-subreddit results
        if subreddit_metrics:
            print_subreddit_comparison(subreddit_metrics)
    
    # 8. Analysis: Show some example predictions vs ground truth
    print("\n" + "="*80)
    print("SAMPLE PREDICTIONS vs GROUND TRUTH")
    print("="*80)
    
    for i in range(min(3, len(records))):
        record = records[i]
        subreddit = record.get("source_subreddit", "unknown")
        print(f"\nSample {i+1}:")
        print(f"Query: {record.get('query', 'N/A')[:100]}...")
        print(f"Subreddit: {subreddit}")
        print(f"Ground Truth: {results['all_ground_truth'][i][:5]}...")  # Show first 5
        print(f"Predicted: {results['all_predictions'][i][:5]}...")      # Show first 5
        
        # Check if any predictions match ground truth
        gt_set = set(results['all_ground_truth'][i])
        pred_set = set(results['all_predictions'][i][:10])  # Top 10 predictions
        matches = gt_set & pred_set
        print(f"Matches in top 10: {len(matches)} - {list(matches)[:3]}...")
        
        # Show popularity scores for top predictions in this subreddit
        subreddit_scores = subreddit_popularity.get(subreddit, {})
        top_3_predicted = results['all_predictions'][i][:3]
        print(f"Top 3 popularity scores in {subreddit}:")
        for song in top_3_predicted:
            score = subreddit_scores.get(song, 0.0)
            print(f"  {song}: {score:.6f}")
    
    # 9. Print subreddit popularity comparison
    print("\n" + "="*80)
    print("SUBREDDIT POPULARITY COMPARISON")
    print("="*80)
    
    for subreddit, popularity_scores in list(subreddit_popularity.items())[:3]:  # Show first 3 subreddits
        if popularity_scores:
            top_songs = sorted(popularity_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"\nTop 5 most popular songs in r/{subreddit}:")
            for song, score in top_songs:
                print(f"  {song}: {score:.6f}")
    
    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    main()