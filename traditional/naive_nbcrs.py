"""
Naive Nearest Neighbor-Based Collaborative Recommendation System (NBCRS)

This script implements a collaborative filtering approach for music recommendations
using Reddit queries. It finds similar queries within the same subreddit and uses
their ground truth music recommendations to generate predictions.

=== ALGORITHM ===
1. Load Reddit music queries with 100 candidate songs and ground truth recommendations
2. Generate embeddings using sentence-transformers/all-MiniLM-L6-v2
3. Group queries by subreddit (music taste varies by community)
4. For each query, find 2 most similar queries within same subreddit
5. Rank the 100 candidates based on relevance to similar queries' ground truth
6. Prioritize candidates that appear in similar queries' ground truth
7. Generate up to 100 ranked predictions per query
8. Evaluate using comprehensive ranking metrics
"""

import json
from pathlib import Path
import os
import random
import math
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import numpy as np
import wandb

# Import our utilities
from utils import (
    normalize_string, normalize_list_of_lists,
    calculate_comprehensive_metrics, calculate_subreddit_metrics,
    print_metrics_summary, print_subreddit_comparison
)

# ─── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_JSONL   = os.getenv("INPUT_JSONL", "")
OUTPUT_JSONL  = os.getenv("NBCRS_OUTPUT", "pipeline/language_models/results/FINAL_NAIVE_NBCRS.jsonl")
MODEL_NAME    = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE    = 64
MAX_PREDICTIONS = 100
RANDOM_SEED   = 42
# ────────────────────────────────────────────────────────────────────────────────

wandb.login()

def mean_pooling(model_output, attention_mask):
    """Average token embeddings, ignoring padding."""
    token_embeds   = model_output[0]  
    mask_expanded  = attention_mask.unsqueeze(-1).expand(token_embeds.size()).float()
    summed         = (token_embeds * mask_expanded).sum(1)
    counts         = mask_expanded.sum(1).clamp(min=1e-9)
    return summed / counts

def load_records(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records

def embed_queries(queries, tokenizer, model, device):
    batches = range(0, len(queries), BATCH_SIZE)
    all_embs = []
    for start in tqdm(batches, desc="Embedding queries"):
        batch_q = queries[start : start + BATCH_SIZE]
        encoded = tokenizer(
            batch_q, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            output = model(**encoded)
        pooled = mean_pooling(output, encoded["attention_mask"])
        normed = F.normalize(pooled, p=2, dim=1).cpu().numpy()
        all_embs.append(normed)
    return np.vstack(all_embs)

def main():
    # Set random seeds for reproducibility
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    wandb.init(
        project="reddit-music-naive-nbcrs",
        entity="musiCRS",
        name=f"naive_nbcrs_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "model_name": MODEL_NAME,
            "batch_size": BATCH_SIZE,
            "max_predictions": MAX_PREDICTIONS,
            "random_seed": RANDOM_SEED,
            "mode": "query_only"
        },
        tags=["naive-nbcrs", "music-recommendation", "query-only"]
    )
    
    results = {
        "all_predictions": [],
        "all_ground_truth": [],
        "all_subreddits": []
    }
    
    records     = load_records(INPUT_JSONL)
    queries     = [rec["query"] for rec in records]
    subreddits  = [rec.get("source_subreddit", "") for rec in records]
    candidates  = [rec.get("final_candidates", []) for rec in records]

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    embs = embed_queries(queries, tokenizer, model, device)

    group_indices = {}
    for idx, sub in enumerate(subreddits):
        group_indices.setdefault(sub, []).append(idx)

    for rec in records:
        rec["predicted"] = []

    for sub, idx_list in group_indices.items():
        n = len(idx_list)
        if n < 2:
            for global_idx in idx_list:
                current_candidates = candidates[global_idx]
                if current_candidates:
                    randomized_candidates = current_candidates.copy()
                    random.shuffle(randomized_candidates)
                    records[global_idx]["predicted"] = randomized_candidates[:MAX_PREDICTIONS]
                else:
                    records[global_idx]["predicted"] = []
            continue

        sub_embs = embs[idx_list]
        k = min(n, 10)  # want self + up to 9 neighbors
        nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(sub_embs)
        _, idxs = nn.kneighbors(sub_embs)

        for local_pos, global_idx in enumerate(idx_list):
            # local_pos → position within sub_embs
            neigh_locals = idxs[local_pos, 1:]  # skip self at position 0
            # map back to global indices
            neigh_globals = [idx_list[loc] for loc in neigh_locals]

            # Get ground truth from similar queries
            similar_gt = set()
            for neigh_idx in neigh_globals:
                gt = records[neigh_idx].get("limited_gt", 
                                          records[neigh_idx].get("combined_ground_truth",
                                                               records[neigh_idx].get("ground_truth", [])))
                similar_gt.update(gt)

            # Rank candidates based on relevance to similar queries' ground truth
            current_candidates = candidates[global_idx]
            if not current_candidates:
                records[global_idx]["predicted"] = []
                continue
                
            # Shuffle candidates to reduce order bias (similar to Qwen approach)
            randomized_candidates = current_candidates.copy()
            random.shuffle(randomized_candidates)
                
            # Simple ranking: prioritize candidates that appear in similar queries' ground truth
            ranked_candidates = []
            seen = set()
            
            # First, add candidates that are in similar ground truth
            for candidate in randomized_candidates:
                if len(ranked_candidates) >= MAX_PREDICTIONS:
                    break
                if candidate in similar_gt and candidate not in seen:
                    ranked_candidates.append(candidate)
                    seen.add(candidate)
            
            # Then add remaining candidates
            for candidate in randomized_candidates:
                if len(ranked_candidates) >= MAX_PREDICTIONS:
                    break
                if candidate not in seen:
                    ranked_candidates.append(candidate)
                    seen.add(candidate)

            records[global_idx]["predicted"] = ranked_candidates

    # 7. Write out updated JSONL
    out_path = Path(OUTPUT_JSONL)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as wf:
        for rec in records:
            wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            
            # Collect results for evaluation
            results["all_predictions"].append(rec["predicted"])
            results["all_ground_truth"].append(rec.get("limited_gt", 
                                                      rec.get("combined_ground_truth", 
                                                             rec.get("ground_truth", []))))
            results["all_subreddits"].append(rec.get("source_subreddit", "unknown"))

    print(f"✅ Saved {len(records)} records with up to {MAX_PREDICTIONS} predictions each to {out_path}")
    
    # Calculate comprehensive evaluation metrics
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
    
    # Log final evaluation metrics (with FINAL_ prefix for clarity)
    final_log_dict = {}
    for metric_name, metric_value in final_eval_metrics.items():
        final_log_dict[f"FINAL_{metric_name}"] = metric_value
    
    # Log per-subreddit metrics to wandb
    for subreddit, sub_metrics in subreddit_metrics.items():
        for metric_name, metric_value in sub_metrics.items():
            final_log_dict[f"SUBREDDIT_{subreddit}_{metric_name}"] = metric_value
    
    wandb.log(final_log_dict)

    # Print FINAL evaluation results
    if final_eval_metrics:
        print_metrics_summary(final_eval_metrics, "FINAL Evaluation Results for Naive NBCRS")
        
        # Print per-subreddit results
        if subreddit_metrics:
            print_subreddit_comparison(subreddit_metrics)
    
    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    main()
