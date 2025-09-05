def print_trainable_parameters(model, vb=0):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            if vb > 0:
                print(_, param.requires_grad, param.numel())
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

"""
Utility functions for music recommendation evaluation and processing.

This module contains common functionality used across different model implementations:
- Evaluation metrics (Recall@K, Precision@K, NDCG@K, MRR, etc.)
- String normalization for robust matching
- Audio processing utilities
- Statistical analysis with standard errors
- Per-subreddit evaluation grouping
"""

import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict

import librosa
import numpy as np
import soundfile as sf
import torch


# ===== STRING NORMALIZATION =====

def normalize_string(s: str) -> str:
    """Normalize string for comparison with proper Unicode quote handling"""
    # Handle various quote types and punctuation
    s = s.lower().strip()
    # Normalize Unicode curly quotes to standard quotes
    s = s.replace('\u2018', "'").replace('\u2019', "'")  # Left/right single quotes
    s = s.replace('\u201C', '"').replace('\u201D', '"')  # Left/right double quotes
    # Remove leading/trailing quotes and spaces
    s = s.strip('" \t')
    return s


def normalize_list_of_lists(lst: List[List[str]]) -> List[List[str]]:
    """Normalize lists of strings for evaluation"""
    return [[normalize_string(x) for x in row] for row in lst]


# ===== CORE EVALUATION METRICS =====

def recall_at_k(preds: List[List[str]], golds: List[List[str]], k: int) -> float:
    """Calculate Recall@K"""
    if not preds or not golds:
        return 0.0
    hits = 0
    for p, g in zip(preds, golds):
        if any(item in p[:k] for item in g):
            hits += 1
    return hits / len(golds)


def precision_at_k(preds: List[List[str]], golds: List[List[str]], k: int) -> float:
    """Calculate Precision@K"""
    if not preds or not golds:
        return 0.0
    total_hits = 0
    for p, g in zip(preds, golds):
        total_hits += sum(1 for item in p[:k] if item in g)
    return total_hits / (len(preds) * k)


def mrr(preds: List[List[str]], golds: List[List[str]]) -> float:
    """Calculate Mean Reciprocal Rank"""
    if not preds or not golds:
        return 0.0
    rr_sum = 0.0
    for p, g in zip(preds, golds):
        for rank, item in enumerate(p, start=1):
            if item in g:
                rr_sum += 1.0 / rank
                break
    return rr_sum / len(preds)


def ndcg_at_k(preds: List[List[str]], golds: List[List[str]], k: int) -> float:
    """Calculate NDCG@K"""
    if not preds or not golds:
        return 0.0
        
    def dcg_at_k(pred: List[str], gold: List[str], k: int) -> float:
        return sum((1.0 / math.log2(i+1))
                   for i, item in enumerate(pred[:k], start=1)
                   if item in gold)
    
    def idcg_at_k(gold: List[str], k: int) -> float:
        ideal_hits = min(len(gold), k)
        return sum((1.0 / math.log2(i+1)) for i in range(1, ideal_hits+1))
    
    total = 0.0
    for p, g in zip(preds, golds):
        idcg = idcg_at_k(g, k)
        total += (dcg_at_k(p, g, k) / idcg) if idcg > 0 else 0.0
    return total / len(preds)


def hit_rate_at_k(preds: List[List[str]], golds: List[List[str]], k: int = 10) -> float:
    """Calculate Hit Rate@K (percentage of queries with ≥1 relevant item in top-K)"""
    if not preds or not golds:
        return 0.0
    hits = sum(1 for p, g in zip(preds, golds) if any(item in p[:k] for item in g))
    return hits / len(preds)


def average_first_relevant_position(preds: List[List[str]], golds: List[List[str]]) -> float:
    """Calculate average position of first relevant item"""
    if not preds or not golds:
        return 0.0
    
    first_relevant_positions = []
    for p, g in zip(preds, golds):
        for i, item in enumerate(p):
            if item in g:
                first_relevant_positions.append(i + 1)
                break
        else:
            first_relevant_positions.append(len(p) + 1)  # Not found
    
    return np.mean(first_relevant_positions)


# ===== COMPREHENSIVE METRICS CALCULATION =====

def calculate_comprehensive_metrics(
    all_preds: List[List[str]], 
    all_golds: List[List[str]], 
    k_values: List[int] = [1, 5, 10, 20]
) -> Dict[str, float]:
    """Calculate comprehensive metrics with statistical rigor including standard errors"""
    if not all_preds or not all_golds:
        return {}
    
    # Normalize data
    preds_norm = normalize_list_of_lists(all_preds)
    golds_norm = normalize_list_of_lists(all_golds)
    
    metrics = {}
    n_samples = len(preds_norm)
    
    # Calculate per-query metrics for SE computation
    for k in k_values:
        # Calculate per-query recall, precision, ndcg
        per_query_recall = []
        per_query_precision = []
        per_query_ndcg = []
        
        for p, g in zip(preds_norm, golds_norm):
            # Per-query recall@k
            hits = sum(1 for item in p[:k] if item in g)
            per_query_recall.append(hits / len(g) if g else 0.0)
            
            # Per-query precision@k
            per_query_precision.append(hits / k)
            
            # Per-query NDCG@k
            if g:
                def dcg_at_k_single(pred: List[str], gold: List[str], k: int) -> float:
                    return sum((1.0 / math.log2(i+1))
                              for i, item in enumerate(pred[:k], start=1)
                              if item in gold)
                
                def idcg_at_k_single(gold: List[str], k: int) -> float:
                    ideal_hits = min(len(gold), k)
                    return sum((1.0 / math.log2(i+1)) for i in range(1, ideal_hits+1))
                
                idcg = idcg_at_k_single(g, k)
                per_query_ndcg.append((dcg_at_k_single(p, g, k) / idcg) if idcg > 0 else 0.0)
            else:
                per_query_ndcg.append(0.0)
        
        # Store means and standard errors
        metrics[f"recall_at_{k}"] = float(np.mean(per_query_recall))
        metrics[f"precision_at_{k}"] = float(np.mean(per_query_precision))
        metrics[f"ndcg_at_{k}"] = float(np.mean(per_query_ndcg))
        
        # Standard errors using sample standard deviation (ddof=1)
        if n_samples > 1:
            metrics[f"recall_at_{k}_se"] = float(np.std(per_query_recall, ddof=1) / math.sqrt(n_samples))
            metrics[f"precision_at_{k}_se"] = float(np.std(per_query_precision, ddof=1) / math.sqrt(n_samples))
            metrics[f"ndcg_at_{k}_se"] = float(np.std(per_query_ndcg, ddof=1) / math.sqrt(n_samples))
    
    # Per-query MRR
    per_query_mrr = []
    for p, g in zip(preds_norm, golds_norm):
        for rank, item in enumerate(p, start=1):
            if item in g:
                per_query_mrr.append(1.0 / rank)
                break
        else:
            per_query_mrr.append(0.0)
    
    metrics["mrr"] = float(np.mean(per_query_mrr))
    if n_samples > 1:
        metrics["mrr_se"] = float(np.std(per_query_mrr, ddof=1) / math.sqrt(n_samples))
    
    # Per-query hit rate@10
    per_query_hit = [1.0 if any(item in p[:10] for item in g) else 0.0 
                     for p, g in zip(preds_norm, golds_norm)]
    metrics["hit_rate"] = float(np.mean(per_query_hit))
    if n_samples > 1:
        metrics["hit_rate_se"] = float(np.std(per_query_hit, ddof=1) / math.sqrt(n_samples))
    
    # Average position of first relevant item
    first_relevant_positions = []
    for p, g in zip(preds_norm, golds_norm):
        for i, item in enumerate(p):
            if item in g:
                first_relevant_positions.append(i + 1)
                break
        else:
            first_relevant_positions.append(len(p) + 1)  # Not found
    
    metrics["avg_first_relevant_position"] = float(np.mean(first_relevant_positions))
    if n_samples > 1:
        metrics["avg_first_relevant_position_se"] = float(np.std(first_relevant_positions, ddof=1) / math.sqrt(n_samples))
    
    metrics["num_samples"] = n_samples
    
    return metrics


# ===== SUBREDDIT-SPECIFIC EVALUATION =====

def calculate_subreddit_metrics(
    all_preds: List[List[str]], 
    all_golds: List[List[str]], 
    all_subreddits: List[str], 
    k_values: List[int] = [1, 5, 10, 20]
) -> Dict[str, Dict[str, float]]:
    """Calculate metrics grouped by subreddit with remapping"""
    if not all_preds or not all_golds or not all_subreddits:
        return {}
    
    # Normalize data
    preds_norm = normalize_list_of_lists(all_preds)
    golds_norm = normalize_list_of_lists(all_golds)
    
    # Remap subreddits if needed (consolidate small communities)
    SUB_MAPPING = {
        "postrock": "indieheads",
        "ambientmusic": "jazz",
    }
    
    # Group by subreddit with remapping
    subreddit_groups = defaultdict(lambda: {"preds": [], "golds": []})
    for p, g, sub in zip(preds_norm, golds_norm, all_subreddits):
        mapped_sub = SUB_MAPPING.get(sub.lower(), sub.lower())
        subreddit_groups[mapped_sub]["preds"].append(p)
        subreddit_groups[mapped_sub]["golds"].append(g)
    
    # Calculate metrics for each subreddit
    subreddit_metrics = {}
    for subreddit, data in subreddit_groups.items():
        if data["preds"] and data["golds"]:
            sub_metrics = calculate_comprehensive_metrics(
                data["preds"], 
                data["golds"], 
                k_values
            )
            subreddit_metrics[subreddit] = sub_metrics
    
    return subreddit_metrics


# ===== AUDIO PROCESSING UTILITIES =====

def load_and_mix_audio(
    wav_paths: List[Path], 
    sr: int = 16000, 
    max_clips: int = 10, 
    audio_budget_secs: float = 300.0
) -> np.ndarray:
    """
    Load and mix multiple audio clips within a time budget.
    
    Args:
        wav_paths: List of paths to audio files
        sr: Target sampling rate
        max_clips: Maximum number of clips to use
        audio_budget_secs: Total time budget in seconds
        
    Returns:
        Mixed audio as numpy array
    """
    if not wav_paths:
        # Return silence placeholder
        return np.zeros(sr, dtype=np.float32)

    num_available = len(wav_paths)
    num_clips_to_use = min(num_available, max_clips)
    clip_duration = audio_budget_secs / num_clips_to_use if num_clips_to_use > 0 else 30.0
    
    clips = []
    selected_paths = random.sample(wav_paths, k=num_clips_to_use)
    
    for path in selected_paths:
        try:
            info = sf.info(str(path))
            total_secs = info.frames / info.samplerate if info.samplerate else 0.0
            start_sec = 0.0
            
            if total_secs > clip_duration:
                start_sec = random.uniform(0, max(0.0, total_secs - clip_duration))
                
            wav, _ = librosa.load(str(path), sr=sr, offset=start_sec, duration=clip_duration)
            
            # Handle stereo files
            if wav.ndim > 1:
                wav = wav[:, 0]
                
            clips.append(wav.astype(np.float32))
            
        except Exception as e:
            print(f"Error loading audio {path}: {e}")
            continue

    if not clips:
        return np.zeros(sr, dtype=np.float32)

    # Concatenate clips
    mixed = np.concatenate(clips, axis=0)
    
    # Ensure we don't exceed the budget and have minimum length
    max_len = int(sr * audio_budget_secs)
    if len(mixed) > max_len:
        mixed = mixed[:max_len]
    if len(mixed) < sr:  # Ensure >= 1s for audio processing
        pad = np.zeros(sr - len(mixed), dtype=np.float32)
        mixed = np.concatenate([mixed, pad], axis=0)
    
    return mixed


def save_temp_audio(audio: np.ndarray, sr: int = 16000) -> str:
    """Save audio array to temporary file and return path"""
    import time
    import tempfile
    import os
    
    temp_path = os.path.join(tempfile.gettempdir(), f"temp_audio_{int(time.time()*1000)}.wav")
    sf.write(temp_path, audio, sr)
    return temp_path


# ===== PROMPT BUILDING UTILITIES =====

def build_ranking_prompt(
    convo: str,
    candidates: List[str],
    mode: str,
    shuffle_candidates: bool = True
) -> Tuple[str, List[str]]:
    """
    Build a standardized ranking prompt across different models.
    
    Args:
        convo: Query/conversation text
        candidates: List of candidate song names
        mode: Input mode ("query_only", "audio_only", "audio_query")
        shuffle_candidates: Whether to randomize candidate order (reduces bias)
        
    Returns:
        Tuple of (prompt_text, randomized_candidates)
    """
    # Randomize candidate order to reduce bias
    if shuffle_candidates:
        randomized_candidates = candidates.copy()
        random.shuffle(randomized_candidates)
    else:
        randomized_candidates = candidates.copy()

    # Build prompt components
    parts = []
    
    # Add query/conversation if needed
    if mode in ("audio_query", "query_only") and convo:
        parts.append(convo.strip())
    

    header = f"Here are {len(randomized_candidates)} candidate songs; rank by relevance:"
    parts.append(header)
    for candidate in randomized_candidates:
        parts.append(f"- {candidate}")
    
    # Add instruction
    instr = (
        f"Rank these {len(randomized_candidates)} songs from most to least relevant based on the audio and query. "
        "Output ONLY the exact song titles separated by commas. "
        "Example format: Song Title A, Song Title B, Song Title C"
    )
    parts.append(instr)
    
    prompt_text = "\n".join(parts)
    return prompt_text, randomized_candidates


# ===== RESULT PROCESSING UTILITIES =====

def parse_and_filter_ranking(
    model_output: str, 
    candidates: List[str],
    fill_missing: bool = True
) -> List[str]:
    """
    Parse model output and filter to valid candidates.
    
    Args:
        model_output: Raw text output from model
        candidates: List of valid candidate names
        fill_missing: Whether to add missing candidates at the end
        
    Returns:
        Filtered and processed ranking list
    """
    # Parse comma-separated output
    raw_ranked = [s.strip() for s in model_output.split(",") if s.strip()]
    seen = set(raw_ranked)
    for c in candidates:  # Use original order for missing items
        if c not in seen:
            raw_ranked.append(c)

    # Finally truncate to exactly match candidates length
    return raw_ranked[:len(candidates)]    
    # Create normalized mapping from candidates to original candidates
    # candidate_map = {normalize_string(c): c for c in candidates}
    
    # # Map model outputs to actual candidates, drop non-matching items
    # mapped_ranked = []
    # seen_candidates = set()
    
    # for item in raw_ranked:
    #     normalized_item = normalize_string(item)
    #     if normalized_item in candidate_map:
    #         actual_candidate = candidate_map[normalized_item]
    #         if actual_candidate not in seen_candidates:
    #             mapped_ranked.append(actual_candidate)
    #             seen_candidates.add(actual_candidate)
    
    # # Fill in missing candidates if requested
    # if fill_missing:
    #     for candidate in candidates:
    #         if candidate not in seen_candidates:
    #             mapped_ranked.append(candidate)
    
    # # Return exactly the number of candidates requested
    # return mapped_ranked[:len(candidates)]


# ===== DISPLAY UTILITIES =====

def print_metrics_summary(metrics: Dict[str, float], title: str = "Metrics Summary") -> None:
    """Print formatted metrics summary"""
    print(f"\n{title}")
    print("-" * len(title))
    
    # Print main metrics
    k_values = [1, 5, 10, 20]
    for k in k_values:
        recall_key = f"recall_at_{k}"
        precision_key = f"precision_at_{k}"
        ndcg_key = f"ndcg_at_{k}"
        
        if all(key in metrics for key in [recall_key, precision_key, ndcg_key]):
            print(f"  R@{k}={metrics[recall_key]:.4f} "
                  f"P@{k}={metrics[precision_key]:.4f} "
                  f"nDCG@{k}={metrics[ndcg_key]:.4f}")
    
    # Print additional metrics
    if "mrr" in metrics:
        print(f"  MRR={metrics['mrr']:.4f}")
    if "hit_rate" in metrics:
        print(f"  Hit Rate@10={metrics['hit_rate']:.4f}")
    if "avg_first_relevant_position" in metrics:
        print(f"  Avg First Relevant Position={metrics['avg_first_relevant_position']:.2f}")
    if "num_samples" in metrics:
        print(f"  Samples: {metrics['num_samples']}")


def print_subreddit_comparison(subreddit_metrics: Dict[str, Dict[str, float]]) -> None:
    """Print per-subreddit metrics comparison"""
    if not subreddit_metrics:
        return
        
    print("\nPer-Subreddit Results:")
    print("-" * 40)
    
    for subreddit in sorted(subreddit_metrics.keys()):
        sub_metrics = subreddit_metrics[subreddit]
        print(f"\n{subreddit.upper():20s} (n={sub_metrics.get('num_samples', 0)})")
        
        for k in [1, 5, 10, 20]:
            recall_key = f"recall_at_{k}"
            precision_key = f"precision_at_{k}"
            ndcg_key = f"ndcg_at_{k}"
            
            if all(key in sub_metrics for key in [recall_key, precision_key, ndcg_key]):
                print(f"  R@{k}={sub_metrics[recall_key]:.4f} "
                      f"P@{k}={sub_metrics[precision_key]:.4f} "
                      f"nDCG@{k}={sub_metrics[ndcg_key]:.4f}")
        
        if "mrr" in sub_metrics:
            print(f"  MRR={sub_metrics['mrr']:.4f}")


# Backward compatibility aliases (for existing code)
calculate_incremental_metrics = calculate_comprehensive_metrics
