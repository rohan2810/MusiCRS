"""
Music Recommendation System using Qwen2-Audio for Multimodal Ranking

This script implements a comprehensive music recommendation system that leverages the Qwen2-Audio
multimodal model to rank candidate songs based on audio content and textual queries from Reddit
music discussion posts.

=== OVERVIEW ===
The system supports multiple ranking approaches:
1. **Reranking Method**: Scores each candidate individually using log-likelihood
2. **Generative Method**: Generates a ranked list directly via text generation
3. **Multimodal Support**: Can use audio-only, text-only, or audio+text inputs
4. **Description Enhancement**: Optionally includes song descriptions for better context

=== RANKING METHODS ===

**Reranking Approach:**
- Processes each candidate song individually
- Calculates log-likelihood scores for each candidate given the context
- Uses masked language modeling to score candidate relevance
- Randomizes candidate presentation order to reduce order bias
- Returns candidates sorted by likelihood scores

**Generative Approach:**
- Presents all candidates to the model simultaneously
- Asks the model to generate a comma-separated ranked list
- Uses order randomization and bullet points to mitigate order bias
- Employs improved prompting strategies for better instruction following
- Post-processes output to ensure all candidates are included

=== MULTIMODAL INPUT MODES ===
- **query_only**: Text query from Reddit post only
- **audio_query**: Both audio clips + text query (multimodal)
- **audio_only**: Audio clips only, no text context

=== EVALUATION FRAMEWORK ===
The system tracks comprehensive metrics with statistical rigor:

**Core Ranking Metrics:**
- Recall@K (K=1,5,10,20): Fraction of relevant items found in top-K
- Precision@K: Fraction of top-K that are relevant  
- NDCG@K: Normalized Discounted Cumulative Gain (position-aware quality)
- MRR: Mean Reciprocal Rank of first relevant item

**Additional Music-Specific Metrics:**
- Hit Rate: Percentage of queries with ≥1 relevant item in top-10
- Avg First Relevant Position: Average rank of first correct recommendation

**Statistical Analysis:**
- Standard Errors (SE) for all metrics using per-query variance
- Confidence intervals for robust method comparison
- Sample size tracking for reliability assessment

=== EXPERIMENTAL DESIGN ===
The system runs controlled experiments across multiple dimensions:

**Method Combinations:**
- ranking_method: ["reranking", "generative"]
- mode: ["query_only", "audio_query", "audio_only"] 
- descriptions: [True, False]

**Wandb Integration:**
- Individual runs for each method combination
- Real-time incremental metrics (logged every 10 samples)
- Final aggregated results with FINAL_ prefix
- Comprehensive config tracking and experiment organization

=== AUDIO PROCESSING ===
- Smart audio budget allocation (5min total per query)
- Random temporal sampling from available audio clips
- Proper sampling rate handling for Qwen2-Audio (16kHz)
- Robust error handling for corrupted/missing audio files

=== BIAS MITIGATION ===
- Candidate order randomization to prevent position bias
- Bullet-point formatting instead of numbered lists
- Explicit instruction formatting with examples
- Separate scoring of original candidate set vs. randomized presentation

=== DATA PIPELINE ===
Input: Reddit music posts with ground truth recommendations
Output: Ranked candidate lists + comprehensive evaluation metrics
Ground Truth: Community-validated relevant songs from comment discussions

This implementation enables rigorous comparison of multimodal vs. text-only approaches,
different ranking paradigms, and the impact of additional context (descriptions) on
music recommendation quality.
"""

import os
import json
import librosa
import soundfile as sf
import torch
import tqdm
import numpy as np
from transformers import AutoProcessor
from transformers import Qwen2AudioForConditionalGeneration
from pathlib import Path
import random
import torch.nn.functional as F
import wandb
import time
import math
from datetime import datetime
from collections import defaultdict

wandb.login(key='d13ff15cc78423826ede4beef6d5e6f91a7cc50e')

MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"   
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
AUDIO_BASE_DIR  = Path("/deepfreeze/junda/abhay/audio_yt/full/wav")
SUM_FILE = Path("/home/junda/rohan/reddit-music/pipeline/.enhanced_cache/enhanced_summaries.json")
with SUM_FILE.open("r", encoding="utf-8") as f:
    summaries = json.load(f)

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Qwen2AudioForConditionalGeneration.from_pretrained(MODEL_ID, 
                                                           device_map="auto",
                                                           dtype=torch.bfloat16)
model.eval()


# ===== EVALUATION METRICS (from evaluate_results.py) =====
def recall_at_k(preds, golds, k):
    """Calculate Recall@K"""
    if not preds or not golds:
        return 0.0
    hits = 0
    for p, g in zip(preds, golds):
        if any(item in p[:k] for item in g):
            hits += 1
    return hits / len(golds)

def precision_at_k(preds, golds, k):
    """Calculate Precision@K"""
    if not preds or not golds:
        return 0.0
    total_hits = 0
    for p, g in zip(preds, golds):
        total_hits += sum(1 for item in p[:k] if item in g)
    return total_hits / (len(preds) * k)

def mrr(preds, golds):
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

def ndcg_at_k(preds, golds, k):
    """Calculate NDCG@K"""
    if not preds or not golds:
        return 0.0
        
    def dcg_at_k(pred, gold, k):
        return sum((1.0 / math.log2(i+1))
                   for i, item in enumerate(pred[:k], start=1)
                   if item in gold)
    
    def idcg_at_k(gold, k):
        ideal_hits = min(len(gold), k)
        return sum((1.0 / math.log2(i+1)) for i in range(1, ideal_hits+1))
    
    total = 0.0
    for p, g in zip(preds, golds):
        idcg = idcg_at_k(g, k)
        total += (dcg_at_k(p, g, k) / idcg) if idcg > 0 else 0.0
    return total / len(preds)

def normalize_string(s):
    """Normalize string for comparison"""
    return s.lower().strip('" ').replace("'", "'")

def normalize_list_of_lists(lst):
    """Normalize lists of strings"""
    return [[normalize_string(x) for x in row] for row in lst]

def calculate_incremental_metrics(all_preds, all_golds, k_values=[1, 5, 10, 20]):
    """Calculate metrics for current predictions"""
    if not all_preds or not all_golds:
        return {}
    
    # Normalize data
    preds_norm = normalize_list_of_lists(all_preds)
    golds_norm = normalize_list_of_lists(all_golds)
    
    metrics = {}
    n_samples = len(preds_norm)
    
    # Calculate per-query metrics for SE computation
    per_query_metrics = {}
    
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
                def dcg_at_k_single(pred, gold, k):
                    return sum((1.0 / math.log2(i+1))
                              for i, item in enumerate(pred[:k], start=1)
                              if item in gold)
                
                def idcg_at_k_single(gold, k):
                    ideal_hits = min(len(gold), k)
                    return sum((1.0 / math.log2(i+1)) for i in range(1, ideal_hits+1))
                
                idcg = idcg_at_k_single(g, k)
                per_query_ndcg.append((dcg_at_k_single(p, g, k) / idcg) if idcg > 0 else 0.0)
            else:
                per_query_ndcg.append(0.0)
        
        # Store means and standard errors
        metrics[f"recall_at_{k}"] = np.mean(per_query_recall)
        metrics[f"precision_at_{k}"] = np.mean(per_query_precision)
        metrics[f"ndcg_at_{k}"] = np.mean(per_query_ndcg)
        
        # Standard errors
        if n_samples > 1:
            metrics[f"recall_at_{k}_se"] = np.std(per_query_recall) / math.sqrt(n_samples)
            metrics[f"precision_at_{k}_se"] = np.std(per_query_precision) / math.sqrt(n_samples)
            metrics[f"ndcg_at_{k}_se"] = np.std(per_query_ndcg) / math.sqrt(n_samples)
    
    # Per-query MRR
    per_query_mrr = []
    for p, g in zip(preds_norm, golds_norm):
        for rank, item in enumerate(p, start=1):
            if item in g:
                per_query_mrr.append(1.0 / rank)
                break
        else:
            per_query_mrr.append(0.0)
    
    metrics["mrr"] = np.mean(per_query_mrr)
    if n_samples > 1:
        metrics["mrr_se"] = np.std(per_query_mrr) / math.sqrt(n_samples)
    
    # Per-query hit rate
    per_query_hit = [1.0 if any(item in p[:10] for item in g) else 0.0 
                     for p, g in zip(preds_norm, golds_norm)]
    metrics["hit_rate"] = np.mean(per_query_hit)
    if n_samples > 1:
        metrics["hit_rate_se"] = np.std(per_query_hit) / math.sqrt(n_samples)
    
    # Average position of first relevant item
    first_relevant_positions = []
    for p, g in zip(preds_norm, golds_norm):
        for i, item in enumerate(p):
            if item in g:
                first_relevant_positions.append(i + 1)
                break
        else:
            first_relevant_positions.append(len(p) + 1)  # Not found
    
    metrics["avg_first_relevant_position"] = np.mean(first_relevant_positions)
    if n_samples > 1:
        metrics["avg_first_relevant_position_se"] = np.std(first_relevant_positions) / math.sqrt(n_samples)
    
    metrics["num_samples"] = n_samples
    
    return metrics


def rank_songs_reranking(
    audio_paths: list[str],
    convo: str,
    candidates: list[str],
    mode: str = "audio_query",
    max_clips: int = 10,
    audio_budget_secs: float = 300.0,  # 5 minutes total budget
    include_descriptions: bool = True,
) -> list[str]:
    
    sr = processor.feature_extractor.sampling_rate
    clips: list[np.ndarray] = []
    if mode in ("audio_query", "audio_only"):
        # Smart audio budget allocation
        num_available = len(audio_paths)
        num_clips_to_use = min(num_available, max_clips)
        clip_duration = audio_budget_secs / num_clips_to_use if num_clips_to_use > 0 else 30.0
        
        print(f"Audio budget: {audio_budget_secs}s, Available clips: {num_available}, "
              f"Using: {num_clips_to_use} clips, {clip_duration:.1f}s each")
        
        # Comment out real audio loading for testing
        sel = random.sample(audio_paths, k=num_clips_to_use)
        for p in sel:
            try:
                info = sf.info(p)
                total_secs = info.frames / info.samplerate
                start_sec = random.uniform(0, max(0, total_secs - clip_duration))
                wav, _ = librosa.load(p, sr=sr, offset=start_sec, duration=clip_duration)
                clips.append(wav)
            except Exception as e:
                print(f"Error loading audio {p}: {e}")
                continue
        
        # Generate white noise instead of loading real audio
        # for _ in range(num_clips_to_use):
        #     # Generate white noise with the same duration as would be loaded
        #     num_samples = int(sr * clip_duration)
        #     white_noise = np.random.normal(0, 0.1, num_samples).astype(np.float32)
        #     clips.append(white_noise)


    sys_msg = {"role": "system", "content": "You are a helpful music recommender."}

    user_content: list[dict] = []
    if mode in ("audio_query", "audio_only"):
        user_content += [{"type": "audio", "audio": c} for c in clips]
    if mode in ("audio_query", "query_only"):
        user_content.append({"type": "text", "text": convo})

    user_q = {"role": "user", "content": user_content}


    # Randomize candidate order for reranking to avoid order bias  
    randomized_candidates = candidates.copy()
    random.shuffle(randomized_candidates)
    
    if include_descriptions:
        lines = ["Here are the candidate song names with descriptions; rank most relevant first based on the query:"]
        lines += [
            f"- {c}: {summaries.get(c, 'No description available.')[:200].strip()}"
            for c in randomized_candidates
        ]
    else:
        lines = ["Here are the candidate song names; rank most relevant first based on the query:"]
        lines += [f"- {c}" for c in randomized_candidates]
    cand_list = {"role": "user", "content": [{"type": "text", "text": "\n".join(lines)}]}

    prefix_msgs = [sys_msg, user_q, cand_list]

    prefix_inputs = (
        processor(
            text=processor.apply_chat_template(prefix_msgs, add_generation_prompt=False, tokenize=False),
            audio=clips,
            return_tensors="pt",
            padding=True,
            sampling_rate=sr
        )
        if clips
        else processor(
            text=processor.apply_chat_template(prefix_msgs, add_generation_prompt=False, tokenize=False),
            return_tensors="pt",
            padding=True,
        )
    )

    for k, v in prefix_inputs.items():
        prefix_inputs[k] = v.to(model.device)

    prompt_len = prefix_inputs["input_ids"].size(1)
    has_audio = "input_features" in prefix_inputs

    scores = {}
    model.eval()
    with torch.no_grad():
        # Score each candidate individually
        for cand in candidates:  # Still score all original candidates
            final_msg = {"role": "user", "content": [{"type": "text", "text": cand}]}
            full_msgs = prefix_msgs + [final_msg]
            full_text = processor.apply_chat_template(full_msgs, add_generation_prompt=False, tokenize=False)
            inp = (
                processor(
                    text=full_text,
                    audio=clips,
                    return_tensors="pt",
                    padding=True,
                    sampling_rate=sr
                )
                if clips
                else processor(
                    text=full_text,
                    return_tensors="pt",
                    padding=True,
                )
            )
            for k, v in inp.items():
                inp[k] = v.to(model.device)

            # forward pass - use audio features from inp (not prefix_inputs) for proper alignment
            if has_audio and "input_features" in inp:
                out = model(
                    input_ids=inp["input_ids"],
                    attention_mask=inp["attention_mask"],
                    input_features=inp["input_features"],
                    feature_attention_mask=inp["feature_attention_mask"],
                    return_dict=True,
                )
            else:
                # text-only: skip audio args
                out = model(
                    input_ids=inp["input_ids"],
                    attention_mask=inp["attention_mask"],
                    return_dict=True,
                )

            logits = out.logits
            labels = inp["input_ids"].clone()
            labels[:, :prompt_len] = -100

            logps = F.log_softmax(logits, dim=-1)
            gather_labels = labels.clone()
            gather_labels[gather_labels < 0] = 0
            tok_logps = logps.gather(-1, gather_labels.unsqueeze(-1)).squeeze(-1)
            mask = (labels != -100).float()
            sum_logp = (tok_logps * mask).sum(dim=1)

            scores[cand] = sum_logp.item()

    # Return ranked by scores (order bias is reduced by randomized context presentation)
    return sorted(candidates, key=lambda c: scores[c], reverse=True)


def rank_songs_generative(
    audio_paths: list[str],
    convo: str,
    candidates: list[str],
    mode: str = "audio_query",
    max_clips: int = 10,
    audio_budget_secs: float = 300.0,  # 5 minutes total budget
    include_descriptions: bool = True,
) -> list[str]:
    
    sr = processor.feature_extractor.sampling_rate
    clips: list[np.ndarray] = []
    
    if mode in ("audio_query", "audio_only"):
        # Smart audio budget allocation
        num_available = len(audio_paths)
        num_clips_to_use = min(num_available, max_clips)
        clip_duration = audio_budget_secs / num_clips_to_use if num_clips_to_use > 0 else 30.0
        
        print(f"Audio budget: {audio_budget_secs}s, Available clips: {num_available}, "
              f"Using: {num_clips_to_use} clips, {clip_duration:.1f}s each")
        
        # Comment out real audio loading for testing
        sel = random.sample(audio_paths, k=num_clips_to_use)
        for p in sel:
            try:
                info = sf.info(p)
                total_secs = info.frames / info.samplerate
                start_sec = random.uniform(0, max(0, total_secs - clip_duration))
                wav, _ = librosa.load(p, sr=sr, offset=start_sec, duration=clip_duration)
                clips.append(wav)
            except Exception as e:
                print(f"Error loading audio {p}: {e}")
                continue
        
        # Generate white noise instead of loading real audio
        # for _ in range(num_clips_to_use):
        #     # Generate white noise with the same duration as would be loaded
        #     num_samples = int(sr * clip_duration)
        #     white_noise = np.random.normal(0, 0.1, num_samples).astype(np.float32)
        #     clips.append(white_noise)

    # KEY FIX: Randomize candidate order to avoid order bias
    original_candidates = candidates.copy()
    randomized_candidates = candidates.copy()
    random.shuffle(randomized_candidates)

    sys_msg = {"role": "system", "content": "You are a helpful music recommendation expert. Rank songs based on relevance to the audio and query."}
    
    user_content: list[dict] = []
    if mode in ("audio_query", "audio_only"):
        user_content += [{"type": "audio", "audio": c} for c in clips]
    if mode in ("audio_query", "query_only"):
        user_content.append({"type": "text", "text": convo})
    
    user_q = {"role": "user", "content": user_content}

    # Use randomized candidates and avoid numbered lists to reduce order bias
    if include_descriptions:
        header = f"Here are {len(randomized_candidates)} candidate songs with descriptions; rank by relevance:"
        lines = [header] + [
            f"- {c}: {summaries.get(c, 'No description available.')[:200].strip()}"
            for c in randomized_candidates
        ]
    else:
        header = f"Here are {len(randomized_candidates)} candidate songs; rank by relevance:"
        lines = [header] + [f"- {c}" for c in randomized_candidates]

    cand_blob = "\n".join(lines)

    # Improved instruction with clearer format specification
    instr = (
        f"Rank these {len(randomized_candidates)} songs from most to least relevant based on the audio and query. "
        "Output ONLY the exact song titles separated by commas. "
        "Example format: Song Title A, Song Title B, Song Title C"
    )

    cand_msg = {"role": "user", "content": [{"type": "text", "text": cand_blob}]}
    instr_msg = {"role": "user", "content": [{"type": "text", "text": instr}]}

    messages = [sys_msg, user_q, cand_msg, instr_msg]

    prompt = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    model_inputs = (
        processor(
            text=prompt,
            audio=clips,
            return_tensors="pt",
            padding=True,
            sampling_rate=sr
        )
        if clips
        else processor(
            text=prompt,
            return_tensors="pt",
            padding=True,
        )
    )
    
    # Move to device
    for k, v in model_inputs.items():
        model_inputs[k] = v.to(model.device)

    with torch.no_grad():
        out_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None
        )
    gen_ids = out_ids[:, model_inputs["input_ids"].size(1):]

    text = processor.batch_decode(
        gen_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )[0].strip()
    ranked = [s.strip() for s in text.split(",") if s.strip()]

    # Sometimes the model might omit a few or reorder; fill in missing from original candidates
    seen = set(ranked)
    for c in original_candidates:  # Use original order for missing items
        if c not in seen:
            ranked.append(c)

    # Finally truncate to exactly match candidates length
    return ranked[:len(candidates)]


def main():
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    
    input_jsonl = "/home/junda/rohan/reddit-music/data/merged_final_cleaned_music_clean_queries_with_candidates_enhanced_v2_100_candidates.jsonl"
    
    # Configuration options -- best: generative (no desc)
    modes = ["audio_query"]  # Options: ["query_only", "audio_query", "audio_only"] ,"audio_query"
    ranking_methods = ["generative"]  # Options: ["reranking", "generative"]
    descriptions_options = [False]  # Test both with and without descriptions
    num_samples = -1  # Number of samples to process (-1 for all samples)
    
    with open(input_jsonl, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    all_method_results = {}

    for ranking_method in ranking_methods:
        for mode in modes:
            for descriptions in descriptions_options:
                method_key = f"{ranking_method}_{mode}_desc_{descriptions}"
                
                # Initialize separate wandb run for each method combination

                wandb.init(
                    project="reddit-music-qwen-audio",
                    name=f"{method_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    config={
                        "model_id": MODEL_ID,
                        "ranking_method": ranking_method,
                        "mode": mode,
                        "include_descriptions": descriptions,
                        "num_samples": num_samples,
                        "random_seed": 42,
                    },
                    tags=["qwen2-audio", "music-ranking", ranking_method, mode, f"descriptions_{descriptions}"]
                )
                
                method_results = {
                    "all_predictions": [],
                    "all_ground_truth": []
                }
                
                print(f"Processing {mode} mode with {ranking_method} method (descriptions: {descriptions})")
                output_jsonl = f"/home/junda/rohan/reddit-music/generative/results/NEW_QWEN_AUDIO/{ranking_method}_{mode}_descriptions_{descriptions}_100_candidates.jsonl"
                output_dir = os.path.dirname(output_jsonl)
                os.makedirs(output_dir, exist_ok=True)

                with open(input_jsonl, "r", encoding="utf-8") as rf, \
                     open(output_jsonl, "w", encoding="utf-8") as wf:

                    # Load samples - use all if num_samples is -1
                    if num_samples == -1:
                        lines = rf.readlines()
                    else:
                        lines = rf.readlines()[:num_samples]
                    
                    for i, line in enumerate(tqdm.tqdm(lines, desc=f"Processing ({method_key})")):
                        try:
                            ex = json.loads(line)
                            sid = ex["name"]
                            convo = ex["query"]
                            candidates = ex.get("final_candidates", [])
                            ground_truth = ex.get("limited_gt", [])
                            audio_dir = AUDIO_BASE_DIR / sid
                            wav_paths = sorted(audio_dir.glob("*.wav"))
                            
                            if not wav_paths or not candidates:
                                continue

                            # Choose ranking method
                            if ranking_method == "reranking":
                                ranked = rank_songs_reranking(
                                    audio_paths=wav_paths,
                                    convo=convo,
                                    candidates=candidates,
                                    mode=mode,
                                    include_descriptions=descriptions
                                )
                            elif ranking_method == "generative":
                                ranked = rank_songs_generative(
                                    audio_paths=wav_paths,
                                    convo=convo,
                                    candidates=candidates,
                                    mode=mode,
                                    include_descriptions=descriptions
                                )
                            else:
                                raise ValueError(f"Unknown ranking method: {ranking_method}")

                            record = ex.copy()
                            record["predicted"] = ranked
                            wf.write(json.dumps(record, ensure_ascii=False) + "\n")
                            
                            # Collect for final evaluation
                            method_results["all_predictions"].append(ranked)
                            method_results["all_ground_truth"].append(ground_truth)
                            
                            # Log incremental evaluation metrics every 10 samples
                            if (i + 1) % 10 == 0:
                                eval_metrics = calculate_incremental_metrics(
                                    method_results["all_predictions"],
                                    method_results["all_ground_truth"]
                                )
                                
                                # Log only evaluation metrics (no method prefix needed since separate runs)
                                wandb.log(eval_metrics)
                        
                        except Exception as e:
                            print(f"Error processing sample {i}: {e}")
                            continue

                # Calculate FINAL evaluation metrics
                final_eval_metrics = calculate_incremental_metrics(
                    method_results["all_predictions"],
                    method_results["all_ground_truth"]
                )
                
                # Log final evaluation metrics (with FINAL_ prefix for clarity)
                final_log_dict = {}
                for metric_name, metric_value in final_eval_metrics.items():
                    final_log_dict[f"FINAL_{metric_name}"] = metric_value
                
                wandb.log(final_log_dict)

                print(f"Wrote {ranking_method}_{mode} results to {output_jsonl}")
                
                # Print FINAL evaluation results
                if final_eval_metrics:
                    print(f"FINAL Evaluation Results for {method_key}:")
                    for k in [1, 5, 10, 20]:
                        if f"recall_at_{k}" in final_eval_metrics:
                            print(f"  R@{k}={final_eval_metrics[f'recall_at_{k}']:.4f} "
                                  f"P@{k}={final_eval_metrics[f'precision_at_{k}']:.4f} "
                                  f"nDCG@{k}={final_eval_metrics[f'ndcg_at_{k}']:.4f}")
                    if "mrr" in final_eval_metrics:
                        print(f"  MRR={final_eval_metrics['mrr']:.4f}")
                    print()
                
                # Store results for potential cross-run comparison
                all_method_results[method_key] = final_eval_metrics
                
                # Finish this wandb run
                wandb.finish()

    # Print final comparison summary
    print(f"\n🎉 All experiments completed!")
    print("="*60)
    print("FINAL COMPARISON SUMMARY:")
    print("="*60)
    
    for method_key, final_metrics in all_method_results.items():
        print(f"{method_key}:")
        print(f"  Samples: {final_metrics.get('num_samples', 0)}")
        print(f"  R@10={final_metrics.get('recall_at_10', 0):.4f} "
              f"P@10={final_metrics.get('precision_at_10', 0):.4f} "
              f"nDCG@10={final_metrics.get('ndcg_at_10', 0):.4f} "
              f"MRR={final_metrics.get('mrr', 0):.4f}")
        print()
    
    print(f"📊 View individual runs in wandb project: reddit-music-qwen-audio")


if __name__ == "__main__":
    main()