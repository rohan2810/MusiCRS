# Copyright (c) 2025 NVIDIA CORPORATION. 
# Licensed under the MIT license.

import os
import json
import argparse
import random
import math
import time
import tempfile
from datetime import datetime
from collections import defaultdict
from pathlib import Path

import torch
import numpy as np
import tqdm
import librosa
import soundfile as sf
from huggingface_hub import snapshot_download
from typing import List, Dict, Optional, Tuple
import wandb
WANDB_AVAILABLE = True

import llava

from utils import (
    recall_at_k, precision_at_k, mrr, ndcg_at_k, hit_rate_at_k, average_first_relevant_position,
    normalize_string, normalize_list_of_lists, calculate_comprehensive_metrics,
    load_and_mix_audio, save_temp_audio, build_ranking_prompt, parse_and_filter_ranking,
    print_metrics_summary
)


def calculate_incremental_metrics(all_preds, all_golds, k_values=[1, 5, 10, 20]):
    """Calculate comprehensive evaluation metrics - wrapper for utils function"""
    return calculate_comprehensive_metrics(all_preds, all_golds, k_values)


def rank_songs_af3_generative(
    audio_paths: list[str],
    convo: str,
    candidates: list[str],
    model,
    mode: str = "audio_query",
    max_clips: int = 10,
    audio_budget_secs: float = 300.0,  # 5 minutes total budget 
    include_descriptions: bool = True,
    summaries: dict = None,
) -> list[str]:
    """Audio Flamingo 3 generative approach with smart audio budgeting (single-Sound concatenation)"""
    
    sound = None
    if mode in ("audio_query", "audio_only"):
        print(f"🎵 Processing audio for mode: {mode}")
        print(f"🎵 Found {len(audio_paths)} audio files")
        print(f"🎵 Audio budget: {audio_budget_secs}s, max_clips: {max_clips}")
        
        mixed_audio = load_and_mix_audio(
            [Path(p) for p in audio_paths], 
            sr=16000, 
            max_clips=max_clips, 
            audio_budget_secs=audio_budget_secs
        )
        
        if mixed_audio is not None and mixed_audio.size > 0:
            print(f"🎵 Mixed audio shape: {mixed_audio.shape}, duration: {mixed_audio.size/16000:.2f}s")
            
            temp_path = None
            try:
                temp_path = save_temp_audio(mixed_audio, sr=16000)
                print(f"🎵 Saved temp audio to: {temp_path}")
                sound = llava.Sound(temp_path)
                print(f"🎵 Created llava.Sound object successfully")
                
            except Exception as e:
                print(f"❌ Failed to create Sound object: {e}")
                sound = None
                if temp_path and os.path.exists(temp_path):
                    os.unlink(temp_path)
        else:
            print(f"❌ No valid audio data loaded")

    # Check if we have valid audio for audio modes
    if mode in ("audio_query", "audio_only") and sound is None:
        print(f"❌ Warning: Audio mode requested but no valid audio clips loaded")
        return candidates  # Return original order as fallback

    # Build prompt using utility function
    prompt_text, randomized_candidates = build_ranking_prompt(
        convo if mode in ("audio_query", "query_only") else "",
        candidates,
        mode,
        shuffle_candidates=True
    )
    
    # For AF3, we need to add the "Ranked list:" suffix (no few-shot)
    text_prompt = f"{prompt_text}\nRanked list:"

    # Use model's default generation config (following original repo pattern)
    generation_config = model.default_generation_config
    generation_config.max_new_tokens = 512
    print(f"🤖 Generation config: {generation_config}")    
    try:
        # Follow the exact pattern from original repo
        if mode in ("audio_query", "audio_only") and sound is not None:
            full_prompt = text_prompt  # Sound object already passed; no <sound> token needed
            print(f"🤖 Using audio+text mode with Sound object")
            print("🤖 --- FULL PROMPT START ---")
            print(full_prompt)
            print("🤖 --- FULL PROMPT END ---")
            print(f"🤖 Prompt length: {len(full_prompt)} chars")
            response = model.generate_content([sound, full_prompt], generation_config=generation_config)
        else:
            # Text-only mode
            print(f"🤖 Using text-only mode")
            print("🤖 --- FULL PROMPT START ---")
            print(text_prompt)
            print("🤖 --- FULL PROMPT END ---")
            print(f"🤖 Prompt length: {len(text_prompt)} chars")
            response = model.generate_content([text_prompt], generation_config=generation_config)
        
        print("🤖 --- FULL MODEL RESPONSE START ---")
        print(response)
        print("🤖 --- FULL MODEL RESPONSE END ---")
        print(f'🤖 Response length: {len(response)} chars')
        # Clean up temporary audio file now that generation is finished
        if "temp_path" in locals() and temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
            print(f"🎵 Cleaned up temp file after generation: {temp_path}")
    except Exception as e:
        print(f"❌ Error generating response: {e}")
        import traceback
        traceback.print_exc()
        return candidates  # Return original order as fallback
    
    # Parse and filter ranking using utility function
    ranked = parse_and_filter_ranking(response, randomized_candidates, fill_missing=True)
    return ranked[:len(candidates)]


def main():
    
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Constants and paths
    AUDIO_BASE_DIR = Path("")
    SUM_FILE = Path("")
    
    # Load summaries
    with SUM_FILE.open("r", encoding="utf-8") as f:
        summaries = json.load(f)
    
    input_jsonl = ""
    
    # Configuration options
    modes = ["audio_query"]
    descriptions_options = [False]
    num_samples = 1
    

    print("Loading Audio Flamingo 3 model...")
    model_path = snapshot_download(repo_id="nvidia/audio-flamingo-3")
    model = llava.load(model_path)
    model = model.to("cuda")
    model.eval()
    print("Model loaded successfully!")

    with open(input_jsonl, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    all_method_results = {}

    for mode in modes:
        for descriptions in descriptions_options:
            method_key = f"flamingo3_generative_{mode}_desc_{descriptions}"
            
            # Initialize separate wandb run for each method combination
            wandb.init(
                project="reddit-music-flamingo3-audio",
                name=f"{method_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    "model": "nvidia/audio-flamingo-3",
                    "ranking_method": "generative",
                    "mode": mode,
                    "include_descriptions": descriptions,
                    "num_samples": num_samples,
                    "max_clips": 10,
                    "random_seed": 42,
                },
                tags=["audio-flamingo-3", "music-ranking", "generative", mode, f"descriptions_{descriptions}"]
            )
            
            method_results = {
                "all_predictions": [],
                "all_ground_truth": []
            }
            
            print(f"Processing {mode} mode with generative method (descriptions: {descriptions})")
            output_jsonl = f""
            os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

            with open(input_jsonl, "r", encoding="utf-8") as rf, \
                 open(output_jsonl, "w", encoding="utf-8") as wf:

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

                        # Rank using AF3 generative approach
                        ranked = rank_songs_af3_generative(
                            audio_paths=wav_paths,
                            convo=convo,
                            candidates=candidates,
                            model=model,
                            mode=mode,
                            max_clips=10,
                            audio_budget_secs=300.0,  # 5 minutes total budget
                            include_descriptions=descriptions,
                            summaries=summaries
                        )

                        record = ex.copy()
                        record["predicted"] = ranked
                        wf.write(json.dumps(record, ensure_ascii=False) + "\n")
                        
                        method_results["all_predictions"].append(ranked)
                        method_results["all_ground_truth"].append(ground_truth)
                        
                        if (i + 1) % 10 == 0:
                            eval_metrics = calculate_incremental_metrics(
                                method_results["all_predictions"],
                                method_results["all_ground_truth"]
                            )
                            
                            wandb.log(eval_metrics)
                    
                    except Exception as e:
                        print(f"Error processing sample {i}: {e}")
                        continue

            final_eval_metrics = calculate_incremental_metrics(
                method_results["all_predictions"],
                method_results["all_ground_truth"]
            )
            
            final_log_dict = {}
            for metric_name, metric_value in final_eval_metrics.items():
                final_log_dict[f"FINAL_{metric_name}"] = metric_value
            
            # Log final evaluation metrics (with FINAL_ prefix for clarity)
            wandb.log(final_log_dict)

            print(f"Wrote {mode} results to {output_jsonl}")
            
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
            
            wandb.finish()

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
    
    print(f"📊 View individual runs in wandb project: reddit-music-flamingo3-audio")


if __name__ == "__main__":
    main()