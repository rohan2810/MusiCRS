# Copyright (2023) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
import random
from pathlib import Path
from typing import Union, Dict, List, Optional

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import wandb
import time
import math
import tqdm
from datetime import datetime

from model import SALMONN

from utils import (
    calculate_comprehensive_metrics, calculate_subreddit_metrics,
    load_and_mix_audio, build_ranking_prompt, parse_and_filter_ranking,
    print_metrics_summary, print_subreddit_comparison
)


parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--ckpt_path", type=str, default='./salomnn_7b.bin')
parser.add_argument("--whisper_path", type=str, default='/home/junda/rohan/reddit-music/generative/SALMONN/whisper-large-v2')
parser.add_argument("--beats_path", type=str, default='/home/junda/rohan/reddit-music/generative/SALMONN/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt')
parser.add_argument("--vicuna_path", type=str, default='/home/junda/rohan/reddit-music/generative/SALMONN-7B/vicuna-7b-v1.5')
parser.add_argument("--low_resource", action='store_true', default=False)
parser.add_argument("--debug", action="store_true", default=False)

# Minimal batch arguments
parser.add_argument("--output-jsonl", type=str, default="", help="Where to write predictions for batch mode")


def _save_audio_for_futga(mixed_audio: np.ndarray, sr: int = 16000) -> str:
    """Save mixed audio to temporary file for FUTGA model."""
    temp_path = f"/tmp/temp_audio_{int(time.time()*1000)}.wav"
    sf.write(temp_path, mixed_audio, sr)
    return temp_path

def rank_songs_generative_futga(
    model: SALMONN,
    wav_paths: list[Path],
    convo: str,
    candidates: list[str],
    mode: str = "audio_query",
    max_clips: int = 10,
    audio_budget_secs: float = 300.0,
) -> list[str]:
    sr = 16000
    if mode in ("audio_query", "audio_only"):
        mixed = load_and_mix_audio([Path(p) for p in wav_paths], sr=sr, max_clips=max_clips, audio_budget_secs=audio_budget_secs)
        temp_audio_path = _save_audio_for_futga(mixed, sr)
    else:
        silence = np.zeros(sr, dtype=float)
        temp_audio_path = _save_audio_for_futga(silence, sr)

    prompt_text, randomized_candidates = build_ranking_prompt(
        convo if mode in ("audio_query", "query_only") else "",
        candidates,
        mode,
        shuffle_candidates=True
    )
    
    try:
        # Use FUTGA model interface with original generation parameters
        with torch.cuda.amp.autocast(dtype=torch.float16):
            out_text = model.generate(
                temp_audio_path, 
                prompt=prompt_text.strip(), 
                repetition_penalty=1.5, 
                num_beams=10, 
                top_p=.7, 
                temperature=.2,
            )[0]

        # Clean up temporary file
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            
        # Use utility function to parse and filter the ranking
        return parse_and_filter_ranking(out_text, randomized_candidates, fill_missing=True)
        
    except Exception as e:
        print(f"Error in generation: {e}")
        # Clean up temporary file
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        return []


def main():
    """Main function similar to SALMONN-7B approach."""
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    args = parser.parse_args()

    # Load model using FUTGA interface but keeping original args
    print('Loading FUTGA model...')
    model = SALMONN(
        ckpt=args.ckpt_path,
        whisper_path=args.whisper_path,
        beats_path=args.beats_path,
        vicuna_path=args.vicuna_path,
    ).to(torch.float16).to(args.device)
    model.eval()
    print('✅ FUTGA model loaded successfully!')
    
    input_jsonl = "/home/junda/rohan/reddit-music/data/merged_final_cleaned_music_clean_queries_with_candidates_enhanced_v2_100_candidates.jsonl"
    audio_base = Path("/deepfreeze/junda/abhay/audio_yt/full/wav")
    
    modes = ["audio_query", "audio_only", "query_only"]  # Options: ["query_only", "audio_query", "audio_only"]
    num_samples = -1
    
    # Audio processing parameters
    max_clips = 10
    audio_budget_secs = 300.0  # 300-second limit
    
    all_method_results = {}
    
    for mode in modes:
        wandb.login(key='d13ff15cc78423826ede4beef6d5e6f91a7cc50e')
        wandb.init(
            project="reddit-music-futga-final",
            entity="musiCRS",
            name=f"{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "model": "FUTGA",
                "ckpt_path": str(args.ckpt_path),
                "whisper_path": str(args.whisper_path),
                "beats_path": str(args.beats_path),
                "vicuna_path": str(args.vicuna_path),
                "mode": mode,
                "num_samples": num_samples,
                "max_clips": max_clips,
                "audio_budget_secs": audio_budget_secs,
                "random_seed": 42,
                "generation_config": {
                    "repetition_penalty": 1.5,
                    "num_beams": 10,
                    "top_p": 0.7,
                    "temperature": 0.2
                }
            },
            tags=["futga-final", "music-ranking", mode]
        )
        
        results = {
            "all_predictions": [],
            "all_ground_truth": [],
            "all_subreddits": []
        }
        
        print(f"Processing {mode} mode")
        output_jsonl = f"/home/junda/rohan/reddit-music/generative/results/FUTGA_FINAL/{mode}.jsonl"
        output_dir = os.path.dirname(output_jsonl)
        os.makedirs(output_dir, exist_ok=True)
        
        with open(input_jsonl, "r", encoding="utf-8") as rf, \
             open(output_jsonl, "w", encoding="utf-8") as wf:
                
            # Load samples - use all if num_samples is -1
            if num_samples == -1:
                lines = rf.readlines()
            else:
                lines = rf.readlines()[:num_samples]
            
            for i, line in enumerate(tqdm.tqdm(lines, desc=f"Processing ({mode})")):
                try:
                    ex = json.loads(line)
                    sid = ex["name"]
                    convo = ex["query"]
                    candidates = ex["final_candidates"]
                    ground_truth = ex["limited_gt"]
                    
                    if not candidates:
                        continue
                        
                    wav_paths: list[Path] = []
                    if audio_base is not None:
                        audio_dir = audio_base / str(sid)
                        if audio_dir.exists():
                            wav_paths = sorted(list(audio_dir.glob("*.wav")))
                    
                    # Skip if audio is required but missing
                    if mode in ("audio_query", "audio_only") and not wav_paths:
                        continue
                    
                    ranked = rank_songs_generative_futga(
                        model=model,
                        wav_paths=wav_paths,
                        convo=convo,
                        candidates=candidates,
                        mode=mode,
                        max_clips=max_clips,
                        audio_budget_secs=audio_budget_secs,
                    )
                    
                    record = ex.copy()
                    record["predicted"] = ranked
                    wf.write(json.dumps(record, ensure_ascii=False) + "\n")
                    
                    # Collect for final evaluation
                    results["all_predictions"].append(ranked)
                    results["all_ground_truth"].append(ground_truth)
                    results["all_subreddits"].append(ex.get("source_subreddit", "unknown"))
                
                except Exception as e:
                    print(f"Error processing sample {i}: {e}")
                
        # Calculate FINAL evaluation metrics
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
        
        print(f"Wrote {mode} results to {output_jsonl}")
        
        # Print FINAL evaluation results
        if final_eval_metrics:
            print_metrics_summary(final_eval_metrics, f"FINAL Evaluation Results for {mode}")
            
            # Print per-subreddit results
            if subreddit_metrics:
                print_subreddit_comparison(subreddit_metrics)
        
        # Store results for potential cross-run comparison
        all_method_results[mode] = final_eval_metrics
        
        # Finish this wandb run
        wandb.finish()
    
    # Print final comparison summary
    print(f"\n🎉 All experiments completed!")
    print("="*60)
    print("FINAL COMPARISON SUMMARY:")
    print("="*60)
    
    for mode, final_metrics in all_method_results.items():
        print(f"{mode}:")
        print(f"  Samples: {final_metrics.get('num_samples', 0)}")
        print(f"  R@10={final_metrics.get('recall_at_10', 0):.4f} "
              f"P@10={final_metrics.get('precision_at_10', 0):.4f} "
              f"nDCG@10={final_metrics.get('ndcg_at_10', 0):.4f} "
              f"MRR={final_metrics.get('mrr', 0):.4f}")
        print()
    
    print(f"📊 View individual runs in wandb project: reddit-music-futga-final")




if __name__ == "__main__":
    main()
