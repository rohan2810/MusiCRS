# Copyright (2024) Tsinghua University, Bytedance Ltd. and/or its affiliates
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

import numpy as np
import torch
from transformers import WhisperFeatureExtractor
import wandb
import tqdm
from datetime import datetime

from config import Config
from models.salmonn import SALMONN

from utils import (
    calculate_comprehensive_metrics, calculate_subreddit_metrics,
    load_and_mix_audio, build_ranking_prompt, parse_and_filter_ranking,
    print_metrics_summary, print_subreddit_comparison
)

parser = argparse.ArgumentParser()
parser.add_argument("--cfg-path", type=str, required=False, default="/home/junda/rohan/reddit-music/generative/SALMONN/configs/decode_config.yaml", help="Path to configuration file")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument(
    "--options",
    nargs="+",
    help="override some settings in the used config, the key-value pair "
    "in xxx=yyy format will be merged into config file (deprecate), "
    "change to --cfg-options instead.",
)

def _prepare_samples_from_np(wav: np.ndarray, wav_processor: WhisperFeatureExtractor, sr: int = 16000, cuda_enabled: bool = True):
    """Prepare SALMONN samples from a numpy waveform, exactly mirroring utils.prepare_one_sample."""
    # Ensure wav is the right type and shape, exactly like prepare_one_sample
    if wav.ndim == 2:
        wav = wav[:, 0]
    
    # Pad to at least 1 second, exactly like prepare_one_sample
    if len(wav) < sr:
        sil = np.zeros(sr - len(wav), dtype=float)
        wav = np.concatenate((wav, sil), axis=0)  # Use tuple like prepare_one_sample
    
    # Truncate to 30 seconds max, exactly like prepare_one_sample
    wav = wav[: sr * 30]
    
    # Process exactly like prepare_one_sample
    spectrogram = wav_processor(wav, sampling_rate=sr, return_tensors="pt")["input_features"]
    
    samples = {
        "spectrogram": spectrogram,
        "raw_wav": torch.from_numpy(wav).unsqueeze(0),
        "padding_mask": torch.zeros(len(wav), dtype=torch.bool).unsqueeze(0),
    }
    
    if cuda_enabled:
        from utils import move_to_cuda
        samples = move_to_cuda(samples)
    
    return samples

def rank_songs_generative_salmonn(
    model: SALMONN,
    wav_paths: list[Path],
    convo: str,
    candidates: list[str],
    wav_processor: WhisperFeatureExtractor,
    cfg,
    mode: str,
    max_clips: int = 10,
    audio_budget_secs: float = 30.0,
) -> list[str]:
    sr = 16000
    if mode in ("audio_query", "audio_only"):
        mixed = load_and_mix_audio([Path(p) for p in wav_paths], sr=sr, max_clips=max_clips, audio_budget_secs=audio_budget_secs)
    else:
        mixed = np.zeros(sr, dtype=float)

    prompt_text, randomized_candidates = build_ranking_prompt(
        convo if mode in ("audio_query", "query_only") else "",
        candidates,
        mode,
        shuffle_candidates=True
    )
    # SALMONN expects the <Speech> tag in the prompt to mark audio location
    user_prompt = [cfg.config.model.prompt_template.format("<Speech><SpeechHere></Speech> " + prompt_text.strip())]

    # Always use the fixed _prepare_samples_from_np with your original mixing logic
    samples = _prepare_samples_from_np(mixed, wav_processor, cuda_enabled=(model.device.type == "cuda"))
    with torch.amp.autocast('cuda', dtype=torch.float16):
        out_text = model.generate(samples, cfg.config.generate, prompts=user_prompt)[0]

    # Clean up the output text - remove special tokens and extra whitespace
    cleaned_text = out_text.replace('<s>', '').replace('</s>', '').replace('<unk>', '').strip()
    
    # Use utility function to parse and filter the ranking
    return parse_and_filter_ranking(cleaned_text, randomized_candidates, fill_missing=True)

def main():
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    
    args = parser.parse_args()
    
    # Load configuration and model
    cfg = Config(args)
    model = SALMONN.from_config(cfg.config.model)
    model.to(args.device)
    model.eval()
    wav_processor = WhisperFeatureExtractor.from_pretrained(cfg.config.model.whisper_path)
    input_jsonl = "/home/junda/rohan/reddit-music/data/merged_final_cleaned_music_clean_queries_with_candidates_enhanced_v2_100_candidates.jsonl"
    audio_base = Path("/deepfreeze/junda/abhay/audio_yt/full/wav")
    
    modes = ["audio_query", "audio_only", "query_only"]  # Options: ["query_only", "audio_query", "audio_only"]
    num_samples = -1
    
    # Audio processing parameters
    max_clips = 10
    audio_budget_secs = 30.0  # SALMONN's 30-second limit
    
    all_method_results = {}
    
    for mode in modes:
        wandb.login(key='d13ff15cc78423826ede4beef6d5e6f91a7cc50e')
        wandb.init(
            project="reddit-music-salmonn-final",
            entity="musiCRS",
            name=f"{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "model": "SALMONN",
                "llama_path": str(cfg.config.model.llama_path),
                "whisper_path": str(cfg.config.model.whisper_path),
                "ckpt": str(cfg.config.model.ckpt),
                "mode": mode,
                "num_samples": num_samples,
                "max_clips": max_clips,
                "audio_budget_secs": audio_budget_secs,
                "random_seed": 42,
            },
            tags=["salmonn-final", "music-ranking", mode]
        )
        
        results = {
            "all_predictions": [],
            "all_ground_truth": [],
            "all_subreddits": []
        }
        
        print(f"Processing {mode} mode")
        output_jsonl = f"/home/junda/rohan/reddit-music/generative/results/SALMONN_FINAL/{mode}.jsonl"
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
                    
                    ranked = rank_songs_generative_salmonn(
                        model=model,
                        wav_paths=wav_paths,
                        convo=convo,
                        candidates=candidates,
                        wav_processor=wav_processor,
                        cfg=cfg,
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
    
    print(f"📊 View individual runs in wandb project: reddit-music-salmonn-final")

if __name__ == "__main__":
    main()
