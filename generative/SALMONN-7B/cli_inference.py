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
    print_metrics_summary, print_subreddit_comparison,
    save_temp_audio, prepare_one_sample
)

parser = argparse.ArgumentParser()
parser.add_argument("--cfg-path", type=str, required=False, default=os.getenv("SALMONN7B_CFG", "generative/SALMONN-7B/configs/decode_config.yaml"), help="Path to configuration file")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument(
    "--options",
    nargs="+",
    help="override some settings in the used config, the key-value pair "
    "in xxx=yyy format will be merged into config file (deprecate), "
    "change to --cfg-options instead.",
)

def _prepare_samples_from_np(wav: np.ndarray, wav_processor: WhisperFeatureExtractor, sr: int = 16000, cuda_enabled: bool = True):
    tmp_path = save_temp_audio(wav.astype(np.float32), sr=sr)
    return prepare_one_sample(tmp_path, wav_processor, cuda_enabled=cuda_enabled)

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
        mixed = load_and_mix_audio(wav_paths, sr=sr, max_clips=max_clips, audio_budget_secs=audio_budget_secs)
    else:
        mixed = np.zeros(sr, dtype=float)

    prompt_text, randomized_candidates = build_ranking_prompt(
        convo if mode in ("audio_query", "query_only") else "",
        candidates,
        mode,
        shuffle_candidates=True
    )
    prompt_template = cfg.config.model.prompt_template or "{}"
    user_prompt = [prompt_template.format("<Speech><SpeechHere></Speech> " + prompt_text.strip())]
    
    samples = _prepare_samples_from_np(mixed, wav_processor, cuda_enabled=(model.device.type == "cuda"))
    with torch.cuda.amp.autocast(dtype=torch.float16):
        out_text = model.generate(samples, cfg.config.generate, prompts=user_prompt)[0]    
    cleaned_text = out_text.replace('<s>', '').replace('</s>', '').replace('<unk>', '').strip()
    return parse_and_filter_ranking(cleaned_text, randomized_candidates, fill_missing=True)

def main():
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    
    args = parser.parse_args()
    
    cfg = Config(args)
    model = SALMONN.from_config(cfg.config.model)
    model.to(args.device)
    model.eval()
    wav_processor = WhisperFeatureExtractor.from_pretrained(cfg.config.model.whisper_path)
    input_jsonl = os.getenv("INPUT_JSONL", "")
    audio_base = Path(os.getenv("AUDIO_BASE_DIR", "audio/full/wav"))
    
    modes = ["audio_query", "audio_only", "query_only"]  # Options: ["query_only", "audio_query", "audio_only"]
    num_samples = -1
    
    max_clips = 10
    audio_budget_secs = 30.0  # SALMONN's 30-second limit
    
    all_method_results = {}
    
    for mode in modes:
        wandb.login()
        wandb.init(
            project="reddit-music-salmonn-7b-final",
            entity="musiCRS",
            name=f"{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "model": "SALMONN-7B",
                "llama_path": str(cfg.config.model.llama_path),
                "whisper_path": str(cfg.config.model.whisper_path),
                "ckpt": str(cfg.config.model.ckpt),
                "mode": mode,
                "num_samples": num_samples,
                "max_clips": max_clips,
                "audio_budget_secs": audio_budget_secs,
                "random_seed": 42,
            },
            tags=["salmonn-7b-final", "music-ranking", mode]
        )
        
        results = {
            "all_predictions": [],
            "all_ground_truth": [],
            "all_subreddits": []
        }
        
        print(f"Processing {mode} mode")
        output_jsonl = os.getenv("SALMONN7B_OUTPUT", f"generative/results/SALMONN_7B_FINAL/{mode}.jsonl")
        output_dir = os.path.dirname(output_jsonl)
        os.makedirs(output_dir, exist_ok=True)
        
        with open(input_jsonl, "r", encoding="utf-8") as rf, \
             open(output_jsonl, "w", encoding="utf-8") as wf:
                    
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
                    
                    results["all_predictions"].append(ranked)
                    results["all_ground_truth"].append(ground_truth)
                    results["all_subreddits"].append(ex.get("source_subreddit", "unknown"))
                
                except Exception as e:
                    print(f"Error processing sample {i}: {e}")
                
        final_eval_metrics = calculate_comprehensive_metrics(
            results["all_predictions"],
            results["all_ground_truth"]
        )
        
        subreddit_metrics = calculate_subreddit_metrics(
            results["all_predictions"],
            results["all_ground_truth"], 
            results["all_subreddits"]
        )
        
        final_log_dict = {}
        for metric_name, metric_value in final_eval_metrics.items():
            final_log_dict[f"FINAL_{metric_name}"] = metric_value
        
        for subreddit, sub_metrics in subreddit_metrics.items():
            for metric_name, metric_value in sub_metrics.items():
                final_log_dict[f"SUBREDDIT_{subreddit}_{metric_name}"] = metric_value
        
        wandb.log(final_log_dict)
        
        print(f"Wrote {mode} results to {output_jsonl}")
        
        if final_eval_metrics:
            print_metrics_summary(final_eval_metrics, f"FINAL Evaluation Results for {mode}")
            
            if subreddit_metrics:
                print_subreddit_comparison(subreddit_metrics)
        
        all_method_results[mode] = final_eval_metrics
        
        wandb.finish()
    
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
    
    print(f"📊 View individual runs in wandb project: reddit-music-salmonn-7b-final")

if __name__ == "__main__":
    main()
