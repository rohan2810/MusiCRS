# Copyright (c) 2025 NVIDIA CORPORATION.
# Licensed under the MIT license.

import argparse
import json
import os
import random
from pathlib import Path
from typing import List

import librosa
import llava
from llava import conversation as clib
from llava.media import Sound
from peft import PeftModel
import torch
import tqdm
import numpy as np
import wandb
from datetime import datetime
from huggingface_hub import snapshot_download

from utils import (
    load_and_mix_audio, save_temp_audio, build_ranking_prompt, 
    parse_and_filter_ranking, normalize_string,
    calculate_comprehensive_metrics, calculate_subreddit_metrics,
    print_metrics_summary, print_subreddit_comparison
)

def rank_songs_generative_af3(
    model,
    wav_paths: List[Path],
    query: str,
    candidates: List[str],
    mode: str,
    max_clips: int = 10,
    audio_budget_secs: float = 30.0,
) -> List[str]:
    sound = None
    if mode in ("audio_query", "audio_only"):
        try:
            mixed = load_and_mix_audio(wav_paths, sr=16000, max_clips=max_clips, audio_budget_secs=audio_budget_secs)
            temp_path = save_temp_audio(mixed, sr=16000)
            sound = Sound(temp_path)
        except Exception as e:
            print(f"⚠️ Audio mixing/loading error: {e}")
            if mode == "audio_only":
                return candidates
    elif mode == "query_only":
        try:
            import numpy as np
            silent_audio = np.zeros(16000, dtype=np.float32)  
            temp_path = save_temp_audio(silent_audio, sr=16000)
            sound = Sound(temp_path)
        except Exception as e:
            print(f"⚠️ Error creating silent audio: {e}")
            return candidates



    prompt_text, randomized_candidates = build_ranking_prompt(
            query if mode in ("audio_query", "query_only") else "",
            candidates,
            mode,
            shuffle_candidates=True
        )

    prompt = []
    if sound:
        prompt.append(sound)
    prompt.append(prompt_text)
    
    print(f"🤖 PROMPT:\n{prompt}\n" + "=" * 50)
    try:
        response = model.generate_content(prompt)
        print(f"🤖 RAW RESPONSE:\n{response}\n" + "=" * 50)
        return parse_and_filter_ranking(response, randomized_candidates, fill_missing=True)
    except Exception as e:
        print(f"❌ Generation error: {e}")
        return candidates


def main():
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    
    model_base = "nvidia/audio-flamingo-3"
    
    model_path = snapshot_download(model_base)
    model_think = os.path.join(model_path, 'stage35')
    
    model = llava.load(model_path)
    model = model.to("cuda")
    
    print(f"✅ Model loaded on: {next(model.parameters()).device}")
    
    clib.default_conversation = clib.conv_templates["auto"].copy()
    
    input_jsonl = ""
    audio_base = Path("")
    modes = ["audio_query","audio_only","query_only"]  # Options: ["query_only", "audio_query", "audio_only"]
    num_samples = -1
    max_clips = 10
    audio_budget_secs = 300.0

    all_method_results = {}

    for mode in modes:
        wandb.init(
            project="reddit-music-af3-final",
            entity="musiCRS",
            name=f"{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "model": "AudioFlamingo3",
                "model_base": model_base,
                "mode": mode,
                "num_samples": num_samples,
                "max_clips": max_clips,
                "audio_budget_secs": audio_budget_secs,
                "random_seed": 42,
            },
            tags=["af3-final", "music-ranking", mode],
        )

        print(f"Processing {mode} mode")
        output_jsonl = f""
        output_dir = os.path.dirname(output_jsonl)
        os.makedirs(output_dir, exist_ok=True)

        results = {
            "all_predictions": [],
            "all_ground_truth": [],
            "all_subreddits": [],
        }

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

                    ranked = rank_songs_generative_af3(
                        model=model,
                        wav_paths=wav_paths,
                        query=convo,
                        candidates=candidates,
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
    
    print(f"📊 View individual runs in wandb project: reddit-music-af3-final")


if __name__ == "__main__":
    main()
