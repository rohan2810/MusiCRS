import os
import json
import math
import random
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import soundfile as sf
import torch
import tqdm
import wandb

from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    GenerationConfig,
)

from utils import (
    calculate_comprehensive_metrics, calculate_subreddit_metrics,
    parse_and_filter_ranking,
    print_metrics_summary, print_subreddit_comparison
)


USER_TOK      = "<|user|>"
ASSISTANT_TOK = "<|assistant|>"
END_TOK       = "<|end|>"

def load_random_segments_for_phi(
    audio_paths,
    max_clips: int,
    total_budget_secs: float,
) -> list[tuple[np.ndarray, int]]:

    audios: list[tuple[np.ndarray, int]] = []
    if not audio_paths:
        return audios

    num_to_use = min(len(audio_paths), max_clips)
    clip_dur = total_budget_secs / num_to_use if num_to_use > 0 else 30.0
    selected = random.sample(audio_paths, k=num_to_use)

    for p in selected:
        try:
            info = sf.info(str(p))
            sr = info.samplerate
            total_secs = info.frames / sr
            start_sec = random.uniform(0, max(0, total_secs - clip_dur))
            start_frame = int(start_sec * sr)
            frames = int(clip_dur * sr)
            wav, _ = sf.read(str(p), start=start_frame, frames=frames, dtype="float32", always_2d=False)
            if wav.ndim > 1:
                wav = np.mean(wav, axis=1).astype(np.float32)
            audios.append((wav, sr))
        except Exception as e:
            print(f"[WARN] Failed to load segment from {p}: {e}")
            continue
    return audios


def build_phi_prompt(
    randomized_candidates: list[str],
    mode: str,
    query_text: str | None,
    num_audios: int
) -> str:

    audio_tag_blob = ""
    if mode in ("audio_query", "audio_only") and num_audios > 0:
        audio_tag_blob = "".join([f"<|audio_{i+1}|>" for i in range(num_audios)])

    query_blob = ""
    if mode in ("audio_query", "query_only") and query_text:
        query_blob = f"\nUser query/context:\n{query_text}\n"

    header = f"Here are {len(randomized_candidates)} candidate songs; rank by relevance:"
    cand_lines = [header] + [f"- {c}" for c in randomized_candidates]
    cand_blob = "\n".join(cand_lines)
    instr = (
        f"Rank these {len(randomized_candidates)} songs from most to least relevant based on the audio and query. "
        "Output ONLY the exact song titles separated by commas. "
        "Example format: Song Title A, Song Title B, Song Title C"
    )

    prompt = (
        f"{USER_TOK}"
        f"{audio_tag_blob}"
        f"{query_blob}\n{cand_blob}\n\n{instr}"
        f"{END_TOK}"
        f"{ASSISTANT_TOK}"
    )
    return prompt


# ----------------- RANKING (Phi) -----------------
def rank_songs_generative_phi(
    processor,
    model,
    generation_config,
    audio_paths: list[str],
    convo: str,
    candidates: list[str],
    mode: str,
    max_clips: int,
    audio_budget_secs: float,
) -> list[str]:
    randomized_candidates = candidates.copy()
    random.shuffle(randomized_candidates)

    # Prepare audio (if needed)
    audios_for_phi: list[tuple[np.ndarray, int]] = []
    if mode in ("audio_query", "audio_only"):
        audios_for_phi = load_random_segments_for_phi(
            audio_paths=audio_paths,
            max_clips=max_clips,
            total_budget_secs=audio_budget_secs,
        )
        if len(audios_for_phi) == 0:
            print("[INFO] No audio segments were successfully loaded.")

    prompt = build_phi_prompt(
        randomized_candidates=randomized_candidates,
        mode=mode,
        query_text=convo,
        num_audios=len(audios_for_phi),
    )

    try:
        if len(audios_for_phi) > 0:
            inputs = processor(
                text=prompt,
                audios=audios_for_phi, 
                return_tensors="pt",
            ).to(model.device)
        else:
            inputs = processor(
                text=prompt,
                return_tensors="pt",
            ).to(model.device)
    except Exception as e:
        print(f"[ERROR] Processor packing failed: {e}")
        return []

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            generation_config=generation_config,
        )
    gen_ids = out_ids[:, inputs["input_ids"].shape[1]:]
    text = processor.batch_decode(
        gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    return parse_and_filter_ranking(text, randomized_candidates, fill_missing=True)


# ----------------- MAIN -----------------
def main():
    # Seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device_map = "auto"
    MODEL_ID = "microsoft/Phi-4-multimodal-instruct" 
    audio_base = Path("")
    print(f"[INFO] Loading model: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map=device_map,
        torch_dtype="auto",
        trust_remote_code=True,
        _attn_implementation="flash_attention_2",
    )
    generation_config = GenerationConfig.from_pretrained(MODEL_ID)
    model.eval()
    modes = ["audio_query", "audio_only"]
    num_samples = -1
    input_jsonl = ""
    output_root = Path("")
    output_root.mkdir(parents=True, exist_ok=True)

    all_method_results = {}

    for mode in modes:
        wandb.init(
            project="reddit-music-phi4mm-final",
            entity="musiCRS",
            name=f"{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "model_id": MODEL_ID,
                "mode": mode,
                "random_seed": 42,
            },
            tags=["phi4-multimodal", "music-ranking", mode]
        )

        results = {"all_predictions": [], "all_ground_truth": [], "all_subreddits": []}
        print(f"[INFO] Processing mode: {mode}")

        out_path = output_root / f"{mode}.jsonl"
        with open(input_jsonl, "r", encoding="utf-8") as rf, \
             open(out_path, "w", encoding="utf-8") as wf:

            lines = rf.readlines()
            if num_samples != -1:
                lines = lines[:num_samples]

            for i, line in enumerate(tqdm.tqdm(lines, desc=f"Processing ({mode})")):
                try:
                    ex = json.loads(line)
                    sid = ex["name"]
                    convo = ex["query"]
                    candidates = ex["final_candidates"]
                    ground_truth = ex["limited_gt"]
                    wav_paths = sorted((audio_base / sid).glob("*.wav"))

                    if not candidates:
                        continue
                    if mode in ("audio_query","audio_only") and not wav_paths:
                        continue

                    ranked = rank_songs_generative_phi(
                        processor=processor,
                        model=model,
                        generation_config=generation_config,
                        audio_paths=[str(p) for p in wav_paths],
                        convo=convo,
                        candidates=candidates,
                        mode=mode,
                        max_clips=10,
                        audio_budget_secs=300.0,
                    )

                    record = ex.copy()
                    record["predicted"] = ranked
                    wf.write(json.dumps(record, ensure_ascii=False) + "\n")

                    results["all_predictions"].append(ranked)
                    results["all_ground_truth"].append(ground_truth)
                    results["all_subreddits"].append(ex.get("source_subreddit", "unknown"))

                    if i == 0:
                        print("\n===== DEBUG SAMPLE =====")
                        print("SID:", sid)
                        print("Mode:", mode)
                        print("Query:", (convo or "")[:300])
                        print("Predicted (top 10):", ranked[:10])
                        print("Ground truth:", ground_truth)
                        print("========================\n")

                except Exception as e:
                    print(f"[WARN] Error on sample {i}: {e}")
                    continue

        final_eval_metrics = calculate_comprehensive_metrics(
            results["all_predictions"], results["all_ground_truth"]
        )
        subreddit_metrics = calculate_subreddit_metrics(
            results["all_predictions"], results["all_ground_truth"], results["all_subreddits"]
        )


        final_log_dict = {}
        for metric_name, metric_value in final_eval_metrics.items():
            final_log_dict[f"FINAL_{metric_name}"] = metric_value
        
        for subreddit, sub_metrics in subreddit_metrics.items():
            for metric_name, metric_value in sub_metrics.items():
                final_log_dict[f"SUBREDDIT_{subreddit}_{metric_name}"] = metric_value
        
        wandb.log(final_log_dict)

        print(f"[INFO] Wrote results → {out_path}")
        
        if final_eval_metrics:
            print_metrics_summary(final_eval_metrics, f"FINAL Evaluation Results for {mode}")
            
            if subreddit_metrics:
                print_subreddit_comparison(subreddit_metrics)
        
        all_method_results[mode] = final_eval_metrics
        
        wandb.finish()

    print("\n🎉 All experiments completed!")
    print("="*60)
    print("FINAL COMPARISON SUMMARY:")
    print("="*60)
    for mode, fm in all_method_results.items():
        print(f"{mode}:")
        print(f"  Samples: {fm.get('num_samples', 0)}")
        print(f"  R@10={fm.get('recall_at_10', 0):.4f} "
              f"P@10={fm.get('precision_at_10', 0):.4f} "
              f"nDCG@10={fm.get('ndcg_at_10', 0):.4f} "
              f"MRR={fm.get('mrr', 0):.4f}")
        print()
    print("Done.")


if __name__ == "__main__":
    main()
