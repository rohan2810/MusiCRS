import os
import json
import math
import random
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import soundfile as sf
import librosa
import torch
import torch.nn.functional as F
import tqdm
import wandb

from transformers import ClapModel, ClapProcessor

# === CONFIG ===================================================================
MODEL_ID              = "laion/larger_clap_music_and_speech"
DEVICE                = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AUDIO_BASE_DIR        = Path(os.getenv("AUDIO_BASE_DIR", "audio/full/wav"))
INPUT_JSONL           = os.getenv("INPUT_JSONL", "")
OUTPUT_DIR            = os.getenv("CLAP_OUTPUT_DIR", "retrieval/results/CLAP_FINAL")

# Eval loop settings
MODES                 = ["audio_only", "query_only", "audio_query"] 
NUM_SAMPLES           = -1   

# W&B
WANDB_PROJECT         = "reddit-music-clap-final"
WANDB_ENTITY          = "musiCRS"   
WANDB_TAGS            = ["clap-final", "music-ranking"]
AUDIO_SEGMENT_SECS    = 10.0       # ~CLAP max receptive field window
AUDIO_BUDGET_SECS     = 300.0      # 5 minutes total per post
MAX_FILES_PER_POST    = 10
AUDIO_WEIGHT          = 0.5        # weight for audio in fusion (0..1)
RANDOM_SEED           = 42

# === UTILS FROM YOUR REPO =====================================================
from utils import (
    calculate_comprehensive_metrics,
    calculate_subreddit_metrics,
    print_metrics_summary,
    print_subreddit_comparison,
)

# === SETUP ====================================================================
def set_all_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_all_seeds(RANDOM_SEED)
torch.backends.cudnn.benchmark = True

print("Loading CLAP model...")
processor = ClapProcessor.from_pretrained(MODEL_ID)
model = ClapModel.from_pretrained(MODEL_ID)
model.to(DEVICE)
model.eval()
print("Model loaded.")

CLAP_SR = getattr(processor, "feature_extractor", None).sampling_rate if hasattr(processor, "feature_extractor") else 48000

# === AUDIO SAMPLING ===========================================================

def _select_files(audio_paths: List[Path], k: int) -> List[Path]:
    if len(audio_paths) <= k:
        return list(audio_paths)
    rng = random.Random(RANDOM_SEED)
    return rng.sample(list(audio_paths), k=k)

def collect_audio_segments(
    audio_paths: List[Path],
    sr: int,
    segment_secs: float,
    budget_secs: float,
    max_files: int,
) -> List[np.ndarray]:
    if not audio_paths:
        return []

    sel_files = _select_files(audio_paths, max_files)
    max_segments = max(1, int(budget_secs // segment_secs))
    segments_per_file = max(1, math.ceil(max_segments / max(1, len(sel_files))))

    segments: List[np.ndarray] = []

    for p in sel_files:
        try:
            info = sf.info(str(p))
            total_secs = info.frames / float(info.samplerate) if info.samplerate > 0 else 0.0
        except Exception:
            try:
                y_tmp, sr_tmp = librosa.load(str(p), sr=None, mono=True)
                total_secs = len(y_tmp) / float(sr_tmp)
            except Exception:
                continue


        if total_secs <= segment_secs:
            starts = [0.0]
        else:
            span = max(total_secs - segment_secs, 0.0)
            starts = [i * (span / (segments_per_file - 1)) if segments_per_file > 1 else 0.0
                      for i in range(segments_per_file)]

        # Extract segments
        for start in starts:
            try:
                wav, _ = librosa.load(
                    str(p), sr=sr, mono=True, offset=float(start), duration=float(segment_secs)
                )
                if wav is None or len(wav) == 0:
                    continue
                segments.append(wav.astype(np.float32))
            except Exception:
                continue

            if len(segments) >= max_segments:
                break
        if len(segments) >= max_segments:
            break

    return segments

# === EMBEDDINGS ===============================================================

@torch.no_grad()
def embed_audio_segments(segments: List[np.ndarray]) -> Optional[torch.Tensor]:
    if not segments:
        return None

    seg_embs: List[torch.Tensor] = []
    for seg in segments:
        inputs = processor(
            audios=[seg],
            return_tensors="pt",
            sampling_rate=CLAP_SR
        )
        inputs = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
        feats = model.get_audio_features(**inputs)   # (1, D)
        feats = F.normalize(feats, dim=-1)          # normalize per segment
        seg_embs.append(feats.squeeze(0))           # (D,)

    all_segs = torch.stack(seg_embs, dim=0)         # (N, D)
    pooled = all_segs.mean(dim=0, keepdim=True)     # (1, D)
    pooled = F.normalize(pooled, dim=-1)            # (1, D)
    return pooled.squeeze(0)                        # (D,)


@torch.no_grad()
def embed_texts(texts: List[str]) -> Optional[torch.Tensor]:
    if not texts:
        return None

    out: List[torch.Tensor] = []
    for t in texts:
        inputs = processor(text=[t], return_tensors="pt", padding=True)
        inputs = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
        feats = model.get_text_features(**inputs)   # (1, D)
        feats = F.normalize(feats, dim=-1)
        out.append(feats.squeeze(0))                # (D,)

    return torch.stack(out, dim=0)                  # (N, D)

# === RANKERS ==================================================================
def rank_songs_clap(
    audio_paths: List[Path],
    convo: Optional[str],
    candidates: List[str],
    mode: str,
    audio_budget_secs: float = AUDIO_BUDGET_SECS,
    segment_secs: float = AUDIO_SEGMENT_SECS,
    audio_weight: float = AUDIO_WEIGHT,
) -> List[str]:
    if not candidates:
        return candidates

    # Candidate text embeddings (always needed)
    cand_embs = embed_texts(candidates)  # (N, D)
    if cand_embs is None or cand_embs.shape[0] == 0:
        return candidates

    # Prepare embeddings per mode
    audio_emb = None
    text_query_emb = None

    if mode in ("audio_only", "audio_query"):
        segments = collect_audio_segments(
            audio_paths=audio_paths,
            sr=CLAP_SR,
            segment_secs=segment_secs,
            budget_secs=audio_budget_secs,
            max_files=MAX_FILES_PER_POST,
        )
        audio_emb = embed_audio_segments(segments)

    if mode in ("query_only", "audio_query"):
        if convo and len(convo.strip()) > 0:
            text_query_emb = embed_texts([convo])
            if text_query_emb is not None:
                text_query_emb = text_query_emb[0]  # (D,)
        else:
            text_query_emb = None

    # Compute similarities
    if mode == "audio_only":
        if audio_emb is None:
            return candidates
        sims = (cand_embs @ audio_emb).detach().float().cpu().numpy()     # (N,)

    elif mode == "query_only":
        if text_query_emb is None:
            return candidates
        sims = (cand_embs @ text_query_emb).detach().float().cpu().numpy()

    elif mode == "audio_query":
        # If one modality is missing, fall back to the other
        if audio_emb is None and text_query_emb is None:
            return candidates

        parts = []
        weights = []
        if audio_emb is not None:
            parts.append(audio_emb)
            weights.append(audio_weight)
        if text_query_emb is not None:
            parts.append(text_query_emb)
            weights.append(1.0 - audio_weight if audio_emb is not None else 1.0)
        fused = torch.stack(parts, dim=0)
        w = torch.tensor(weights, dtype=fused.dtype, device=fused.device).view(-1, 1)
        fused = F.normalize((w * fused).sum(dim=0, keepdim=True), dim=-1).squeeze(0)  # (D,)
        sims = (cand_embs @ fused).detach().float().cpu().numpy()


    idxs = np.argsort(sims)[::-1]
    return [candidates[i] for i in idxs]

# === MAIN LOOP ================================================================
def run_mode(mode: str, num_samples: int = -1):
    set_all_seeds(RANDOM_SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_jsonl = os.path.join(OUTPUT_DIR, f"{mode}.jsonl")

    # W&B
    run_name = f"{mode}_{MODEL_ID.split('/')[-1]}_{random.randint(0, 99999):05d}"
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=run_name,
        config={
            "model_id": MODEL_ID,
            "mode": mode,
            "num_samples": num_samples,
            "random_seed": RANDOM_SEED,
            "audio_segment_secs": AUDIO_SEGMENT_SECS,
            "audio_budget_secs": AUDIO_BUDGET_SECS,
            "audio_weight": AUDIO_WEIGHT,
        },
        tags=WANDB_TAGS + [mode],
    )

    all_predictions, all_ground_truth, all_subreddits = [], [], []

    with open(INPUT_JSONL, "r", encoding="utf-8") as rf:
        lines = rf.readlines()
    if num_samples != -1:
        lines = lines[:num_samples]

    with open(output_jsonl, "w", encoding="utf-8") as wf:
        for i, line in enumerate(tqdm.tqdm(lines, desc=f"Processing ({mode})")):
            try:
                ex = json.loads(line)
                sid = ex["name"]
                convo = ex.get("query", "")
                candidates = ex.get("final_candidates") or ex.get("candidates") or []
                ground_truth = ex.get("limited_gt", [])

                # Gather audio files
                audio_dir = AUDIO_BASE_DIR / sid
                wav_paths = sorted(audio_dir.glob("*.wav"))

                # Minimal gating per mode
                if mode == "audio_only" and (not wav_paths or not candidates):
                    continue
                if mode == "query_only" and (not candidates or not convo):
                    continue
                if mode == "audio_query" and (not candidates or (not wav_paths and not convo)):
                    continue

                ranked = rank_songs_clap(
                    audio_paths=wav_paths,
                    convo=convo,
                    candidates=candidates,
                    mode=mode,
                    audio_budget_secs=AUDIO_BUDGET_SECS,
                    segment_secs=AUDIO_SEGMENT_SECS,
                    audio_weight=AUDIO_WEIGHT,
                )

                record = ex.copy()
                record["predicted"] = ranked
                wf.write(json.dumps(record, ensure_ascii=False) + "\n")

                # For evaluation aggregation
                all_predictions.append(ranked)
                all_ground_truth.append(ground_truth)
                all_subreddits.append(ex.get("source_subreddit", "unknown"))

            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue

    # === FINAL METRICS =========================================================
    final_eval_metrics = calculate_comprehensive_metrics(all_predictions, all_ground_truth)
    subreddit_metrics = calculate_subreddit_metrics(all_predictions, all_ground_truth, all_subreddits)

    # Log final metrics with FINAL_ prefix
    final_log = {}
    for k, v in final_eval_metrics.items():
        final_log[f"FINAL_{k}"] = v
    for subr, subm in subreddit_metrics.items():
        for k, v in subm.items():
            final_log[f"SUBREDDIT_{subr}_{k}"] = v
    wandb.log(final_log)

    # Console summary
    if final_eval_metrics:
        print_metrics_summary(final_eval_metrics, f"FINAL Evaluation Results for CLAP ({mode})")
        if subreddit_metrics:
            print_subreddit_comparison(subreddit_metrics)

    wandb.finish()
    print(f"Wrote {mode} results to {output_jsonl}")

def main():
    wandb.login()
    for mode in MODES:
        run_mode(mode, num_samples=NUM_SAMPLES)
    print("\n🎉 All CLAP experiments completed.")

if __name__ == "__main__":
    main()
