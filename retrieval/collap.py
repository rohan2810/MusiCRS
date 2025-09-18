
import os
import json
import random
import tempfile
import traceback
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import soundfile as sf
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from transformers import GPT2Tokenizer, GPT2Model

import sys
from model import SALMONN

# ─── CONFIG ─────────────────────────────────────────────────────────
DEVICE              = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
INPUT_JSONL         = Path("")
AUDIO_DIR           = Path("")
OUTPUT_DIR          = Path("")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODES               = ["audio_only", "query_only", "audio_query"]
SR_SALMONN         = 16000      # SALMONN/Whisper expects ~16 kHz
MAX_CLIPS          = 10         # max files sampled per post
AUDIO_BUDGET_SECS  = 300.0      # total seconds per post
RANDOM_SEED        = 42


AUDIO_WEIGHT        = 0.5
TEXT_BATCH_SIZE     = 64
TEXT_MAX_LENGTH     = 128
WANDB_PROJECT       = "reddit-music-collap-final"
WANDB_ENTITY        = "musiCRS"
WANDB_TAGS          = ["collap-final", "music-ranking"]
from utils import (
    calculate_comprehensive_metrics,
    calculate_subreddit_metrics,
    print_metrics_summary,
    print_subreddit_comparison,
    load_and_mix_audio,
)

# ─── REPRO SETUP ──────────────────────────────────────────────
def set_all_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_all_seeds(RANDOM_SEED)
torch.backends.cudnn.benchmark = True

# ─── LOAD SALMONN (audio feature extractor) ───────────────────
print("Loading SALMONN…")
salmonn_model = SALMONN(
    ckpt="",
    whisper_path="",
    beats_path=""
).to(DEVICE)
salmonn_model.beats.float()
salmonn_model.eval()

# ─── CoLLAP Text Tower ────────────────────────────────────────
class CoLLAPTextModel(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.text_encoder = GPT2Model.from_pretrained("gpt2")
        self.tokenizer    = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.text_proj    = nn.Linear(self.text_encoder.config.hidden_size, embed_dim)
        self.audio_proj   = nn.Linear(2048, embed_dim)
        # shared temperature
        self.logit_scale  = nn.Parameter(torch.ones([]) * np.log(1/0.07))

    def forward(self, audio_feat, input_ids, attention_mask):
        # audio_feat: [1,2048], input_ids: [B,L], attention_mask: [B,L]
        idx = attention_mask.sum(dim=1) - 1  # index of last non-pad token
        text_h = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state                   # [B,L, H]
        # gather the last token embedding per example
        text_feat = text_h[torch.arange(text_h.size(0)), idx]  # [B, H]
        text_feat = self.text_proj(text_feat)                  # [B, E]
        audio_feat = self.audio_proj(audio_feat)               # [1, E]
        text_feat  = F.normalize(text_feat, dim=-1)
        audio_feat = F.normalize(audio_feat, dim=-1)
        # return [B,1]
        return self.logit_scale.exp() * text_feat @ audio_feat.T

print("Loading CoLLAP text model…")
text_model = CoLLAPTextModel().to(DEVICE)
text_model.eval()


# ─── AUDIO EMBEDDING ──────────────────────────────────────────
@torch.no_grad()
def extract_audio_embedding_mixed(
    audio_paths: List[Path],
    sr: int = SR_SALMONN,
    max_clips: int = MAX_CLIPS,
    audio_budget_secs: float = AUDIO_BUDGET_SECS,
    temp_dir: Path = OUTPUT_DIR / "tmp_segments",
) -> Optional[torch.Tensor]:
    mixed = load_and_mix_audio(audio_paths, sr=sr, max_clips=max_clips, audio_budget_secs=audio_budget_secs)
    if mixed is None or mixed.size == 0:
        return None

    temp_dir.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(prefix="mix_", suffix=".wav", dir=str(temp_dir), delete=False)
    tmp_path = tmp.name
    tmp.close()

    np_feat = None
    try:
        sf.write(tmp_path, mixed, sr)
        np_feat = salmonn_model.extract_auditory_feature(tmp_path)  # expected [S,F,2048] or [T,2048]
    except Exception as e:
        print(f"[extract_audio_embedding_mixed] SALMONN failed on {tmp_path}: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    if np_feat is None:
        return None

    t = torch.from_numpy(np_feat).float().to(DEVICE)
    # Mean-pool over time/frame dims -> [1, 2048]
    if t.ndim == 3:
        t = t.mean(dim=1).mean(dim=0, keepdim=True)
    elif t.ndim == 2:
        t = t.mean(dim=0, keepdim=True)
    else:
        return None

    return t  # [1, 2048]

# ─── TEXT EMBEDDINGS (BATCHED) ───────────────────────────────
@torch.no_grad()
def build_text_embeddings(texts: List[str], batch_size: int = TEXT_BATCH_SIZE, max_length: int = TEXT_MAX_LENGTH) -> Optional[torch.Tensor]:
    if not texts:
        return None

    outs: List[torch.Tensor] = []
    tok = text_model.tokenizer

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tok(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        h = text_model.text_encoder(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"]
        ).last_hidden_state                    # [B, L, H]
        # index of last non-pad token
        idx = enc["attention_mask"].sum(dim=1) - 1
        feats = h[torch.arange(h.size(0), device=h.device), idx]  # [B, H]
        proj  = text_model.text_proj(feats)                       # [B, E]
        outs.append(F.normalize(proj, dim=-1))

    return torch.cat(outs, dim=0)  # (N, E)

# ─── SCORING / RANKING ───────────────────────────────────────
def rank_one(
    cands: List[str],
    cand_embs: torch.Tensor,             # (N, E)
    audio_emb: Optional[torch.Tensor],   # (1, E) normalized (after proj) or None
    query_emb: Optional[torch.Tensor],   # (1, E) normalized or None
    mode: str,
    audio_weight: float = AUDIO_WEIGHT
) -> List[str]:
    if len(cands) == 0:
        return []

    if mode == "audio_only":
        if audio_emb is None:
            return []
        sims = (audio_emb @ cand_embs.T).squeeze(0)  # (N,)

    elif mode == "query_only":
        if query_emb is None:
            return []
        sims = (query_emb @ cand_embs.T).squeeze(0)

    elif mode == "audio_query":
        if audio_emb is None and query_emb is None:
            return []
        sims_a = (audio_emb @ cand_embs.T).squeeze(0) if audio_emb is not None else None
        sims_q = (query_emb @ cand_embs.T).squeeze(0) if query_emb is not None else None
        if sims_a is not None and sims_q is not None:
            sims = audio_weight * sims_a + (1.0 - audio_weight) * sims_q
        else:
            sims = sims_a if sims_a is not None else sims_q
    else:
        raise ValueError(f"Unknown mode: {mode}")

    idxs = torch.argsort(sims, descending=True).tolist()
    return [cands[i] for i in idxs]

# ─── MAIN EVAL LOOP (per-mode, like CLAP) ─────────────────────
def run_mode(mode: str, num_samples: int = -1):
    set_all_seeds(RANDOM_SEED)

    out_path = OUTPUT_DIR / f"{mode}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # W&B (one run per mode)
    run_name = f"{mode}_collap_{random.randint(0, 99999):05d}"
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=run_name,
        config={
            "mode": mode,
            "num_samples": num_samples,
        },
        tags=WANDB_TAGS + [mode],
    )

    with open(INPUT_JSONL, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if num_samples != -1:
        lines = lines[:num_samples]

    all_predictions, all_ground_truth, all_subreddits = [], [], []

    with open(out_path, "w", encoding="utf-8") as wf:
        for i, line in enumerate(tqdm(lines, desc=f"Processing ({mode})")):
            try:
                ex = json.loads(line)
                sid   = ex["name"]
                query = ex.get("query", "").strip()
                cands = ex.get("final_candidates") or ex.get("candidates") or []
                if not cands:
                    continue

                # Gather audio files (may be empty)
                wav_paths = sorted((AUDIO_DIR / sid).glob("*.wav"))

                # Mode-specific gating
                if mode == "audio_only" and not wav_paths:
                    continue
                if mode == "query_only" and not query:
                    continue
                if mode == "audio_query" and (not wav_paths and not query):
                    continue

                # Candidate text embs
                cand_embs = build_text_embeddings(cands)  # (N, E)
                if cand_embs is None or cand_embs.shape[0] == 0:
                    continue

                # Audio emb (project + norm)
                audio_emb = None
                if wav_paths and mode in ("audio_only", "audio_query"):
                    raw_audio = extract_audio_embedding_mixed(
                        wav_paths,
                        sr=SR_SALMONN,
                        max_clips=MAX_CLIPS,
                        audio_budget_secs=AUDIO_BUDGET_SECS,
                        temp_dir=OUTPUT_DIR / "tmp_segments",
                    )  # [1,2048] or None
                    if raw_audio is not None:
                        audio_emb = F.normalize(text_model.audio_proj(raw_audio), dim=-1)  # [1, E]

                # Query emb
                query_emb = None
                if query and mode in ("query_only", "audio_query"):
                    q = build_text_embeddings([query])
                    if q is not None:
                        query_emb = q[0:1]  # [1, E]

                ranked = rank_one(
                    cands=cands,
                    cand_embs=cand_embs,
                    audio_emb=audio_emb,
                    query_emb=query_emb,
                    mode=mode,
                    audio_weight=AUDIO_WEIGHT
                )
                if not ranked:
                    continue

                record = ex.copy()
                record["predicted"] = ranked
                wf.write(json.dumps(record, ensure_ascii=False) + "\n")

                # For metrics
                gt = ex.get("limited_gt", [])
                all_predictions.append(ranked)
                all_ground_truth.append(gt)
                all_subreddits.append(ex.get("source_subreddit", "unknown"))

            except Exception as e:
                print(f"Error on sample {i} (sid={ex.get('name', 'unknown')}): {e}")
                traceback.print_exc()
                continue

    # Metrics + W&B
    final_eval = calculate_comprehensive_metrics(all_predictions, all_ground_truth)
    subr_eval  = calculate_subreddit_metrics(all_predictions, all_ground_truth, all_subreddits)

    log_dict = {f"FINAL_{k}": v for k, v in final_eval.items()}
    for subr, metrics in subr_eval.items():
        for k, v in metrics.items():
            log_dict[f"SUBREDDIT_{subr}_{k}"] = v
    wandb.log(log_dict)

    if final_eval:
        print_metrics_summary(final_eval, f"FINAL Evaluation Results for CoLLAP ({mode})")
        if subr_eval:
            print_subreddit_comparison(subr_eval)

    wandb.finish()
    print(f"Wrote {mode} results to {out_path}")

def main():
    wandb.login()
    for m in MODES:
        run_mode(m, num_samples=-1)   # set to a number for smaller runs
    print("\n🎉 All CoLLAP experiments completed.")

if __name__ == "__main__":
    main()
