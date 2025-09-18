#!/usr/bin/env python3
"""
eval_top5_overlap.py 

Evaluate generative model results across ALL models under a results root.
For each model directory (e.g., generative/results/QWEN_AUDIO_FINAL_FINAL/), it looks for
JSONL files corresponding to three configs: query_only, audio_only, audio_query.

Per-example, it computes:
  - overlap_{cfg}_topK     (how many ground-truth items appear in top-K predictions)
  - hit_{cfg}@K            (>=1 ground-truth in top-K predictions)
  - aq_better_than_both    (audio_query strictly > both others)
  - aq_ties_best           (audio_query equals the max of the others)

Aggregates:
  - overall (means & counts)
  - by subreddit

Outputs:
  out_dir/
    <model_name>/
      per_example.csv
      overall.csv
      by_subreddit.csv
    _combined/
      summary_overall.csv
      summary_by_subreddit.csv
      summary_overall.md   # quick Markdown table for pasting in docs

Usage:
  python eval_top5_overlap.py \
    --results-root generative/results \
    --out-dir eval_stats \
    --k 5
"""

from __future__ import annotations
import os
import re
import json
import argparse
from pathlib import Path
import pandas as pd


# ----------------------------
# I/O helpers
# ----------------------------
def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except json.JSONDecodeError:
                # skip malformed lines
                continue
    return rows


def find_config_files(model_dir: Path) -> dict[str, Path]:
    """
    Return mapping {'query_only': path, 'audio_only': path, 'audio_query': path}
    by scanning *.jsonl filenames (case-insensitive). Supports minor variants like
    'audio+query' / 'audio_query'.
    """
    out: dict[str, Path] = {}
    for f in model_dir.glob("*.jsonl"):
        name = f.name.lower()
        if "query_only" in name:
            out.setdefault("query_only", f)
        elif "audio_only" in name:
            out.setdefault("audio_only", f)
        elif ("audio_query" in name) or ("audio+query" in name):
            out.setdefault("audio_query", f)
    return out


# ----------------------------
# Metrics helpers
# ----------------------------
def norm_list(x):
    return [str(t).strip().lower() for t in x] if isinstance(x, list) else []


def topk_overlap(pred, gt, k: int = 5) -> int:
    p = set(norm_list(pred[:k]))
    g = set(norm_list(gt))
    return len(p & g)


def hit_at_k(pred, gt, k: int = 5) -> int:
    return 1 if topk_overlap(pred, gt, k) > 0 else 0


# ----------------------------
# Alignment and DataFrames
# ----------------------------
def align_by_id(datasets: dict[str, list[dict]]) -> list[dict]:
    """
    Align entries by 'name' (submission/thread id).
    datasets: {'query_only': [...], 'audio_only': [...], 'audio_query': [...]}
    Returns list of aligned dicts with fields:
      name, subreddit, query, ground_truth,
      preds_query_only, preds_audio_only, preds_audio_query
    Only includes examples that exist in ALL THREE configs and have non-empty ground_truth.
    """
    by_id: dict[str, dict] = {}

    for cfg, lst in datasets.items():
        for d in lst:
            sid = d.get("name")
            if not sid:
                continue
            rec = by_id.setdefault(sid, {"name": sid})
            # carry metadata where available
            if rec.get("subreddit") is None and d.get("source_subreddit") is not None:
                rec["subreddit"] = d.get("source_subreddit")
            if (not rec.get("query")) and d.get("query"):
                rec["query"] = d.get("query")
            if "ground_truth" not in rec and isinstance(d.get("ground_truth"), list):
                rec["ground_truth"] = d.get("ground_truth", [])
            rec[f"preds_{cfg}"] = d.get("predicted", [])

    aligned = []
    for sid, r in by_id.items():
        if (
            isinstance(r.get("ground_truth"), list) and len(r["ground_truth"]) > 0
            and isinstance(r.get("preds_query_only"), list)
            and isinstance(r.get("preds_audio_only"), list)
            and isinstance(r.get("preds_audio_query"), list)
        ):
            aligned.append(r)
    return aligned


def per_example_metrics(records: list[dict], k: int) -> pd.DataFrame:
    rows = []
    for r in records:
        gt = r["ground_truth"]
        qo = r["preds_query_only"]
        ao = r["preds_audio_only"]
        aq = r["preds_audio_query"]

        overlap_qo = topk_overlap(qo, gt, k)
        overlap_ao = topk_overlap(ao, gt, k)
        overlap_aq = topk_overlap(aq, gt, k)

        row = {
            "name": r["name"],
            "subreddit": r.get("subreddit"),
            f"overlap_qo_top{k}": overlap_qo,
            f"overlap_ao_top{k}": overlap_ao,
            f"overlap_aq_top{k}": overlap_aq,
            f"hit_qo@{k}": hit_at_k(qo, gt, k),
            f"hit_ao@{k}": hit_at_k(ao, gt, k),
            f"hit_aq@{k}": hit_at_k(aq, gt, k),
            "aq_better_than_both": int(overlap_aq > max(overlap_qo, overlap_ao)),
            "aq_ties_best": int(overlap_aq == max(overlap_qo, overlap_ao)),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_stats(df: pd.DataFrame, k: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame([{
            "n_examples": 0,
            f"hit@{k}_qo": 0.0, f"hit@{k}_ao": 0.0, f"hit@{k}_aq": 0.0,
            f"mean_overlap_qo_top{k}": 0.0, f"mean_overlap_ao_top{k}": 0.0, f"mean_overlap_aq_top{k}": 0.0,
            "aq_better_count": 0, "aq_ties_best_count": 0
        }])

    return pd.DataFrame({
        "n_examples": [len(df)],
        f"hit@{k}_qo": [df[f"hit_qo@{k}"].mean()],
        f"hit@{k}_ao": [df[f"hit_ao@{k}"].mean()],
        f"hit@{k}_aq": [df[f"hit_aq@{k}"].mean()],
        f"mean_overlap_qo_top{k}": [df[f"overlap_qo_top{k}"].mean()],
        f"mean_overlap_ao_top{k}": [df[f"overlap_ao_top{k}"].mean()],
        f"mean_overlap_aq_top{k}": [df[f"overlap_aq_top{k}"].mean()],
        "aq_better_count": [df["aq_better_than_both"].sum()],
        "aq_ties_best_count": [df["aq_ties_best"].sum()],
    })


def aggregate_by_subreddit(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    Return per-subreddit aggregates with a single 'subreddit' column (no duplicates).
    Also avoids the FutureWarning by not resetting/inserting duplicate columns.
    """
    cols = [
        "subreddit", "n_examples",
        f"hit@{k}_qo", f"hit@{k}_ao", f"hit@{k}_aq",
        f"mean_overlap_qo_top{k}", f"mean_overlap_ao_top{k}", f"mean_overlap_aq_top{k}",
        "aq_better_count", "aq_ties_best_count"
    ]
    if df.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    # (group_keys=False keeps output tidy on newer pandas; older versions ignore it)
    for sub, g in df.groupby("subreddit", dropna=False, group_keys=False):
        agg = aggregate_stats(g, k)
        agg.insert(0, "subreddit", sub)
        rows.append(agg)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=cols)


def to_md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_no data_\n"
    return df.to_markdown(index=False)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True, help="Path like generative/results")
    ap.add_argument("--out-dir", required=True, help="Directory to write outputs")
    ap.add_argument("--k", type=int, default=5, help="Top-K cutoff (default=5)")
    args = ap.parse_args()

    results_root = Path(args.results_root)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    model_dirs = [p for p in results_root.iterdir() if p.is_dir()]
    combined_overall = []
    combined_by_sub = []

    for model_dir in sorted(model_dirs, key=lambda p: p.name.lower()):
        model_name = model_dir.name
        cfg_files = find_config_files(model_dir)

        required = {"query_only", "audio_only", "audio_query"}
        if not required.issubset(cfg_files.keys()):
            # Skip models missing any required config files
            continue

        datasets = {cfg: load_jsonl(path) for cfg, path in cfg_files.items()}
        aligned = align_by_id(datasets)
        df_examples = per_example_metrics(aligned, args.k)
        df_overall = aggregate_stats(df_examples, args.k)
        df_by_sub = aggregate_by_subreddit(df_examples, args.k)

        # Tag with model name
        df_overall.insert(0, "model", model_name)
        if not df_by_sub.empty:
            df_by_sub.insert(0, "model", model_name)

        # Write per-model outputs
        model_out = out_root / model_name
        model_out.mkdir(parents=True, exist_ok=True)
        df_examples.to_csv(model_out / "per_example.csv", index=False)
        df_overall.to_csv(model_out / "overall.csv", index=False)
        df_by_sub.to_csv(model_out / "by_subreddit.csv", index=False)

        combined_overall.append(df_overall)
        if not df_by_sub.empty:
            combined_by_sub.append(df_by_sub)

    # Combined summaries
    combined_dir = out_root / "_combined"
    combined_dir.mkdir(exist_ok=True)

    if combined_overall:
        summary_overall = pd.concat(combined_overall, ignore_index=True)
    else:
        summary_overall = pd.DataFrame(columns=[
            "model", "n_examples",
            f"hit@{args.k}_qo", f"hit@{args.k}_ao", f"hit@{args.k}_aq",
            f"mean_overlap_qo_top{args.k}", f"mean_overlap_ao_top{args.k}", f"mean_overlap_aq_top{args.k}",
            "aq_better_count", "aq_ties_best_count"
        ])

    if combined_by_sub:
        summary_by_sub = pd.concat(combined_by_sub, ignore_index=True)
    else:
        summary_by_sub = pd.DataFrame(columns=[
            "model", "subreddit", "n_examples",
            f"hit@{args.k}_qo", f"hit@{args.k}_ao", f"hit@{args.k}_aq",
            f"mean_overlap_qo_top{args.k}", f"mean_overlap_ao_top{args.k}", f"mean_overlap_aq_top{args.k}",
            "aq_better_count", "aq_ties_best_count"
        ])

    summary_overall.to_csv(combined_dir / "summary_overall.csv", index=False)
    summary_by_sub.to_csv(combined_dir / "summary_by_subreddit.csv", index=False)

    md = "# Overall Summary (per model)\n\n" + to_md_table(summary_overall) + "\n\n"
    md += "# By Subreddit Summary\n\n" + to_md_table(summary_by_sub) + "\n"
    (combined_dir / "summary_overall.md").write_text(md, encoding="utf-8")

    print(f"[OK] Wrote per-model stats to: {out_root}")
    print(f"[OK] Combined CSVs: {combined_dir / 'summary_overall.csv'} , {combined_dir / 'summary_by_subreddit.csv'}")
    print(f"[OK] Markdown summary: {combined_dir / 'summary_overall.md'}")


if __name__ == "__main__":
    main()