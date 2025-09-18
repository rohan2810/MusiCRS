#!/usr/bin/env python3
"""Merge per-submission WAV folders from multiple roots into one target.

Each root is expected to contain subfolders like `<submission_id>/...wav`.
We copy missing subfolders to the target (skip if exists).

Example:
python merge_wav_dirs.py --roots data/part1/wav data/part2/wav --target data/full/wav --workers 8
"""
import os, shutil, argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

def copy_one(src: Path, dst: Path) -> str:
    try:
        if dst.exists():
            return f"[SKIP] {dst.name} exists"
        shutil.copytree(src, dst)
        return f"[COPIED] {src.name} -> {dst.name}"
    except Exception as e:
        return f"[ERROR] {src}: {e}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--roots', nargs='+', required=True)
    ap.add_argument('--target', required=True)
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()

    target = Path(args.target)
    target.mkdir(parents=True, exist_ok=True)

    tasks = []
    for r in args.roots:
        for sub in Path(r).iterdir():
            if sub.is_dir():
                tasks.append((sub, target / sub.name))

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for msg in ex.map(lambda t: copy_one(*t), tasks):
            print(msg)

if __name__ == '__main__':
    main()
