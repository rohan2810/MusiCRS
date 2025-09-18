#!/usr/bin/env python3
"""Batch-convert audio files to WAV using ffmpeg via `ffmpeg -i` through `pydub` or `ffmpeg` directly.

Example:
python convert_to_wav.py --in-dir data/audio/raw --extensions .m4a .webm .mp3 --workers 4
"""
import os, argparse, subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

def convert_one(path: Path) -> str:
    out = path.with_suffix('.wav')
    if out.exists():
        return f"[SKIP] {out.name} exists"
    cmd = ['ffmpeg', '-y', '-i', str(path), str(out)]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode == 0:
        return f"[OK] {path.name} -> {out.name}"
    return f"[ERROR] {path.name}: {r.stderr.splitlines()[-1] if r.stderr else 'unknown'}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-dir', required=True)
    ap.add_argument('--extensions', nargs='+', default=['.m4a','.webm','.mp3'])
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()

    files = [p for p in Path(args.in_dir).rglob('*') if p.suffix.lower() in set(args.extensions)]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for msg in ex.map(convert_one, files):
            print(msg)

if __name__ == '__main__':
    main()
