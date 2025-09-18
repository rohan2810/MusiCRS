#!/usr/bin/env python3
"""Remove non-WAV files under a directory tree."""
import os, argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='Directory containing audio files')
    ap.add_argument('--keep', nargs='+', default=['.wav'], help='Extensions to keep')
    args = ap.parse_args()

    keep = set(e.lower() for e in args.keep)
    for path in Path(args.root).rglob('*'):
        if path.is_file() and path.suffix.lower() not in keep:
            try:
                path.unlink()
                print(f"[DELETED] {path}")
            except Exception as e:
                print(f"[ERROR] {path}: {e}")

if __name__ == '__main__':
    main()
