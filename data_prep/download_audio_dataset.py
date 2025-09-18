#!/usr/bin/env python3
"""Download audio and write metadata from a JSONL file of links.

Input JSONL format (one object per line) must contain at least:
- "link": the YouTube (or supported) URL string
- "submission_id": stable ID to group audio into subfolders (optional but recommended)

Outputs:
- metadata CSV and JSON mapping submissions to links
- per-submission WAV directory tree (one folder per submission_id)

Example:
python download_audio_dataset.py \\
  --input links.jsonl \\
  --out-root data/audio \\
  --yt-dlp /usr/local/bin/yt-dlp \\
  --max-threads 8

"""
from __future__ import annotations
import os, csv, json, re, subprocess, argparse
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from pathlib import Path

def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', '_', name)

def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def fetch_metadata(url: str, ytdlp: str) -> dict | None:
    cmd = [ytdlp, '-J', url]
    r = run(cmd)
    if r.returncode != 0:
        return None
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return None

def download_audio(url: str, ytdlp: str, out_dir: Path, yt_id: str, title: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = sanitize_filename(title)[:120]
    out_tmpl = str(out_dir / f"{yt_id}__{safe}.%(ext)s")
    # prefer wav (ffmpeg needed)
    cmd = [ytdlp, url, '-x', '--audio-format', 'wav', '-o', out_tmpl]
    run(cmd)

def worker(item: dict, out_root: Path, ytdlp: str, csv_writer, csv_lock: Lock) -> None:
    url = item.get('link') or item.get('url')
    subid = item.get('submission_id') or item.get('id') or 'unknown'
    if not url:
        return
    meta = fetch_metadata(url, ytdlp)
    if not meta:
        return
    yt_id = meta.get('id', 'unknown')
    title = meta.get('title', 'unknown')
    subdir = out_root / 'wav' / str(subid)
    download_audio(url, ytdlp, subdir, yt_id, title)
    row = {
        'submission_id': subid,
        'yt_id': yt_id,
        'title': title,
        'webpage_url': meta.get('webpage_url', url),
        'duration': meta.get('duration'),
        'channel': (meta.get('channel') or meta.get('uploader')),
    }
    with csv_lock:
        csv_writer.writerow(row)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='JSONL file of {link, submission_id, ...}')
    ap.add_argument('--out-root', required=True, help='Output root directory')
    ap.add_argument('--yt-dlp', default='yt-dlp', help='Path to yt-dlp')
    ap.add_argument('--max-threads', type=int, default=8)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    meta_dir = out_root / 'meta'
    meta_dir.mkdir(parents=True, exist_ok=True)

    csv_path = meta_dir / 'metadata.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=['submission_id','yt_id','title','webpage_url','duration','channel'])
        writer.writeheader()
        lock = Lock()
        with open(args.input, 'r', encoding='utf-8') as fj:
            lines = [json.loads(line) for line in fj if line.strip()]
        with ThreadPoolExecutor(max_workers=args.max_threads) as ex:
            for item in lines:
                ex.submit(worker, item, out_root, args.yt_dlp, writer, lock)

    # also dump submission->links for convenience
    mapping = {}
    for row in csv.DictReader(open(csv_path, encoding='utf-8')):
        mapping.setdefault(row['submission_id'], []).append(row['webpage_url'])
    with open(meta_dir / 'submission_to_links.json', 'w', encoding='utf-8') as fj:
        json.dump(mapping, fj, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()
