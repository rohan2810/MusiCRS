#!/usr/bin/env python3
"""Extract all comments (and their parent submissions) for a subreddit from a monthly
Reddit dump using DuckDB + zstd.

Requires: duckdb>=0.10 (Python), zstd CLI on PATH.

Example:
python extract_month_subreddit.py \\
  --month 2019-01 \\
  --subreddit musictheory \\
  --comments-root /path/to/reddit/comments \\
  --submissions-root /path/to/reddit/submissions \\
  --tmp-dir /tmp/reddit_mt \\
  --out data/reddit/musictheory \\
  --format csv
"""
from __future__ import annotations
import os, shutil, subprocess, argparse, sys
from pathlib import Path
import duckdb

def which(cmd: str) -> str | None:
    for p in os.environ.get('PATH', '').split(os.pathsep):
        c = Path(p) / cmd
        if c.exists() and os.access(c, os.X_OK):
            return str(c)
    return None

def purge_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for entry in path.iterdir():
        try:
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()
        except Exception as e:
            print(f"[WARN] Could not delete {entry}: {e}")

def unzip_zst(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    print(f"» Decompressing {src.name} …")
    cmd = ["zstd", "-d", "--memory=2048MB", "--keep", "-o", str(dst), str(src)]
    subprocess.run(cmd, check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--month', required=True, help='YYYY-MM, e.g., 2019-01')
    ap.add_argument('--subreddit', required=True, help='Target subreddit (no r/)')
    ap.add_argument('--comments-root', required=True, help='Dir containing RC_YYYY-MM.zst')
    ap.add_argument('--submissions-root', required=True, help='Dir containing RS_YYYY-MM.zst')
    ap.add_argument('--tmp-dir', required=True, help='Temporary working directory')
    ap.add_argument('--out', required=True, help='Output directory')
    ap.add_argument('--format', choices=['csv','parquet'], default='csv')
    args = ap.parse_args()

    if which('zstd') is None:
        print('[ERROR] zstd not found on PATH.')
        sys.exit(1)

    month = args.month
    sub = args.subreddit

    comments_zst = Path(args.comments_root) / f"RC_{month}.zst"
    submissions_zst = Path(args.submissions_root) / f"RS_{month}.zst"
    if not comments_zst.exists():
        sys.exit(f"[ERROR] Missing {comments_zst}")
    if not submissions_zst.exists():
        sys.exit(f"[ERROR] Missing {submissions_zst}")

    tmp = Path(args.tmp_dir)
    print("» Cleaning temporary cache …")
    tmp.mkdir(parents=True, exist_ok=True)
    for p in tmp.iterdir():
        try:
            p.unlink() if p.is_file() else shutil.rmtree(p)
        except Exception as e:
            print(f"[WARN] Could not delete {p}: {e}")

    comments_jsonl = tmp / f"RC_{month}.jsonl"
    submissions_jsonl = tmp / f"RS_{month}.jsonl"
    unzip_zst(comments_zst, comments_jsonl)
    unzip_zst(submissions_zst, submissions_jsonl)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    print(f"» Filtering comments for r/{sub} …")
    con.execute("""        CREATE OR REPLACE TABLE filtered_comments AS
        SELECT * FROM read_json(?) WHERE subreddit = ?;
    """, [str(comments_jsonl), sub])

    print("» Filtering corresponding submissions …")
    con.execute("""        CREATE OR REPLACE TABLE filtered_submissions AS
        WITH submissions_raw AS (
            SELECT *, 't3_' || id AS name
            FROM read_json(?, ignore_errors = TRUE)
        )
        SELECT sr.*
        FROM submissions_raw sr
        WHERE sr.name IN (SELECT DISTINCT link_id FROM filtered_comments);
    """, [str(submissions_jsonl)])

    comments_out = outdir / f"comments_{month}.{args.format}"
    subs_out     = outdir / f"submissions_{month}.{args.format}"

    if args.format == 'csv':
        print("» Saving CSV …")
        con.execute("COPY filtered_comments TO ? (HEADER, DELIMITER ',');", [str(comments_out)])
        con.execute("COPY filtered_submissions TO ? (HEADER, DELIMITER ',');", [str(subs_out)])
    else:
        print("» Saving Parquet …")
        con.execute("COPY filtered_comments TO ? (FORMAT 'parquet');", [str(comments_out)])
        con.execute("COPY filtered_submissions TO ? (FORMAT 'parquet');", [str(subs_out)])

    print("Done →", outdir)

if __name__ == '__main__':
    main()
