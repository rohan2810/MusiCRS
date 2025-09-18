#!/usr/bin/env python3
"""Filter Reddit submissions/comments for dataset curation.

Rules are configurable and cover:
- subreddit whitelist
- minimum comment length
- minimum number of direct replies to the submission
- presence of YouTube links in comments

Example:
python filter_threads.py \\
  --submissions submissions.csv \\
  --comments comments.csv \\
  --subreddits popheads hiphopheads \\
  --min-comment-len 120 \\
  --min-replies 3 \\
  --require-youtube \\
  --out filtered_submissions.csv
"""
from __future__ import annotations
import re, argparse
from pathlib import Path
import pandas as pd

YOUTUBE_REGEX = re.compile(
    r'(?:https?://)?(?:www\.)?'
    r'(?:youtube\.com/watch\?v=|youtu\.be/)'
    r'([A-Za-z0-9_-]{6,})'
)

def has_youtube(text: str) -> bool:
    if not isinstance(text, str):
        return False
    return bool(YOUTUBE_REGEX.search(text))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--submissions', required=True, help='CSV with submissions')
    ap.add_argument('--comments', required=True, help='CSV with comments')
    ap.add_argument('--subreddits', nargs='+', default=['popheads','hiphopheads'])
    ap.add_argument('--min-comment-len', type=int, default=100)
    ap.add_argument('--min-replies', type=int, default=3)
    ap.add_argument('--require-youtube', action='store_true')
    ap.add_argument('--out', required=True, help='Output CSV of kept submissions')
    args = ap.parse_args()

    subs = pd.read_csv(args.submissions)
    coms = pd.read_csv(args.comments)

    # keep only whitelisted subreddits
    subs = subs[ subs['source_subreddit'].isin(args.subreddits) ].copy()

    # long comments only
    coms['is_long'] = coms['body'].astype(str).str.len() >= args.min_comment_len

    # direct replies to the submission: parent_id should equal submission 'name' (e.g., t3_XXXX)
    # compute per-submission counts of long replies
    counts = (coms[coms['is_long']]
              .groupby('parent_id')
              .size()
              .rename('n_replies'))

    subs = subs.merge(counts, how='left', left_on='name', right_index=True)
    subs['n_replies'] = subs['n_replies'].fillna(0).astype(int)
    subs = subs[ subs['n_replies'] >= args.min_replies ].copy()

    if args.require_youtube:
        # Keep submissions that have at least one comment under them with a YouTube link
        coms['has_yt'] = coms['body'].apply(has_youtube)
        ymap = (coms[coms['has_yt']]
                .groupby('parent_id')
                .size()
                .rename('n_youtube'))
        subs = subs.merge(ymap, how='left', left_on='name', right_index=True)
        subs['n_youtube'] = subs['n_youtube'].fillna(0).astype(int)
        subs = subs[ subs['n_youtube'] > 0 ]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    subs.to_csv(args.out, index=False)
    print(f"[OK] Kept {len(subs)} submissions → {args.out}")

if __name__ == '__main__':
    main()
