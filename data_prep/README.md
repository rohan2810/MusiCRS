# Data Preparation Scripts

This directory provides lightweight, CLI-based Python utilities to prepare Reddit conversation data and associated audio for the **MusiCRS** project. All scripts expose arguments via `argparse` and can be chained into a reproducible pipeline.

---

## Prerequisites

- Python packages:
  - `duckdb`, `pandas`, `yt-dlp`
- System tools:
  - `zstd` (CLI) for decompressing Reddit dumps (used by `extract_month_subreddit.py`)
  - `ffmpeg` for audio conversion (used by `convert_to_wav.py` and indirectly by `yt-dlp`)

Suggested installs (macOS/Homebrew):
```bash
brew install zstd ffmpeg
pip install duckdb pandas yt-dlp
```

---

## Scripts

- **`extract_month_subreddit.py`**  
  Extract comments and submissions for a given subreddit from monthly Reddit dumps (`RC_YYYY-MM.zst`, `RS_YYYY-MM.zst`) using DuckDB + zstd. Outputs CSV or Parquet files.
  - Inputs: `--comments-root` (RC_*.zst), `--submissions-root` (RS_*.zst)
  - Output: `comments_YYYY-MM.{csv|parquet}`, `submissions_YYYY-MM.{csv|parquet}` in `--out`.
  - Example:
    ```bash
    python extract_month_subreddit.py \
      --month 2019-01 \
      --subreddit musictheory \
      --comments-root /data/reddit/comments \
      --submissions-root /data/reddit/submissions \
      --tmp-dir ./.tmp/reddit_mt \
      --out ./data/reddit/musictheory \
      --format csv
    ```

- **`filter_threads.py`**  
  Filter subreddit submissions and comments with configurable rules: subreddit whitelist, minimum comment length, minimum number of direct replies, optional YouTube link requirement.
  - Inputs: `--submissions` CSV, `--comments` CSV
  - Output: filtered submissions CSV (`--out`)
  - Example:
    ```bash
    python filter_threads.py \
      --submissions ./data/reddit/musictheory/submissions_2019-01.csv \
      --comments ./data/reddit/musictheory/comments_2019-01.csv \
      --subreddits popheads hiphopheads \
      --min-comment-len 120 \
      --min-replies 3 \
      --require-youtube \
      --out ./data/reddit/filtered_2019-01.csv
    ```

- **`download_audio_dataset.py`**  
  Download YouTube audio referenced in threads and save as `.wav` with metadata (CSV + JSON mapping submissions to links).
  - Input JSONL fields: `link` (or `url`), `submission_id` (recommended)
  - Outputs in `--out-root`:
    - `meta/metadata.csv`, `meta/submission_to_links.json`
    - `wav/<submission_id>/*.wav`
  - Example:
    ```bash
    python download_audio_dataset.py \
      --input ./data/links.jsonl \
      --out-root ./data/audio \
      --yt-dlp yt-dlp \
      --max-threads 8
    ```

- **`convert_to_wav.py`**  
  Batch convert audio files (`.m4a`, `.webm`, `.mp3`, etc.) into `.wav` format using ffmpeg.
  - Inputs: `--in-dir`, `--extensions`
  - Output: creates `.wav` beside each source file (skips if exists)
  - Example:
    ```bash
    python convert_to_wav.py \
      --in-dir ./data/audio/raw \
      --extensions .m4a .webm .mp3 \
      --workers 4
    ```

- **`clean_nonwav.py`**  
  Remove all non-WAV files under a specified directory tree.
  - Inputs: `--root`, optional `--keep` (default .wav)
  - Example:
    ```bash
    python clean_nonwav.py --root ./data/audio/wav --keep .wav
    ```

- **`merge_wav_dirs.py`**  
  Merge multiple per-submission WAV directory trees into a single consolidated dataset.
  - Inputs: `--roots` (directories with `<submission_id>/...wav`), `--target`
  - Example:
    ```bash
    python merge_wav_dirs.py \
      --roots ./data/part1/wav ./data/part2/wav \
      --target ./data/full/wav \
      --workers 8
    ```

---

## Example Workflow

```bash
# 1. Extract subreddit slice from monthly dumps
python extract_month_subreddit.py --month 2019-01 --subreddit musictheory \
  --comments-root /path/to/comments --submissions-root /path/to/submissions \
  --tmp-dir /tmp/work --out data/musictheory --format csv

# 2. Filter for high-quality threads
python filter_threads.py --submissions submissions.csv --comments comments.csv \
  --subreddits popheads hiphopheads --min-comment-len 120 --min-replies 3 \
  --require-youtube --out filtered.csv

# 3. Download referenced audio
python download_audio_dataset.py --input links.jsonl --out-root data/audio \
  --yt-dlp yt-dlp --max-threads 8

# 4. Convert to WAV
python convert_to_wav.py --in-dir data/raw --extensions .m4a .webm .mp3 --workers 4

# 5. Clean up non-WAV files
python clean_nonwav.py --root data/wav

# 6. Merge multiple parts
python merge_wav_dirs.py --roots data/part1/wav data/part2/wav --target data/full/wav --workers 8