#!/usr/bin/env python3
"""
candidate_pool_selection_enhanced.py

Enhanced music candidate selection with:
1. Smart YouTube title parsing (song name extraction)  
2. Multi-source Wikipedia lookups (song-focused + original entity)
3. Intelligent song-centric summary combination
4. Ground truth prioritization (music entities first)
5. Robust error handling and fallbacks

Note: All music entities are treated as songs, not artists.
"""

import os
import re
import sys
import json
import shutil
import threading
import requests
import faiss
import numpy as np
import unicodedata
import random
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Set random seed for reproducible candidate shuffling
random.seed(42)

# ── CONFIG ───────────────────────────────────────────────────
FILE_PATH    = Path("")
OUTPUT_PATH  = FILE_PATH.with_stem(FILE_PATH.stem + "_with_candidates_enhanced")
MODEL_NAME   = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE       = "cuda"

# YouTube Data API settings
API_KEY      = os.getenv("YOUTUBE_API_KEY")
if not API_KEY:
    print("Warning: No YOUTUBE_API_KEY set, will skip YouTube enhancement")
    USE_YOUTUBE = False
else:
    USE_YOUTUBE = True

SUMMARY_LEN  = 512       # max chars per summary
WORKERS      = 8         # concurrent workers
POOL_R       = 100       # FAISS pool size
TOP_K        = 20        # final candidate count

# Cache dir for enhanced summaries + FAISS index
CACHE_DIR    = Path(".enhanced_cache")
SUM_FILE     = CACHE_DIR / "enhanced_summaries.json"
IDX_FILE     = CACHE_DIR / "index.faiss"
MAP_FILE     = CACHE_DIR / "idx2id.json"
CACHE_DIR.mkdir(exist_ok=True)

# ── MODEL ────────────────────────────────────────────────────
encoder = SentenceTransformer(MODEL_NAME, device=DEVICE)

# ── THREAD-LOCAL HELPERS ─────────────────────────────────────
thread_local = threading.local()

def get_yt_client():
    if not USE_YOUTUBE:
        return None
    if not hasattr(thread_local, "yt_client"):
        thread_local.yt_client = build("youtube", "v3", developerKey=API_KEY)
    return thread_local.yt_client

def get_wiki_session():
    if not hasattr(thread_local, "wiki_sess"):
        sess = requests.Session()
        sess.headers.update({"User-Agent": "MusicRecResearch/1.0"})
        thread_local.wiki_sess = sess
    return thread_local.wiki_sess

# ── LOAD POSTS & BUILD ENTITY POOL ──────────────────────────
posts = []
entity_pool = set()

print("[1] Reading JSONL…")
with FILE_PATH.open(encoding="utf-8") as fp:
    for line in fp:
        post = json.loads(line)
        posts.append(post)
        entity_pool.update(post.get("combined_comment_entities", []))

entity_pool = sorted(entity_pool)
print(f"    {len(posts)} posts → {len(entity_pool)} unique entities")

# ── MUSIC FILTERING UTILITIES ───────────────────────────────
MUSIC_KEYWORDS = {
    'song', 'track', 'single', 'album', 'music', 'band', 'artist', 'singer',
    'musician', 'vocal', 'guitar', 'piano', 'drums', 'bass', 'symphony',
    'orchestra', 'concert', 'performance', 'recording', 'studio', 'label',
    'genre', 'rock', 'pop', 'jazz', 'classical', 'hip hop', 'rap', 'country',
    'folk', 'electronic', 'dance', 'blues', 'metal', 'punk', 'reggae',
    'disco', 'soundtrack', 'musical', 'opera', 'choir', 'composition',
    'melody', 'harmony', 'rhythm', 'lyrics', 'verse', 'chorus', 'bridge'
}

NON_MUSIC_KEYWORDS = {
    'politics', 'government', 'military', 'war', 'battle', 'weapon', 'army',
    'science', 'physics', 'chemistry', 'biology', 'mathematics', 'technology',
    'computer', 'software', 'programming', 'sports', 'football', 'basketball',
    'baseball', 'soccer', 'olympics', 'game', 'player', 'team', 'league',
    'food', 'recipe', 'restaurant', 'cooking', 'medicine', 'doctor', 'hospital',
    'disease', 'treatment', 'geography', 'country', 'city', 'mountain', 'river',
    'space', 'planet', 'star', 'galaxy', 'astronaut', 'nasa', 'satellite'
}

def is_music_related(text: str, title: str = "") -> bool:
    """
    Determine if a text/title is music-related using keyword analysis
    """
    if not text and not title:
        return False
    
    # Combine title and text for analysis
    full_text = f"{title} {text}".lower()
    
    # Count music vs non-music keyword occurrences
    music_score = sum(1 for keyword in MUSIC_KEYWORDS if keyword in full_text)
    non_music_score = sum(1 for keyword in NON_MUSIC_KEYWORDS if keyword in full_text)
    
    # Special patterns that strongly indicate music
    music_patterns = [
        r'\b(song|track|single|album)\b',
        r'\b(released|recorded|produced)\b.*\b(by|in)\b',
        r'\b(music|musical|musician|singer|band|artist)\b',
        r'\b(guitar|piano|drums|bass|vocal|singing)\b',
        r'\b(rock|pop|jazz|blues|metal|punk|folk|country|hip hop|rap|electronic)\b',
        r'\b(chart|billboard|number.{1,3}hit|top.{1,3}\d+)\b',
        r'\b(album|ep|lp|compilation|soundtrack)\b',
        r'\b(lyrics|verse|chorus|bridge|melody|harmony|rhythm)\b'
    ]
    
    pattern_matches = sum(1 for pattern in music_patterns if re.search(pattern, full_text))
    
    # Decision logic
    if pattern_matches >= 2:  # Strong musical patterns
        return True
    elif music_score >= 3 and non_music_score <= 1:  # Clear music dominance
        return True
    elif music_score >= 2 and non_music_score == 0:  # Some music, no non-music
        return True
    elif non_music_score >= 2 and music_score <= 1:  # Clear non-music
        return False
    
    # Default: if unclear, lean towards including (false positives better than false negatives)
    return music_score > non_music_score or music_score >= 1

# ── SMART PARSING UTILITIES ──────────────────────────────────
def clean_text(text: str) -> str:
    """Clean text while preserving important music information"""
    if not text:
        return ""
    
    # Normalize Unicode characters first
    text = unicodedata.normalize('NFKC', text)
    
    # Remove common YouTube artifacts but preserve music info
    text = re.sub(r'\[.*?\]', '', text)  # Remove [Official Video], [HD], etc.
    text = re.sub(r'\((?:Official|HD|4K|Video|Audio|Lyric).*?\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_song_name(original_entity: str, youtube_title: str = "") -> dict:
    """
    Extract the song name from YouTube title and original entity.
    Focus purely on song identification since entities are always songs.
    """
    result = {'original': original_entity}
    
    # Try extracting song name from YouTube title
    if youtube_title:
        title = clean_text(youtube_title)
        
        # Pattern 1: "Artist - Song" or "Artist – Song" -> Extract Song
        dash_match = re.search(r'.+?\s*[–-]\s*(.+)', title)
        if dash_match:
            song_candidate = dash_match.group(1).strip()
            if len(song_candidate) > 2:  # Basic sanity check
                result['song'] = song_candidate
                return result
        
        # Pattern 2: "Song by Artist" -> Extract Song
        by_match = re.search(r'(.+?)\s+by\s+.+', title, re.IGNORECASE)
        if by_match:
            song_candidate = by_match.group(1).strip()
            if len(song_candidate) > 2:
                result['song'] = song_candidate
                return result
        
        # Pattern 3: "Artist: Song" or "Artist | Song" -> Extract Song
        colon_match = re.search(r'.+?\s*[:|]\s*(.+)', title)
        if colon_match:
            song_candidate = colon_match.group(1).strip()
            if len(song_candidate) > 2:
                result['song'] = song_candidate
                return result
        
        # Pattern 4: If no clear pattern, use cleaned title as song candidate
        if title and len(title) > 2:
            result['song'] = title
    
    # If YouTube parsing didn't work, try original entity
    if 'song' not in result:
        # Try to extract song from original entity patterns
        if ' - ' in original_entity:
            parts = original_entity.split(' - ', 1)
            if len(parts) == 2:
                result['song'] = parts[1].strip()  # Take the second part as song
        elif ' by ' in original_entity.lower():
            by_match = re.search(r'(.+?)\s+by\s+.+', original_entity, re.IGNORECASE)
            if by_match:
                result['song'] = by_match.group(1).strip()
        else:
            # Use original entity as song name
            result['song'] = original_entity
    
    return result

# ── WIKIPEDIA API UTILITIES ──────────────────────────────────
WIKI_API = "https://en.wikipedia.org/w/api.php"

def fetch_wikipedia_summary(title: str, max_length: int = 200) -> str:
    """Fetch Wikipedia summary for a given title"""
    try:
        sess = get_wiki_session()
        
        # Try direct lookup first
        params = {
            "action": "query", "format": "json",
            "prop": "extracts", "explaintext": 1,
            "redirects": 1, "titles": title, "formatversion": 2
        }
        
        r = sess.get(WIKI_API, params=params, timeout=10)
        if not r.ok:
            return ""
        
        pages = r.json().get("query", {}).get("pages", [])
        if pages and pages[0].get("extract"):
            text = pages[0]["extract"][:max_length]
            return unicodedata.normalize('NFKC', text)
        
        # If direct lookup fails, try search
        search_params = {
            "action": "query", "format": "json",
            "list": "search", "srsearch": title,
            "srlimit": 1, "formatversion": 2
        }
        
        r = sess.get(WIKI_API, params=search_params, timeout=10)
        if not r.ok:
            return ""
        
        hits = r.json().get("query", {}).get("search", [])
        if hits:
            # Get the first search result
            best_title = hits[0]["title"]
            params["titles"] = best_title
            r = sess.get(WIKI_API, params=params, timeout=10)
            if r.ok:
                pages = r.json().get("query", {}).get("pages", [])
                if pages and pages[0].get("extract"):
                    text = pages[0]["extract"][:max_length]
                    return unicodedata.normalize('NFKC', text)
    
    except (requests.RequestException, ValueError, KeyError):
        pass
    
    return ""

def get_youtube_context(entity: str) -> dict:
    """Get YouTube video context for entity"""
    if not USE_YOUTUBE:
        return {}
    
    client = get_yt_client()
    if not client:
        return {}
    
    try:
        # Search for music-related videos
        search_queries = [
            f"{entity} song",
            f"{entity} music",
            f"{entity} official"
        ]
        
        for query in search_queries:
            try:
                resp = client.search().list(
                    q=query,
                    part="snippet",
                    type="video",
                    maxResults=1,
                    videoCategoryId="10"  # Music category
                ).execute()
                
                items = resp.get("items", [])
                if items:
                    video = items[0]["snippet"]
                    return {
                        'title': video.get("title", ""),
                        'description': video.get("description", ""),
                        'channel': video.get("channelTitle", "")
                    }
            except HttpError:
                continue
                
    except Exception:
        pass
    
    return {}

def combine_summaries(song_info: dict, summaries: dict) -> str:
    """Combine multiple summaries with focus on song information"""
    combined_parts = []
    
    # Prioritize song-specific information (most important)
    if summaries.get('song_specific'):
        combined_parts.append(summaries['song_specific'])
    
    # Add enhanced song info if available and different
    elif summaries.get('song_enhanced') and summaries['song_enhanced'] != summaries.get('song_specific', ''):
        combined_parts.append(summaries['song_enhanced'])
    
    # Add original entity info if it provides additional context
    if summaries.get('original'):
        existing_text = " ".join(combined_parts).lower()
        if not existing_text or summaries['original'][:50].lower() not in existing_text:
            # Only add if it provides new information
            combined_parts.append(summaries['original'][:200])
    
    # Combine and ensure we don't exceed length limit
    if combined_parts:
        combined = " | ".join(combined_parts)
        return combined[:SUMMARY_LEN]
    
    # Fallback to original entity if nothing else worked
    return song_info.get('original', '')[:SUMMARY_LEN]

def fetch_enhanced_summary_for_entity(entity: str) -> tuple:
    """
    Multi-stage approach for getting rich song summaries:
    1. Get YouTube context
    2. Extract song name from YouTube title  
    3. Fetch multiple Wikipedia summaries focused on songs
    4. Filter non-music content
    5. Combine intelligently with song priority
    """
    # Stage 1: Get YouTube context
    youtube_data = get_youtube_context(entity)
    
    # Stage 2: Extract song information
    song_info = extract_song_name(entity, youtube_data.get('title', ''))
    
    # Stage 3: Fetch multiple song-focused Wikipedia summaries
    summaries = {}
    
    # Try song-specific Wikipedia lookup
    if 'song' in song_info and song_info['song'] != entity:
        # Try enhanced song query first
        song_query = f"{song_info['song']} song"
        summaries['song_specific'] = fetch_wikipedia_summary(song_query, 250)
        
        # If that fails, try just the extracted song name
        if not summaries['song_specific']:
            summaries['song_enhanced'] = fetch_wikipedia_summary(song_info['song'], 250)
    
    # Always try original entity lookup as fallback
    summaries['original'] = fetch_wikipedia_summary(entity, 200)
    
    # Stage 4: Combine summaries with song focus
    combined_summary = combine_summaries(song_info, summaries)
    
    # Stage 5: Music content filtering
    # Check if the summary is music-related
    is_music = is_music_related(combined_summary, entity)
    
    # If not clearly music-related, try to get better music-specific content
    if not is_music and summaries.get('original'):
        # Try with explicit "song" suffix
        music_query = f"{entity} song"
        music_summary = fetch_wikipedia_summary(music_query, 300)
        if music_summary and is_music_related(music_summary, entity):
            combined_summary = music_summary
            is_music = True
    
    # If still not music-related, return empty (will be filtered out)
    final_summary = combined_summary if is_music else ""
    
    return entity, final_summary

# ── BUILD/LOAD ENHANCED SUMMARIES ───────────────────────────
if SUM_FILE.exists():
    print("[2] Loading cached enhanced summaries…")
    summaries = json.loads(SUM_FILE.read_text())
else:
    summaries = {}
    print(f"[2] Generating enhanced summaries with {WORKERS} workers…")
    
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(fetch_enhanced_summary_for_entity, e)
                   for e in entity_pool]
        
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Enhancing entities"):
            entity, summary = fut.result()
            summaries[entity] = summary
    
    # Save to cache with proper Unicode handling
    SUM_FILE.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding='utf-8')

# Filter entities with meaningful summaries and music content
all_entities_with_summaries = [(e, txt) for e, txt in summaries.items() if txt.strip()]
music_entities = [(e, txt) for e, txt in all_entities_with_summaries if len(txt.strip()) > 20]
non_music_filtered = len(all_entities_with_summaries) - len(music_entities)

keep_ids = [e for e, txt in music_entities]
print(f"[2] Content filtering results:")
print(f"    • Total entities processed: {len(entity_pool)}")
print(f"    • Entities with summaries: {len(all_entities_with_summaries)}")
print(f"    • Non-music content filtered out: {non_music_filtered}")
print(f"    • Music entities kept: {len(keep_ids)}")

# Show some examples of what was filtered out
if non_music_filtered > 0:
    print(f"    • Sample filtered entities: ", end="")
    filtered_examples = [e for e, txt in all_entities_with_summaries if len(txt.strip()) <= 20][:5]
    print(", ".join(filtered_examples))

# ── BUILD/LOAD FAISS INDEX ──────────────────────────────────
def build_enhanced_index(titles):
    """Build FAISS index with enhanced summaries"""
    print("    Encoding enhanced summaries…")
    embeddings = encoder.encode(
        [summaries[t] for t in titles],
        batch_size=32, 
        show_progress_bar=True,
        convert_to_numpy=True, 
        normalize_embeddings=True
    ).astype("float32")
    
    # Use inner product for cosine similarity with normalized embeddings
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    
    # Save to cache
    faiss.write_index(index, str(IDX_FILE))
    MAP_FILE.write_text(json.dumps(titles, ensure_ascii=False), encoding='utf-8')
    
    return index

if IDX_FILE.exists() and MAP_FILE.exists():
    print("[3] Loading FAISS cache…")
    cpu_index = faiss.read_index(str(IDX_FILE))
    keep_ids = json.loads(MAP_FILE.read_text())
else:
    print("[3] Building enhanced FAISS index…")
    cpu_index = build_enhanced_index(keep_ids)

# Create index mappings
id2idx = {entity_id: i for i, entity_id in enumerate(keep_ids)}
idx2id = {i: entity_id for entity_id, i in id2idx.items()}

# GPU setup with fallback
try:
    print("    Setting up GPU acceleration…")
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    emb_mat = gpu_index.reconstruct_n(0, gpu_index.ntotal)
    print("    ✓ GPU acceleration enabled")
except Exception as e:
    print(f"    ⚠ GPU unavailable, using CPU: {e}")
    gpu_index = cpu_index
    emb_mat = np.vstack([cpu_index.reconstruct(i) for i in range(cpu_index.ntotal)])

# ── PER‐SUBREDDIT MAPPING ───────────────────────────────────
print("[3] Building subreddit mappings…")
entity2subs = {entity: set() for entity in keep_ids}
for post in posts:
    sub = post["source_subreddit"]
    for entity in post.get("combined_comment_entities", []):
        if entity in entity2subs:
            entity2subs[entity].add(sub)

# Build reverse mapping: subreddit -> entity indices
idx_by_sub = {}
for idx, entity in idx2id.items():
    for subreddit in entity2subs[entity]:
        idx_by_sub.setdefault(subreddit, set()).add(idx)

# ── ENHANCED RETRIEVAL ──────────────────────────────────────
def retrieve_candidates_enhanced(query: str, gt_list: list, subreddit: str) -> list:
    """
    Enhanced retrieval with ground truth prioritization and smart filtering
    """
    # Encode query
    qv = encoder.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")
    
    # Get initial candidate pool from FAISS
    _, idxs = gpu_index.search(qv[None], POOL_R)
    
    # Separate ground truth indices
    gt_indices = []
    for gt in gt_list:
        idx = id2idx.get(gt)
        if idx is not None:
            gt_indices.append(idx)
    
    # Get semantic candidates (excluding ground truth)
    semantic_pool = [i for i in idxs[0] 
                     if i in idx_by_sub.get(subreddit, set()) 
                     and i not in gt_indices]
    
    # Fallback: if subreddit filtering yields too few results, expand search
    if len(semantic_pool) < TOP_K - len(gt_indices):
        additional_candidates = [i for i in idxs[0] if i not in gt_indices and i not in semantic_pool]
        semantic_pool.extend(additional_candidates)
    
    # Rank semantic candidates by similarity
    sims = emb_mat @ qv
    ranked_semantic = sorted(semantic_pool, key=lambda i: sims[i], reverse=True)
    
    
    # Step 1: Always include ALL ground truth 
    final_candidates = gt_indices.copy()
    
    # Step 2: Fill remaining slots with best semantic candidates (avoid GT duplicates)  
    remaining_slots = TOP_K - len(final_candidates)
    available_semantic = [idx for idx in ranked_semantic if idx not in gt_indices]
    final_candidates.extend(available_semantic[:remaining_slots])
    
    # Step 3: Shuffle ALL positions to remove bias (GT can appear anywhere)
    random.shuffle(final_candidates)
    
    return [idx2id[i] for i in final_candidates[:TOP_K]]

# ── GENERATE CANDIDATES ─────────────────────────────────────
print("[4] Generating enhanced candidates…")
tmp_path = OUTPUT_PATH.with_suffix(".jsonl.tmp")

with tmp_path.open("w", encoding="utf-8") as out:
    for post in tqdm(posts, desc="Processing posts"):
        gt_entities = post.get("combined_comment_entities", [])
        subreddit = post["source_subreddit"]
        
        # Store ground truth and generate candidates
        post["ground_truth"] = gt_entities
        post["candidates"] = retrieve_candidates_enhanced(post["query"], gt_entities, subreddit)
        
        # Write enhanced post
        out.write(json.dumps(post, ensure_ascii=False) + "\n")

# Move temp file to final location
shutil.move(tmp_path, OUTPUT_PATH)

print(f"[✓] Enhanced music candidates written to {OUTPUT_PATH}")
print(f"    ✅ Music entities prioritized: ✓")
print(f"    ✅ Multi-source song summaries: ✓") 
print(f"    🎵 Music content filtering: ✓")
print(f"    🎬 Smart YouTube song parsing: {'✓' if USE_YOUTUBE else '⚠ (API key not set)'}")
print(f"    🚀 GPU acceleration: {'✓' if 'gpu_index' in locals() and gpu_index != cpu_index else '⚠ (using CPU)'}")
