"""
Music Recommendation System using Qwen2.5-Omni for Multimodal Ranking


=== RANKING METHODS ===

**Generative Approach:**
- Presents all candidates to the model simultaneously
- Asks the model to generate a comma-separated ranked list
- Uses order randomization and bullet points to mitigate order bias
- Employs improved prompting strategies for better instruction following
- Post-processes output to ensure all candidates are included

=== MULTIMODAL INPUT MODES ===
- **query_only**: Text query from Reddit post only
- **audio_query**: Both audio clips + text query (multimodal)
- **audio_only**: Audio clips only, no text context

=== EVALUATION FRAMEWORK ===
The system tracks comprehensive metrics with statistical rigor:

**Core Ranking Metrics:**
- Recall@K (K=1,5,10,20): Fraction of relevant items found in top-K
- Precision@K: Fraction of top-K that are relevant  
- NDCG@K: Normalized Discounted Cumulative Gain (position-aware quality)
- MRR: Mean Reciprocal Rank of first relevant item
- Hit Rate: Percentage of queries with ≥1 relevant item in top-10
- Avg First Relevant Position: Average rank of first correct recommendation

=== AUDIO PROCESSING ===
- Smart audio budget allocation (5min total per query)
- Random temporal sampling from available audio clips
- Proper sampling rate handling for Qwen2.5-Omni
- Robust error handling for corrupted/missing audio files

"""

import os
import json
import librosa
import soundfile as sf
import torch
import tqdm
import numpy as np
from transformers import Qwen2_5OmniForConditionalGeneration,Qwen2_5OmniProcessor
from pathlib import Path
import random
import torch.nn.functional as F
import wandb
import time
import math
from datetime import datetime
from collections import defaultdict

# Import our utilities
from utils import (
    normalize_string, normalize_list_of_lists,
    calculate_comprehensive_metrics, calculate_subreddit_metrics,
    load_and_mix_audio, build_ranking_prompt, parse_and_filter_ranking,
    print_metrics_summary, print_subreddit_comparison
)


wandb.login(key='d13ff15cc78423826ede4beef6d5e6f91a7cc50e')

MODEL_ID = "Qwen/Qwen2.5-Omni-7B"   
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
AUDIO_BASE_DIR  = Path("/deepfreeze/junda/abhay/audio_yt/full/wav")

processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID)
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(MODEL_ID, 
                                                           device_map="auto",
                                                           torch_dtype="auto")
# Disable audio output to save memory since we only need text ranking
model.disable_talker()
model.eval()


# ===== UTILITY FUNCTIONS FOR QWEN2.5-OMNI =====
def process_mm_info(conversations, use_audio_in_video=False):
    """Process multimedia information from conversations for Qwen2.5-Omni"""
    audios = []
    images = []
    videos = []
    
    for conversation in conversations:
        for message in conversation:
            if "content" in message and isinstance(message["content"], list):
                for content_item in message["content"]:
                    if isinstance(content_item, dict):
                        if content_item.get("type") == "audio":
                            audios.append(content_item["audio"])
                        elif content_item.get("type") == "image":
                            images.append(content_item["image"])
                        elif content_item.get("type") == "video":
                            videos.append(content_item["video"])
    
    return audios, images, videos

def rank_songs_generative(
    audio_paths: list[str],
    convo: str,
    candidates: list[str],
    mode: str,
    max_clips: int = 10,
    audio_budget_secs: float = 300.0,  # 5 minutes total budget
) -> list[str]:
    
    sr = processor.feature_extractor.sampling_rate
    clips: list[np.ndarray] = []
    
    if mode in ("audio_query", "audio_only"):
        # Smart audio budget allocation
        num_available = len(audio_paths)
        num_clips_to_use = min(num_available, max_clips)
        clip_duration = audio_budget_secs / num_clips_to_use if num_clips_to_use > 0 else 30.0
        sel = random.sample(audio_paths, k=num_clips_to_use)
        for p in sel:
            try:
                info = sf.info(p)
                total_secs = info.frames / info.samplerate
                start_sec = random.uniform(0, max(0, total_secs - clip_duration))
                wav, _ = librosa.load(p, sr=sr, offset=start_sec, duration=clip_duration)
                clips.append(wav)
            except Exception as e:
                print(f"Error loading audio {p}: {e}")
                continue
        
        if mode in ("audio_query", "audio_only") and len(clips) == 0:
            print("No audio clips loaded - this may affect audio processing quality")
        

    randomized_candidates = candidates.copy()
    random.shuffle(randomized_candidates)

    sys_msg = {"role": "system", "content": [{"type": "text", "text": "You are a helpful music recommendation expert. Rank songs based on relevance to the audio and query."}]}
    
    user_content: list[dict] = []
    if mode in ("audio_query", "audio_only"):
        user_content += [{"type": "audio", "audio": c} for c in clips]
    if mode in ("audio_query", "query_only"):
        user_content.append({"type": "text", "text": convo})
    
    user_q = {"role": "user", "content": user_content}

    # Use randomized candidates and avoid numbered lists to reduce order bias
    header = f"Here are {len(randomized_candidates)} candidate songs; rank by relevance:"
    lines = [header] + [f"- {c}" for c in randomized_candidates]

    cand_blob = "\n".join(lines)

    instr = (
        f"Rank these {len(randomized_candidates)} songs from most to least relevant based on the audio and query. "
        "Output ONLY the exact song titles separated by commas. "
        "Example format: Song Title A, Song Title B, Song Title C"
    )

    cand_msg = {"role": "user", "content": [{"type": "text", "text": cand_blob}]}
    instr_msg = {"role": "user", "content": [{"type": "text", "text": instr}]}

    messages = [sys_msg, user_q, cand_msg, instr_msg]

    # Process conversation with multimedia for Qwen2.5-Omni
    conversations = [messages]
    conversation_audios, images, videos = process_mm_info(conversations, use_audio_in_video=False)
    
    try:
        prompt = processor.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=False
        )
    except Exception as e:
        print(f"Error applying chat template: {e}")
        return []

    try:
        model_inputs = processor(
            text=prompt,
            audio=conversation_audios if conversation_audios else None,
            images=None,
            videos=None,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False
        )
    except Exception as e:
        print(f"Error processing model inputs: {e}")
        return []
    
    # Move to device
    for k, v in model_inputs.items():
        model_inputs[k] = v.to(model.device)

    with torch.no_grad():
        out_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=False,
            use_audio_in_video=False,
            return_audio=False
        )
    gen_ids = out_ids[:, model_inputs["input_ids"].size(1):]

    text = processor.batch_decode(
        gen_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )[0].strip()
    
    # Use utility function to parse and filter the ranking
    return parse_and_filter_ranking(text, randomized_candidates, fill_missing=True)


def main():
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    
    input_jsonl = "/home/junda/rohan/reddit-music/data/merged_final_cleaned_music_clean_queries_with_candidates_enhanced_v2_100_candidates.jsonl"
    
    modes = ["audio_query", "audio_only", "query_only"]  # Options: ["query_only", "audio_query", "audio_only"]
    num_samples = -1
    
    all_method_results = {}

    for mode in modes:
                
        wandb.init(
            project="reddit-music-qwen-omni-final",
            entity="musiCRS",
            name=f"{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "model_id": MODEL_ID,
                "mode": mode,
                "num_samples": num_samples,
                "random_seed": 42,
            },
            tags=["qwen2.5-omni-final", "music-ranking", mode]
        )
        
        results = {
            "all_predictions": [],
            "all_ground_truth": [],
            "all_subreddits": []
        }
        
        print(f"Processing {mode} mode")
        output_jsonl = f"/home/junda/rohan/reddit-music/generative/results/QWEN_OMNI_FINAL/{mode}.jsonl"
        output_dir = os.path.dirname(output_jsonl)
        os.makedirs(output_dir, exist_ok=True)

        with open(input_jsonl, "r", encoding="utf-8") as rf, \
             open(output_jsonl, "w", encoding="utf-8") as wf:

            if num_samples == -1:
                lines = rf.readlines()
            else:
                lines = rf.readlines()[:num_samples]
            
            for i, line in enumerate(tqdm.tqdm(lines, desc=f"Processing ({mode})")):
                try:
                    ex = json.loads(line)
                    sid = ex["name"]
                    convo = ex["query"]
                    candidates = ex["final_candidates"]
                    ground_truth = ex["limited_gt"]
                    audio_dir = AUDIO_BASE_DIR / sid
                    wav_paths = sorted(audio_dir.glob("*.wav"))
                    
                    if not wav_paths or not candidates:
                        continue

                    ranked = rank_songs_generative(
                        audio_paths=wav_paths,
                        convo=convo,
                        candidates=candidates,
                        mode=mode,
                    )

                    record = ex.copy()
                    record["predicted"] = ranked
                    wf.write(json.dumps(record, ensure_ascii=False) + "\n")
                    
                    # Collect for final evaluation
                    results["all_predictions"].append(ranked)
                    results["all_ground_truth"].append(ground_truth)
                    results["all_subreddits"].append(ex.get("source_subreddit", "unknown"))
                    
                except Exception as e:
                    print(f"Error processing sample {i}: {e}")
                    continue

        # Calculate FINAL evaluation metrics
        final_eval_metrics = calculate_comprehensive_metrics(
            results["all_predictions"],
            results["all_ground_truth"]
        )
        
        # Calculate per-subreddit metrics
        subreddit_metrics = calculate_subreddit_metrics(
            results["all_predictions"],
            results["all_ground_truth"], 
            results["all_subreddits"]
        )
        
        final_log_dict = {}
        for metric_name, metric_value in final_eval_metrics.items():
            final_log_dict[f"FINAL_{metric_name}"] = metric_value
        
        for subreddit, sub_metrics in subreddit_metrics.items():
            for metric_name, metric_value in sub_metrics.items():
                final_log_dict[f"SUBREDDIT_{subreddit}_{metric_name}"] = metric_value
        
        wandb.log(final_log_dict)

        print(f"Wrote {mode} results to {output_jsonl}")
        
        # Print FINAL evaluation results
        if final_eval_metrics:
            print_metrics_summary(final_eval_metrics, f"FINAL Evaluation Results for {mode}")
            
            # Print per-subreddit results
            if subreddit_metrics:
                print_subreddit_comparison(subreddit_metrics)
        
        # Store results for potential cross-run comparison
        all_method_results[mode] = final_eval_metrics
        
        # Finish this wandb run
        wandb.finish()

    # Print final comparison summary
    print(f"\n🎉 All experiments completed!")
    print("="*60)
    print("FINAL COMPARISON SUMMARY:")
    print("="*60)
    
    for mode, final_metrics in all_method_results.items():
        print(f"{mode}:")
        print(f"  Samples: {final_metrics.get('num_samples', 0)}")
        print(f"  R@10={final_metrics.get('recall_at_10', 0):.4f} "
              f"P@10={final_metrics.get('precision_at_10', 0):.4f} "
              f"nDCG@10={final_metrics.get('ndcg_at_10', 0):.4f} "
              f"MRR={final_metrics.get('mrr', 0):.4f}")
        print()
    
    print(f"📊 View individual runs in wandb project: reddit-music-qwen-omni-final")


if __name__ == "__main__":
    main()