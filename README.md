# MusiCRS: Multimodal Music Conversational Recommendation System

A concise research codebase for ranking song candidates for Reddit-style conversational posts using multimodal LLMs (Qwen2-Audio, Qwen2.5-Omni, SALMONN variants, FUTGA) and retrieval baselines (CLAP, CoLLAP).

![Figure 2](img/fig2_page1.png)
[Full PDF](img/fig2.pdf)

![Figure 3](img/fig3_up.png)

---

## Dataset

- MusiCRS dataset on Hugging Face: [MusiCRS (rohan2810/MusiCRS)](https://huggingface.co/datasets/rohan2810/MusiCRS)

---

## Quickstart

```bash
# 1) Create env
conda create -n musicrs python=3.10 -y && conda activate musicrs

# 2) Install dependencies (choose what you need)
pip install -r requirements_qwen_audio.txt
pip install -r requirements_qwen_omni.txt
pip install -r generative/SALMONN/requirements.txt
pip install -r generative/SALMONN-7B/requirements.txt

# Optional utilities
pip install sentence-transformers faiss-cpu google-api-python-client streamlit duckdb zstandard

# 3) (Optional) Set credentials
export WANDB_API_KEY="..."      # if using Weights & Biases logging
export YOUTUBE_API_KEY="..."    # if enabling YouTube enhancement in pipeline
```

Defaults are repo-relative (env-overridable). Examples:
- `AUDIO_BASE_DIR=audio/full/wav`
- `INPUT_JSONL=data/merged_final_cleaned_music_clean_queries_with_candidates_enhanced_v2_100_candidates.jsonl`

Override via environment, e.g.:
```bash
export INPUT_JSONL=./data/your_file.jsonl
export AUDIO_BASE_DIR=./audio/full/wav
```

---

## Run Examples

- CLAP retrieval
```bash
python retrieval/clap.py
```

- Qwen2-Audio generative ranking
```bash
python generative/qwen.py
```

- Qwen2.5-Omni generative ranking
```bash
python generative/qwen-omni.py
```

- SALMONN (config-driven)
```bash
python generative/SALMONN/cli_inference.py \
  --cfg-path generative/SALMONN/configs/decode_config.yaml
```

- SALMONN-7B (config-driven)
```bash
python generative/SALMONN-7B/cli_inference.py \
  --cfg-path generative/SALMONN-7B/configs/decode_config.yaml
```

---

## Authors

Rohan Surana*, Amit Namburi*, Gagan Mundada*, Abhay Lal*, Zachary Novack, Julian McAuley, Junda Wu

- *Equal contribution.

---

## Citation

If you find MusiCRS useful, please cite:

Surana, R., Namburi, A., Mundada, G., Lal, A., Novack, Z., McAuley, J., Wu, J. MusiCRS: Multimodal Music Conversational Recommendation System. 2025.

```bibtex
@misc{MusiCRS2025,
  title        = {MusiCRS: Multimodal Music Conversational Recommendation System},
  author       = {Surana, Rohan and Namburi, Amit and Mundada, Gagan and Lal, Abhay and Novack, Zachary and McAuley, Julian and Wu, Junda},
  year         = {2025},
  howpublished = {\url{https://huggingface.co/datasets/rohan2810/MusiCRS}},
  note         = {Equal contribution for authors marked *}
}
```
