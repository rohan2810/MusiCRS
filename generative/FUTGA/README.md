---
license: apache-2.0
datasets:
- JoshuaW1997/FUTGA
language:
- en
metrics:
- bertscore
- bleu
- rouge
base_model:
- lmsys/vicuna-7b-v1.5
tags:
- music
---

# FUTGA: Fine-grained Music Understanding through Temporally-enhanced Generative Augmentation

<div align=center><img src="futga.jpg" height="256px" width="256px"/></div>

## News

- [10/02] We released [**FUTGA-Dataset**](https://huggingface.co/datasets/JoshuaW1997/FUTGA) including MusicCaps, SongDescriber, HarmonixSet, and AudioSet (train and eval).
- [07/28] We released [**FUTGA-7B**](https://huggingface.co/JoshuaW1997/FUTGA) and **training/inference code** based on [SALMONN-7B](https://huggingface.co/tsinghua-ee/SALMONN) backbone!

## Overview
FUTGA is an audio LLM with fine-grained music understanding, learning from generative augmentation with temporal compositions. By leveraging existing music caption datasets and large language models (LLMs), we synthesize detailed music captions with structural descriptions and time boundaries for full-length songs. This synthetic dataset enables FUTGA to identify temporal changes at key transition points, their musical functions, and generate dense captions for full-length songs.

<div align=center><img src="fig1.png" height="512px" width="512px"/></div>


## Comparing FUTGA dense captioning with MusicCaps/SongDescriber/LP-MusicCaps
![image](demo.png)


## How to load the model
We build **FUTGA-7B** based on SALMONN. Follow the instructions from [SALMONN](https://huggingface.co/tsinghua-ee/SALMONN) to load:
1. [whisper large v2](https://huggingface.co/openai/whisper-large-v2/tree/main) to ```whisper_path```, 
2. [Fine-tuned BEATs_iter3+ (AS2M) (cpt2)](https://valle.blob.core.windows.net/share/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D) to `beats_path`
3. [vicuna 7B v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5/tree/main) to ```vicuna_path```,
4. [FUTGA-7b](https://huggingface.co/JoshuaW1997/FUTGA/blob/main/salomnn_7b.bin) to ```ckpt_path```.


## Datasets
Please visit our dataset repo [FUTGA-Dataset](https://huggingface.co/datasets/JoshuaW1997/FUTGA). We currently include MusicCaps, SongDescriber, HarmonixSet, and AudioSet (train and eval).

## Citation
If you use our models or datasets in your research, please cite it as follows:
```bibtex
@article{wu2024futga,
  title={Futga: Towards Fine-grained Music Understanding through Temporally-enhanced Generative Augmentation},
  author={Wu, Junda and Novack, Zachary and Namburi, Amit and Dai, Jiaheng and Dong, Hao-Wen and Xie, Zhouhang and Chen, Carol and McAuley, Julian},
  journal={arXiv preprint arXiv:2407.20445},
  year={2024}
}