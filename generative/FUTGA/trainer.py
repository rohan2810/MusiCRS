# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
from dataclasses import dataclass, field
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import torch
import transformers
import sys

from salmonn_trainer import SALMONNTrainer, get_state
from dataset import make_supervised_data_module, DataArguments
from model import SALMONN
from utils import print_trainable_parameters

import wandb


@dataclass
class ModelArguments:
    ckpt_path: Optional[str] = field(default='./salmonn_7b_v0.pth')
    whisper_path: Optional[str] = field(default='./whisper-large-v2')
    beats_path: Optional[str] = field(default='./BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt')
    vicuna_path: Optional[str] = field(default='./vicuna-7b-v1.5')
    version: Optional[str] = field(default="v0")
    device: Optional[str] = field(default='cuda')


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: Optional[str] = field(default='./checkpoints/')
    optim: str = field(default="adamw_torch")
    bf16: bool = True
    fp16: bool = False
    lora_alpha: int = 32
    model_max_length: int = 2048
    use_cache: bool = False
    gradient_checkpointing: bool = False


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    wandb.init(project='SALMONN', name=training_args.run_name)

    model = SALMONN(
        model_args.ckpt_path, model_args.whisper_path, model_args.beats_path, model_args.vicuna_path,
        lora_alpha=training_args.lora_alpha, compute_dtype=compute_dtype
    ).cuda()
    print_trainable_parameters(model, vb=0)

    data_module = make_supervised_data_module(tokenizer=model.tokenizer, data_args=data_args)
    trainer = SALMONNTrainer(model=model, tokenizer=model.tokenizer, args=training_args, **data_module)

    trainer.train()

    # Only save Adapter
    weight_to_save = get_state(model.named_parameters())
    torch.save(weight_to_save, os.path.join(training_args.output_dir, f'salomnn_7b.bin'))

    trainer.save_state()


if __name__ == "__main__":
    train()
