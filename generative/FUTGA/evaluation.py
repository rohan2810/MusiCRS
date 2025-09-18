# Copyright (2023) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import argparse
import json
import pandas as pd
import copy
import numpy as np
from tqdm import tqdm
from model import SALMONN

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ckpt_path", type=str, default='./salomnn_7b.bin')
    parser.add_argument("--whisper_path", type=str, default='whisper-large-v2')
    parser.add_argument("--beats_path", type=str, default='BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt')
    parser.add_argument("--vicuna_path", type=str, default='vicuna-7b-v1.5')
    parser.add_argument("--audio_path", type=str, default='./Harmonixset/music_data')
    parser.add_argument("--caption_path", type=str, default='./Harmonixset/captions')
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=10000)
    parser.add_argument("--low_resource", action='store_true', default=False)
    parser.add_argument("--debug", action="store_true", default=False)

    args = parser.parse_args()

    os.makedirs(args.caption_path, exist_ok=True)

    model = SALMONN(
        ckpt=args.ckpt_path,
        whisper_path=args.whisper_path,
        beats_path=args.beats_path,
        vicuna_path=args.vicuna_path
    ).to(torch.float16).cuda()
    model.eval()

    prompt_tmp = 'First describe the music in general in terms of mood, theme, tempo, melody, instruments and chord progression. Then provide a detailed music analysis by describing each functional segment and its time boundaries.'

    sample_list = os.listdir(args.audio_path)[args.start:args.end]
    with torch.cuda.amp.autocast(dtype=torch.float16):
        for sample in tqdm(sample_list):
            if os.path.exists(f'{args.caption_path}/{sample}.json'):
                continue
            try:
                wav_path = f'{args.audio_path}/{sample}'
                prompt = prompt_tmp
                save_sample = {'wav_path': sample}
                captions = model.generate(
                    wav_path,
                    prompt=prompt,
                    bdr=(0, 180),
                    repetition_penalty=1.5,
                    num_return_sequences=1,
                    num_beams=5,
                    top_p=0.95,
                    top_k=50,
                )
                save_sample['captions'] = captions
                json.dump(save_sample, open(f'{args.caption_path}/{sample}.json', 'w'))
            except Exception as e:
                print(e)
