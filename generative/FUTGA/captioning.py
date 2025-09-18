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
    parser.add_argument("--meta_path", type=str, default='./Harmonixset/metadata.csv')
    parser.add_argument("--segment_path", type=str, default='./Harmonixset/segments')
    parser.add_argument("--caption_path", type=str, default='./Harmonixset/captions')
    parser.add_argument("--low_resource", action='store_true', default=False)
    parser.add_argument("--debug", action="store_true", default=False)

    args = parser.parse_args()

    os.makedirs(args.caption_path, exist_ok=True)
    meta = pd.read_csv(args.meta_path, header=0)[['File', 'BPM', 'Genre']]
    samples = []
    for i, row in meta.iterrows():
        fname = row['File']
        sample = row.to_dict()
        sample['audio'] = f'{args.audio_path}/{fname}.wav'
        sample['segment'] = f'{args.segment_path}/{fname}.txt'
        if os.path.exists(sample['audio']) and os.path.exists(sample['segment']):
            samples.append(sample)


    model = SALMONN(
        ckpt=args.ckpt_path,
        whisper_path=args.whisper_path,
        beats_path=args.beats_path,
        vicuna_path=args.vicuna_path
    ).to(torch.float16).cuda()
    model.eval()

    # prompt_tmp = 'Please describe functional music segments and their time boundaries. In each segment, describe the musical change of each segment and provide detailed analysis.'
    # prompt_tmp = 'First describe the music in general in terms of mood, theme, tempo, melody, instruments and chord progression. Then provide a detailed music analysis by describing each functional segment and its time boundaries.'
    prompt_tmp = 'This is a {genre} music of {bpm} beat-per-minute (BPM). First describe the music in general in terms of mood, theme, tempo, melody, instruments and chord progression. Then provide a detailed music analysis by describing each functional segment and its time boundaries. Please note that the music boundaries are {segments}.'

    with torch.cuda.amp.autocast(dtype=torch.float16):
        for sample in tqdm(samples):
            fname = sample['File']
            if os.path.exists(f'{args.caption_path}/{fname}.json'):
                continue
            # try:
            wav_path = sample['audio']
            ts, tag = zip(*[line.split(' ') for line in open(sample['segment']) if 'silence' not in line and line.strip()])
            ts = np.asarray([float(t) for t in ts])
            bdr = (ts[0], ts[-1])
            ts = (ts - ts[0]) / (ts[-1] - ts[0])
            ts = [np.round(t * 100) for t in ts]

            prompt = prompt_tmp.format(genre=sample['Genre'], bpm=sample['BPM'], segments=ts)

            save_sample = copy.deepcopy(sample)
            captions = model.generate(wav_path, prompt=prompt, bdr=bdr, repetition_penalty=1.5, num_return_sequences=5, num_beams=10)
            save_sample['tags'] = tag
            save_sample['ts'] = ts
            save_sample['captions'] = captions
            json.dump(save_sample, open(f'{args.caption_path}/{fname}.json', 'w'))
            # except Exception as e:
            #     print(e)
