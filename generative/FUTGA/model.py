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


import torch
import soundfile as sf
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    WhisperFeatureExtractor,
    WhisperModel,
    # LlamaForCausalLM,
    LlamaTokenizer
)
from modeling_llama import LlamaForCausalLM
import librosa
from beats.BEATs import BEATsConfig, BEATs
from qformer.Qformer import BertConfig, BertLMHeadModel
from typing import List, Optional, Tuple, Union

IGNORE_INDEX = -100


class SALMONN(nn.Module):
    def __init__(self, ckpt, whisper_path, beats_path, vicuna_path,
                 speech_qformer_token_num=1, speech_qformer_layer=2,
                 lora=True, lora_alpha=32, lora_rank=8, lora_dropout=0.1,
                 second_per_frame=0.333333, second_stride=0.333333, compute_dtype=torch.float16):

        super().__init__()

        # feature_extractor
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_path)

        # whisper
        self.speech_encoder = WhisperModel.from_pretrained(whisper_path).encoder
        for name, param in self.speech_encoder.named_parameters():
            param.requires_grad = False
        self.ln_speech = nn.LayerNorm(self.speech_encoder.config.d_model)
        print('Whisper model loaded ........')

        # beats
        self.beats_ckpt = beats_path
        beats_checkpoint = torch.load(self.beats_ckpt, map_location='cpu')
        beats = BEATs(BEATsConfig(beats_checkpoint['cfg']))
        beats.load_state_dict(beats_checkpoint['model'])
        self.beats = beats
        for name, param in self.beats.named_parameters():
            param.requires_grad = False
        self.beats.eval()
        self.ln_audio = nn.LayerNorm(self.beats.cfg.encoder_embed_dim)
        print('Beats model loaded ........')

        # init speech Qformer
        self.speech_Qformer, self.speech_query_tokens = self.init_speech_Qformer(
            speech_qformer_token_num,
            self.speech_encoder.config.d_model + self.beats.cfg.encoder_embed_dim,
            speech_qformer_layer,
        )
        self.second_per_frame = second_per_frame
        self.second_stride = second_stride
        print('Qformer model initialised ........')

        # vicuna
        self.llama_model = LlamaForCausalLM.from_pretrained(vicuna_path, torch_dtype=compute_dtype)
        self.config = self.llama_model.config
        print('Vicuna model loaded ........')

        # lora
        self.lora = lora
        if lora:
            target_modules = None
            self.peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, inference_mode=False, target_modules=target_modules,
                r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            )
            self.llama_model = get_peft_model(self.llama_model, self.peft_config)
            print('Added LoRA ........')

        # tokenizer
        self.tokenizer = LlamaTokenizer.from_pretrained(vicuna_path, use_fast=False)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.tokenizer.padding_side = "right"

        # proj
        self.speech_llama_proj = nn.Linear(self.speech_Qformer.config.hidden_size, self.llama_model.config.hidden_size)

        # load ckpt
        print('Loading Parameters ........')
        ckpt_dict = torch.load(ckpt, map_location='cpu')
        if 'model' in ckpt_dict:
            ckpt_dict = ckpt_dict['model']
        for name, param in ckpt_dict.items():
            if name in self.state_dict():
                print('Loaded:', name)
        self.load_state_dict(ckpt_dict, strict=False)

    def forward(
            self,
            input_ids, labels, speeches, audios,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        speech_embeds, sources, targets = [], [], []
        for speech_embed, audio_embed, input_id, label in zip(speeches, audios, input_ids, labels):
            speech_embed, audio_embed = speech_embed.to('cuda'), audio_embed.to('cuda')
            # auditory embeds
            speech_embed = self.ln_speech(speech_embed)
            audio_embed = self.ln_audio(audio_embed)
            audio_embed = F.pad(audio_embed, (0, 0, 0, speech_embed.size(1) - audio_embed.size(1)))
            speech_embed = torch.cat([speech_embed, audio_embed], dim=-1)

            # split frames
            B, T, C = speech_embed.shape
            kernel, stride = round(T * self.second_per_frame / 30.0), round(T * self.second_stride / 30.0)
            kernel, stride = (1, kernel), (1, stride)
            speech_embeds_tr = speech_embed.transpose(1, 2).unsqueeze(2)
            speech_embeds_overlap = F.unfold(speech_embeds_tr, kernel_size=kernel, dilation=1, padding=0, stride=stride)
            _, _, L = speech_embeds_overlap.shape
            speech_embeds_overlap = speech_embeds_overlap.view(B, -1, kernel[1], L)
            speech_embeds_overlap = torch.permute(speech_embeds_overlap, [0, 3, 2, 1])
            speech_embed = speech_embeds_overlap.reshape(-1, kernel[1], C)
            speech_atts = torch.ones(speech_embed.size()[:-1], dtype=torch.long, device=speech_embed.device)

            # Qformer
            query_tokens = self.speech_query_tokens.expand(speech_embed.shape[0], -1, -1)
            query_output = self.speech_Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=speech_embed,
                encoder_attention_mask=speech_atts,
                return_dict=True,
                use_cache=False
            )
            speech_embed = self.speech_llama_proj(query_output.last_hidden_state)
            speech_embed = speech_embed.view(B, -1, speech_embed.size(2)).contiguous()

            sources.append(
                torch.concat([
                    torch.LongTensor([self.tokenizer.bos_token_id]).to(input_id[0].device),
                    input_id[0],
                    torch.LongTensor([self.tokenizer.bos_token_id] * speech_embed.shape[1]).to(input_id[0].device),
                    input_id[1],
                    torch.LongTensor([self.tokenizer.eos_token_id]).to(input_id[0].device),
                ])
            )
            targets.append(
                torch.concat([
                    torch.LongTensor([IGNORE_INDEX]).to(label[0].device),
                    label[0],
                    torch.LongTensor([IGNORE_INDEX] * speech_embed.shape[1]).to(label[0].device),
                    label[1],
                    torch.LongTensor([self.tokenizer.eos_token_id]).to(label[0].device),
                ])
            )
            speech_embeds.append(speech_embed)

        start_length = len(input_ids[0][0]) + 1

        # USER: <Speech>speech_embeds<Speech> prompt\nASSISTANT:
        PADDING_TOKEN = 0
        embed_tokens = self.llama_model.model.model.embed_tokens if self.lora else self.llama_model.model.embed_tokens

        input_ids = torch.nn.utils.rnn.pad_sequence(sources, batch_first=True, padding_value=PADDING_TOKEN)
        labels = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=PADDING_TOKEN)
        attention_mask = input_ids.ne(PADDING_TOKEN)

        inputs_embeds = []
        for input_id, speech_embed in zip(input_ids, speech_embeds):
            left_embeds = embed_tokens(input_id[:start_length])
            right_embeds = embed_tokens(input_id[start_length + speech_embed.shape[1]:])
            concat_tensor = torch.concat([left_embeds, speech_embed[0], right_embeds], dim=0).contiguous()
            inputs_embeds.append(concat_tensor)

        inputs_embeds = torch.stack(inputs_embeds)

        return self.llama_model.forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=False,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    def generate(
            self,
            wav_path, prompt, prompt_pattern="USER: <Speech><SpeechHere></Speech> {}\nASSISTANT:", device='cuda:0',
            max_length=2048, num_beams=4, do_sample=True, min_length=1, top_p=0.9, top_k=50,
            repetition_penalty=1.0, length_penalty=1.0, temperature=1.0, bdr=(0, 240), num_return_sequences=1
    ):
        # read wav
        wav, sr = sf.read(wav_path)
        if len(wav.shape) == 2:
            wav = wav[:, 0]
        wav = wav[int(bdr[0] * sr): int(bdr[1] * sr)]
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000, res_type="fft")

        # whisper
        spectrogram = self.feature_extractor(wav, return_tensors="pt", sampling_rate=16000).input_features.to(
            device)  # [1, 80, 3000]
        speech_embeds = self.speech_encoder(spectrogram, return_dict=True).last_hidden_state

        # beats
        raw_wav = torch.from_numpy(wav).to(device).unsqueeze(0)
        audio_padding_mask = torch.zeros(raw_wav.shape, device=device).bool()
        audio_embeds, _ = self.beats.extract_features(raw_wav, padding_mask=audio_padding_mask, feature_only=True)

        # auditory embeds
        speech_embeds = self.ln_speech(speech_embeds)
        audio_embeds = self.ln_audio(audio_embeds)
        audio_embeds = F.pad(audio_embeds, (0, 0, 0, speech_embeds.size(1) - audio_embeds.size(1)))
        speech_embeds = torch.cat([speech_embeds, audio_embeds], dim=-1)

        # split frames
        B, T, C = speech_embeds.shape
        kernel, stride = round(T * self.second_per_frame / 30.0), round(T * self.second_stride / 30.0)
        kernel, stride = (1, kernel), (1, stride)
        speech_embeds_tr = speech_embeds.transpose(1, 2).unsqueeze(2)
        speech_embeds_overlap = F.unfold(speech_embeds_tr, kernel_size=kernel, dilation=1, padding=0, stride=stride)
        _, _, L = speech_embeds_overlap.shape
        speech_embeds_overlap = speech_embeds_overlap.view(B, -1, kernel[1], L)
        speech_embeds_overlap = torch.permute(speech_embeds_overlap, [0, 3, 2, 1])
        speech_embeds = speech_embeds_overlap.reshape(-1, kernel[1], C)
        speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long, device=speech_embeds.device)

        # Qformer
        query_tokens = self.speech_query_tokens.expand(speech_embeds.shape[0], -1, -1)
        query_output = self.speech_Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=speech_embeds,
            encoder_attention_mask=speech_atts,
            return_dict=True,
        )
        speech_embeds = self.speech_llama_proj(query_output.last_hidden_state)
        speech_embeds = speech_embeds.view(B, -1, speech_embeds.size(2)).contiguous()

        # USER: <Speech>speech_embeds<Speech> prompt\nASSISTANT:
        embed_tokens = self.llama_model.model.model.embed_tokens if self.lora else self.llama_model.model.embed_tokens
        prompt_left, prompts_right = prompt_pattern.format(prompt).split('<SpeechHere>')

        prompt_left_ids = self.tokenizer(prompt_left, return_tensors="pt", add_special_tokens=False).to(
            speech_embeds.device).input_ids
        prompt_left_embeds = embed_tokens(prompt_left_ids)

        prompt_right_ids = self.tokenizer(prompts_right, return_tensors="pt", add_special_tokens=False).to(
            speech_embeds.device).input_ids
        prompt_right_embeds = embed_tokens(prompt_right_ids)

        bos_embeds = self.llama_model.model.embed_tokens(
            torch.ones([1, 1], dtype=torch.long, device=device) * self.tokenizer.bos_token_id
        ) if not self.lora else self.llama_model.model.model.embed_tokens(
            torch.ones([1, 1], dtype=torch.long, device=device) * self.tokenizer.bos_token_id
        )

        embeds = torch.cat([bos_embeds, prompt_left_embeds, speech_embeds, prompt_right_embeds], dim=1)
        atts = torch.ones(embeds.size()[:-1], dtype=torch.long).to(embeds.device)

        # generate
        output = self.llama_model.generate(
            inputs_embeds=embeds,
            max_length=max_length,
            num_beams=num_beams,
            do_sample=do_sample,
            min_length=min_length,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            attention_mask=atts,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            # use_cache=False
        )

        output_text = self.tokenizer.batch_decode(output, add_special_tokens=False, skip_special_tokens=True)

        # output_text = self.tokenizer.batch_decode(output)
        return output_text

    def init_speech_Qformer(self, num_query_token, speech_width, num_hidden_layers=2):
        encoder_config = BertConfig()
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = speech_width
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens
