import random
import copy
import json
import torch
import transformers
import numpy as np
import pickle as pkl
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

IGNORE_INDEX = -100
MAX_LENGTH = 2048


@dataclass
class DataArguments:
    data_path: str = field(default='./MusicCaps', metadata={"help": "Path to the training data."})
    feat_folder: Optional[str] = field(default='./MusicCaps/music_feat')

def preprocess_v1(sources: str, tokenizer: transformers.PreTrainedTokenizer, metadata,
                  prompt_pattern="USER: <Speech><SpeechHere></Speech> Describe the music in detail.\nASSISTANT:\n") -> Dict:
    sources = sources.split('\n')
    clips, duration, caption = metadata['clips'], metadata['duration'], []
    length = 0
    for l, c in zip(clips, sources):
        caption.append(
            f'From {int(length / duration * 100)} to {int((length + l) / duration * 100)},'
            + ','.join(c.split(',')[1:])
        )
        length += l

    targets = prompt_pattern + '\n'.join(caption)

    targets_left, targets_right = targets.split('<SpeechHere>')
    targets_right = tokenizer(targets_right, return_tensors="pt", add_special_tokens=False).input_ids[0]

    sources_left, sources_right = prompt_pattern.split('<SpeechHere>')
    sources_left = tokenizer(sources_left, return_tensors="pt", add_special_tokens=False).input_ids[0]
    sources_right_length = tokenizer(sources_right, return_tensors="pt", add_special_tokens=False).input_ids.shape[-1]

    sources_right = copy.deepcopy(targets_right)

    targets_left = torch.LongTensor([IGNORE_INDEX] * len(sources_left))
    targets_right[:sources_right_length] = IGNORE_INDEX

    sources_right, targets_right = sources_right[:MAX_LENGTH], targets_right[:MAX_LENGTH]

    return dict(input_ids=(sources_left, sources_right), labels=(targets_left, targets_right))


def preprocess(sources: str, tokenizer: transformers.PreTrainedTokenizer, metadata,
               prompt_pattern="USER: <Speech><SpeechHere></Speech> Describe the music in detail.\nASSISTANT:\n") -> Dict:
    targets = prompt_pattern + sources

    targets_left, targets_right = targets.split('<SpeechHere>')
    targets_right = tokenizer(targets_right, return_tensors="pt", add_special_tokens=False).input_ids[0]

    sources_left, sources_right = prompt_pattern.split('<SpeechHere>')
    sources_left = tokenizer(sources_left, return_tensors="pt", add_special_tokens=False).input_ids[0]
    sources_right_length = tokenizer(sources_right, return_tensors="pt", add_special_tokens=False).input_ids.shape[-1]

    sources_right = copy.deepcopy(targets_right)

    targets_left = torch.LongTensor([IGNORE_INDEX] * len(sources_left))
    targets_right[:sources_right_length] = IGNORE_INDEX

    sources_right, targets_right = sources_right[:MAX_LENGTH], targets_right[:MAX_LENGTH]

    return dict(input_ids=(sources_left, sources_right), labels=(targets_left, targets_right))


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path, tokenizer, data_args):
        super(LazySupervisedDataset, self).__init__()

        self.tokenizer = tokenizer
        self.list_data_dict = json.load(open(data_path, "r"))
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        source = copy.deepcopy(self.list_data_dict[i])

        feature_path = '{}/{}.pkl'.format(self.data_args.feat_folder, source['id'])  # Added
        music = pkl.load(open(feature_path, 'rb'))  # <N, 768> float16
        speech = torch.from_numpy(music['speech'])
        audio = torch.from_numpy(music['audio'])

        captions = source['caption']
        if not isinstance(captions, str):
            weights = np.asarray([len(c) for c in captions])
            weights = weights / weights.sum()
            captions = random.choices(captions, weights, k=1)[0]

        data_dict = preprocess(captions, self.tokenizer, source['meta'])
        
        data_dict['speeches'] = speech
        data_dict['audios'] = audio
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids, labels, speeches, audios = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels", "speeches", "audios"))
        batch = dict(input_ids=input_ids, labels=labels, speeches=speeches, audios=audios)
        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
