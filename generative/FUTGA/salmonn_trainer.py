import os
import torch

from transformers import Trainer
from typing import Optional

LOG_INTERVAL = 5000

def get_state(model):
    trainable_state_dict = dict()
    for name, param in model.state_dict().items():
        try:
            if model.get_parameter(name).requires_grad:
                trainable_state_dict[name] = param
        except:
            trainable_state_dict[name] = param
    return trainable_state_dict


class SALMONNTrainer(Trainer):

    def _save_checkpoint(self, model, trial, metrics=None):
        from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{(self.state.global_step // LOG_INTERVAL) * LOG_INTERVAL}"

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        os.makedirs(output_dir, exist_ok=True)

        # Only save Adapter
        weight_to_save = get_state(self.model)
        torch.save(weight_to_save, os.path.join(output_dir, f'salomnn_7b.bin'))

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        super(SALMONNTrainer, self)._save(output_dir, state_dict)
