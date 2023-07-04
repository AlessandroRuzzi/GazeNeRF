import logging
import os
import time

import pandas as pd
import torch
from tqdm import tqdm

from utils.logging import log_one_number


def _format_summary(summary):
    """Make a formatted string for logging summary info"""
    return " ".join(f"{k} {v:.4g}" for (k, v) in summary.items())


class BaseTrainer(object):
    """
    Base class for PyTorch trainers.
    This implements the common training logic,
    logging of summaries, and checkpoints.
    """

    def __init__(self, checkpoint_dir=None, batch_size=16, gpu=None, log=False):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.checkpoint_dir = checkpoint_dir
        self.batch_size = batch_size
        self.gpu = gpu
        self.summaries = None
        self.log = log
        if gpu is not None and torch.cuda.is_available():
            self.device = "cuda:%i" % gpu
            torch.cuda.set_device(gpu)
        else:
            self.device = "cpu"

    def _get_summary_file(self):
        return os.path.join("logs/", "summaries_%i.csv" % self.gpu)

    def save_summary(self, summary, write_file=True):
        """Save new summary information"""

        # First summary
        if self.summaries is None:
            self.summaries = pd.DataFrame([summary])

        # Append a new summary row
        else:
            self.summaries = self.summaries.append([summary], ignore_index=True)

        # Write current summaries to file (note: overwrites each time)
        if write_file:
            self.summaries.to_csv(
                self._get_summary_file(), index=False, float_format="%.6f", sep="\t"
            )

    def load_summaries(self):
        self.summaries = pd.read_csv(self._get_summary_file(), delim_whitespace=True)

    def _get_checkpoint_file(self, checkpoint_id):
        return os.path.join("checkpoints", "checkpoint_%03i.pth.tar" % checkpoint_id)

    def save_checkpoint(self, current_epoch, current_loss):
        """Write a checkpoint for the trainer"""
        raise NotImplementedError

    def load_checkpoint(self):
        """Load from checkpoint"""
        raise NotImplementedError

    def state_dict(self):
        """Virtual method to return state dict for checkpointing"""
        raise NotImplementedError

    def load_state_dict(self, state_dict):
        """Virtual method to load a state dict from a checkpoint"""
        raise NotImplementedError

    def build(self, config):
        """Virtual method to build model, optimizer, etc."""
        raise NotImplementedError

    def train_epoch(self, data_loader):
        """Virtual method to train a model"""
        raise NotImplementedError

    @staticmethod
    def enable_gradient(tensor_list):
        for ele in tensor_list:
            ele.requires_grad = True

    def eulurangle2Rmat(self, angles):
        batch_size = self.batch_size

        sinx = torch.sin(angles[:, 0])
        siny = torch.sin(angles[:, 1])
        sinz = torch.sin(angles[:, 2])
        cosx = torch.cos(angles[:, 0])
        cosy = torch.cos(angles[:, 1])
        cosz = torch.cos(angles[:, 2])

        rotXs = (
            torch.eye(3, device=angles.device).view(1, 3, 3).repeat(batch_size, 1, 1)
        )
        rotYs = rotXs.clone()
        rotZs = rotXs.clone()

        rotXs[:, 1, 1] = cosx
        rotXs[:, 1, 2] = -sinx
        rotXs[:, 2, 1] = sinx
        rotXs[:, 2, 2] = cosx

        rotYs[:, 0, 0] = cosy
        rotYs[:, 0, 2] = siny
        rotYs[:, 2, 0] = -siny
        rotYs[:, 2, 2] = cosy

        rotZs[:, 0, 0] = cosz
        rotZs[:, 0, 1] = -sinz
        rotZs[:, 1, 0] = sinz
        rotZs[:, 1, 1] = cosz

        res = rotZs.bmm(rotYs.bmm(rotXs))
        return res

    def evaluate(self, data_loader):
        """Virtual method to evaluate a model"""
        raise NotImplementedError

    def prepare_optimizer_opt(self):
        raise NotImplementedError

    def train(self, train_data_loader, n_epochs, valid_data_loader=None):
        """Run the model training"""
        self.train_len = len(train_data_loader.dataset)
        self.prepare_optimizer_opt()
        start_epoch = 0

        if self.checkpoint_dir is not None:
            self.save_checkpoint(current_epoch=0, current_loss=0)

        # Loop over epochs
        loop_bar = tqdm(range(start_epoch, n_epochs), desc="Epoch Progess")
        for i in loop_bar:
            summary = dict(epoch=i)

            # Train on this epoch
            start_time = time.time()
            train_summary = self.train_epoch(train_data_loader, i)
            if self.is_gradual_loss:
                log_one_number(
                    self.loss_utils.eye_loss_importance, "eye loss importance"
                )
                self.loss_utils.increase_eye_importance()
            train_summary["time"] = time.time() - start_time
            loop_bar.set_postfix(loss=train_summary["loss"])
            for (k, v) in train_summary.items():
                summary[f"train_{k}"] = v
                
            # Save summary, checkpoint
            self.save_summary(summary)
            if self.checkpoint_dir is not None:
                self.save_checkpoint(
                    current_epoch=i, current_loss=train_summary["loss"]
                )

        return self.summaries
