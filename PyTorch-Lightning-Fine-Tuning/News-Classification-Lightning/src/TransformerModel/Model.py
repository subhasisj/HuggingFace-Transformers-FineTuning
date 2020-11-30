# https://arthought.com/transformer-model-fine-tuning-for-text-classification-with-pytorch-lightning/

import pandas as pd
import numpy as np
import os


import torch
from torch import optim
import torch.nn as nn
import torch.functional as F

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import AutoConfig

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

# from transformers.modeling_bart import Attention

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping


class Model(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.save_hyperparameters()

        self._frozen = False

        config = AutoConfig.from_pretrained(
            self.hparams.pretrained,
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False,
        )

        # print(config)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.hparams.pretrained, config=config
        )

        print(f"Model Type:{type(self.model)}")

    def forward(self, batch):

        input_ids = batch[0]
        attention_mask = batch[1]

        has_labels = len(batch) > 2
        labels = batch[2] if has_labels else None

        if has_labels:
            loss, logits = self.model(
                input_ids, attention_mask=attention_mask, labels=labels,return_dict=False
            )

        if not has_labels:
            loss, logits = None, self.model(
                input_ids, attention_mask=attention_mask, labels=labels,return_dict=False
            )

        return loss, logits

    def training_step(self, batch, batch_nb):
        loss, logits = self(batch)

        tensorboard_logs = {"train_loss": loss}

        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_nb):

        # print(f"The batch:{(batch)}")
        loss, logits = self(batch)
        # print(f"The logits:{(logits)}")

        labels = batch[2]
        predictions = torch.argmax(logits, dim=1)
        accuracy = (labels == predictions).float().mean()

        return {"val_loss": loss, "accuracy": accuracy}

    def validation_epoch_end(self, validation_step_outputs):

        avg_loss = torch.stack([x["val_loss"] for x in validation_step_outputs]).mean()
        avg_accuracy = torch.stack(
            [x["accuracy"] for x in validation_step_outputs]
        ).mean()

        tensorboard_logs = {"val_loss": avg_loss, "val_accuracy": avg_accuracy}
        return {
            "val_loss": avg_loss,
            "log": tensorboard_logs,
            "progress_bar": {"avg_loss": avg_loss, "avg_accuracy": avg_accuracy},
        }

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]

        optimizers_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(
            optimizers_grouped_parameters, lr=self.hparams.learning_rate, eps=1e-8
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=self.hparams.num_training_steps,
        )

        return [optimizer], [scheduler]

    def freeze(self):
        # freeze all layers, except the final classifier layers
        for name, param in self.model.named_parameters():
            if "classifier" not in name:  # classifier layer
                param.requires_grad = False

        self._frozen = True

    def on_epoch_start(self):
        """pytorch lightning hook"""
        if self.current_epoch < self.hparams.nr_frozen_epochs:
            self.freeze()

        if self.current_epoch >= self.hparams.nr_frozen_epochs:
            self.unfreeze()