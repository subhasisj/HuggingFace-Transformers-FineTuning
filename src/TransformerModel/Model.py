# https://arthought.com/transformer-model-fine-tuning-for-text-classification-with-pytorch-lightning/

import pandas as pd
import numpy as np
import os


import torch
import torch.nn as nn
import torch.functional as F

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import AutoConfig

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers.modeling_bart import Attention

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Earlystopping


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

        print(config)

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
                input_ids, attention_mask=attention_mask, labels=labels
            )

        if not has_labels:
            loss, logits = None, self.model(
                input_ids, attention_mask=attention_mask, labels=labels
            )

        return loss, logits

    def training_step(self, batch, batch_nb):
        loss, logits = self(batch)

        tensorboard_logs = {"train_loss": loss}

        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_nb):

        loss, logits = self(batch)

        labels = batch[2]
        predictions = torch.argmax(logits, dim=1)
        accuracy = (labels == predictions).float().mean()

        return {"val_loss": loss, "accuracy": accuracy}
