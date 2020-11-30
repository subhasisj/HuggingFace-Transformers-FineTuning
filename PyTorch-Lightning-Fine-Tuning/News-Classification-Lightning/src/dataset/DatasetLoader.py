import torch
from torch import nn
from torch import functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import random_split
from torch.utils.data import RandomSampler
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import pytorch_lightning as pl
import pandas as pd


class DatasetLoader(pl.LightningDataModule):
    def __init__(self, *args, **kwargs):

        super().__init__()

        if isinstance(args, tuple):
            args = args[0]

        self.hparams = args

        print("args:", args)
        print("kwargs:", kwargs)

        print("Loading BERT tokenizer")
        print(f"HPARAMS:{self.hparams}")
        print(f"PRETRAINED:{self.hparams.pretrained}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.pretrained)
        print("Type tokenizer:", type(self.tokenizer))

    # def prepare_data(self):

    #     print()

    def setup(self, stage=None):

        df = pd.read_csv("../../../Data/news/news.csv", index_col="Unnamed: 0")
        df["label"] = df["label"].apply(lambda x: 1 if x == "FAKE" else 0)
        # Report the number of sentences.
        print("Number of training sentences: {:,}\n".format(df.shape[0]))

        # Get the lists of sentences and their labels.
        sentences = df.text.values
        labels = df.label.values

        t = self.tokenizer(
            sentences.tolist(),
            add_special_tokens=True,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        input_ids = t["input_ids"]
        attention_mask = t["attention_mask"]

        labels = torch.tensor(labels)

        dataset = TensorDataset(input_ids, attention_mask, labels)

        train_size = int(self.hparams.training_portion * len(dataset))
        val_size = len(dataset) - train_size

        print("{:>5,} training samples".format(train_size))
        print("{:>5,} validation samples".format(val_size))

        self.train_dataset, self.val_dataset = random_split(
            dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):

        return DataLoader(
            self.train_dataset,
            sampler=RandomSampler(self.train_dataset),
            batch_size=self.hparams.batch_size,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            sampler=RandomSampler(self.val_dataset),
            batch_size=self.hparams.batch_size,
            shuffle=False,
        )
