# %%
import os

print(os.getcwd())
# %%
from TransformerModel.Model import Model
from dataset.DatasetLoader import DatasetLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import argparse
from argparse import ArgumentParser, ArgumentTypeError

# %%


def run_training(arguments_parser):
    data = DatasetLoader(arguments_parser)
    data.setup()

    arguments_parser.num_training_steps = (
        len(data.train_dataloader()) * arguments_parser.max_epochs
    )

    dict_args = vars(arguments_parser)

    model = Model(**dict_args)

    arguments_parser.early_stop_callback = EarlyStopping("val_loss")

    trainer = pl.Trainer.from_argparse_args(arguments_parser)

    trainer.fit(model, data)


# %%
if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--pretrained", type=str, default="bert-base-uncased")
    parser.add_argument("--nr_frozen_epochs", type=int, default=5)
    parser.add_argument("--training_portion", type=float, default=0.9)
    parser.add_argument("--batch_size", type=float, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--frac", type=float, default=1)

    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    run_training(args)


# %%
