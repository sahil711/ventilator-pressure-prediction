import pytorch_lightning as pl
import argparse
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
import pandas as pd
from sklearn.model_selection import GroupKFold
import os
from litmodellib import Model
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import datalib
import torch

DATA_DIR = "/mnt/disks/extra_data/kaggle/ventilator_prediction/"
R_MAP = {5: 0, 50: 1, 20: 2}
C_MAP = {20: 0, 50: 1, 10: 2}


def create_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


def map_dataset(data):
    data.breath_id = data.breath_id.map(
        dict(zip(data.breath_id.unique().tolist(), range(data.breath_id.nunique())))
    )
    data.R = data.R.map(R_MAP)
    data.C = data.C.map(C_MAP)
    return data


def train_model(config_path, fold_nums):
    df = pd.read_csv(DATA_DIR + "train.csv")
    folds = GroupKFold(n_splits=5)
    folds = list(folds.split(df, groups=df["breath_id"]))
    for i in fold_nums:
        train = df.iloc[folds[i][0]]
        val = df.iloc[folds[i][1]]
        train = map_dataset(train)
        val = map_dataset(val)
        config = OmegaConf.load(config_path)
        path = "../model_zoo/{}".format(config["experiment_name"])
        create_path(path)
        if not os.path.exists(path + "/config.yaml"):
            OmegaConf.save(config, path + "/config.yaml")
        path = path + "/fold_{}".format(i)

        train_df = getattr(datalib, config.dataset.train["class"])(
            **config.dataset.train["kwargs"], df=train
        )
        val_df = getattr(datalib, config.dataset.val["class"])(
            **config.dataset.val["kwargs"], df=val
        )

        train_dl = DataLoader(
            dataset=train_df,
            shuffle=True,
            num_workers=config.num_workers.train,
            batch_size=config.batch_size.train,
        )
        val_dl = DataLoader(
            dataset=val_df,
            shuffle=False,
            num_workers=config.num_workers.val,
            batch_size=config.batch_size.val,
        )
        print(len(train_dl), len(val_dl))
        esr = EarlyStopping(**config.esr)

        ckpt_callback = ModelCheckpoint(
            dirpath=path,
            monitor="val_MAE",
            save_top_k=3,
            mode="min",
            filename="model-{epoch}-{val_MAE:.4f}-{val_loss:.4f}",
        )

        lr_callback = LearningRateMonitor(
            logging_interval=config.schedular.schedular_interval
        )

        num_gpus = torch.cuda.device_count()

        train_iter = int(len(train_dl) / num_gpus)

        lit_model = Model(config, num_train_iter=train_iter)

        precision = 16 if config.mp_training else 32

        # wandb_exp_name = "{}_fold_{}".format(config.experiment_name, fold_num)
        logger = WandbLogger(project=config.wandb_project, name=config.experiment_name)
        train = pl.Trainer(
            logger=logger,
            log_every_n_steps=50,
            gpus=-1,
            accelerator="ddp",
            max_epochs=config.num_epochs,
            precision=precision,
            deterministic=True,
            callbacks=[esr, ckpt_callback, lr_callback],
            # resume_from_checkpoint="",
        )
        train.fit(model=lit_model, train_dataloader=train_dl, val_dataloaders=val_dl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="config path", required=True, type=str)
    parser.add_argument(
        "--folds", help="folds to train", nargs="+", default=[0, 1, 2, 3, 4]
    )
    args = parser.parse_args()
    folds = [int(x) for x in args.folds]
    print("folds {}".format(folds))
    train_model(args.config, folds)
