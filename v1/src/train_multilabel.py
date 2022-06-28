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
from litmodellib import ClassifcationMultiLabelModel
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import datalib
import torch
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm
import joblib
import math
import gc

# import numpy as np
from utils import fc

DATA_DIR = "/mnt/disks/extra_data/kaggle/ventilator_prediction/"
R_MAP = {5: 0, 50: 1, 20: 2}
C_MAP = {20: 0, 50: 1, 10: 2}
RC_MAP = {
    "2050": 0,
    "2020": 1,
    "5020": 2,
    "5050": 3,
    "550": 4,
    "520": 5,
    "5010": 6,
    "2010": 7,
    "510": 8,
}


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


def get_group_dict(data):
    group_dict = {}
    for i, x in tqdm(data.groupby("breath_id"), total=data.breath_id.nunique()):
        m = x.to_dict(orient="records")
        group_dict[i] = {
            k: [a[k] for a in m]
            for k in [k for k in m[0].keys() if k not in ["id", "breath_id"]]
        }
    return group_dict


def train_model(config_path, fold_nums):
    df = pd.read_csv(DATA_DIR + "train.csv")
    config = OmegaConf.load(config_path)
    pressure_dict = dict(
        zip(df["pressure"].unique().tolist(), range(df["pressure"].nunique()))
    )
    pressure_reverse_dict = {v: k for k, v in pressure_dict.items()}
    joblib.dump(pressure_reverse_dict, "../pressure_mapper.pkl")
    df["pressure"] = df["pressure"].map(pressure_dict)
    df["RC"] = df["R"].astype("str") + df["C"].astype("str")
    df["RC"] = df["RC"].map(RC_MAP)
    # df = create_feats(df)
    df = fc(df, include_R_C=False)
    df = df.groupby("breath_id").head(config.seq_len)
    df.reset_index(drop=True, inplace=True)
    num_classes = df["pressure"].nunique()
    config.model.kwargs["output_dim"] = num_classes
    if config.normalization.is_norm:
        scl = RobustScaler()
        print(config.dataset.train.kwargs.numerical_columns)
        for col in config.dataset.train.kwargs.numerical_columns:
            df[col] = scl.fit_transform(df[[col]])

    df = map_dataset(df)
    grp_dict = get_group_dict(df)
    folds = GroupKFold(n_splits=5)
    folds = list(folds.split(df, groups=df["breath_id"]))
    for i in fold_nums:
        train = df.iloc[folds[i][0]]
        val = df.iloc[folds[i][1]]
        print(train.shape, val.shape)
        train_tar_df = train[["breath_id", "RC"]].drop_duplicates()
        val_tar_df = val[["breath_id", "RC"]].drop_duplicates()
        path = "../experiments/{}".format(config["experiment_name"])
        create_path(path)
        if not os.path.exists(path + "/config.yaml"):
            OmegaConf.save(config, path + "/config.yaml")
        path = path + "/fold_{}".format(i)

        train_df = getattr(datalib, config.dataset.train["class"])(
            **config.dataset.train["kwargs"], group_dict=grp_dict, label_df=train_tar_df
        )
        val_df = getattr(datalib, config.dataset.val["class"])(
            **config.dataset.val["kwargs"], group_dict=grp_dict, label_df=val_tar_df
        )

        train_dl = DataLoader(
            dataset=train_df,
            shuffle=True,
            num_workers=config.num_workers.train,
            batch_size=config.batch_size.train,
            pin_memory=True,
        )
        val_dl = DataLoader(
            dataset=val_df,
            shuffle=False,
            num_workers=config.num_workers.val,
            batch_size=config.batch_size.val,
            pin_memory=True,
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

    train_iter = math.ceil(len(train_dl) / num_gpus)
    num_steps = math.ceil((len(train_dl) * config.num_epochs) / num_gpus)

    lit_model = ClassifcationMultiLabelModel(
        config,
        num_train_iter=train_iter,
        num_steps=num_steps,
        mapping="../pressure_mapper.pkl",
        topk=config.topk,
    )
    print(len(train_dl))
    print(train_iter)
    # precision = 16 if config.mp_training else 32

    wandb_exp_name = "{}_fold_{}".format(config.experiment_name, i)
    logger = WandbLogger(project=config.wandb_project, name=wandb_exp_name)
    train = pl.Trainer(
        logger=logger,
        log_every_n_steps=50,
        gpus=-1,
        accelerator="ddp",
        max_epochs=config.num_epochs,
        # precision=precision,
        deterministic=True,
        callbacks=[esr, ckpt_callback, lr_callback],
        # resume_from_checkpoint="../experiments/RNN-simple-v4/fold_0/model-epoch=71-val_MAE=0.2419-val_loss=0.2419.ckpt",
    )
    train.fit(model=lit_model, train_dataloader=train_dl, val_dataloaders=val_dl)
    gc.collect()


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
