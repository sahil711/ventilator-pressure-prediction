import pytorch_lightning as pl
import argparse
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import GroupKFold
import os
from litmodellib import Model
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import datalib
import torch
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm
import math
import gc

# import numpy as np
from utils import fc, create_feats
import pickle

# add_feature, add_lag_feature

DATA_DIR = "/mnt/disks/extra_data/kaggle/ventilator_prediction/"
R_MAP = {5: 0, 50: 1, 20: 2}
C_MAP = {20: 0, 50: 1, 10: 2}


def load_dict(path):
    with open(path, "rb") as f:
        return pickle.load(f)


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


# def get_u_in_matrix(grp):
#     i, data = grp
#     m = torch.tensor(pairwise_distances(data.u_in.values.reshape(-1, 1)))
#     m = (m - m.mean(dim=0)) / m.std(dim=0)
#     return {i: m}
def get_u_in_matrix(grp):
    i, data = grp
    t1 = torch.tensor(pairwise_distances(data.u_in.values.reshape(-1, 1)))
    t2 = torch.tensor(pairwise_distances(data.time_step.values.reshape(-1, 1)))
    t3 = torch.tensor(pairwise_distances(data.u_out.values.reshape(-1, 1)))
    return {i: torch.stack([t1, t2, t3]).float()}


def get_matrix_dict(df):
    grp_df = df[["breath_id", "u_in", "time_step", "u_out"]].groupby("breath_id")
    u_in_matrix_dict = []
    for i in tqdm(grp_df, total=len(grp_df)):
        u_in_matrix_dict.append(get_u_in_matrix(i))
    _dict = dict((key, d[key]) for d in u_in_matrix_dict for key in d)
    return _dict


def train_model(config_path, fold_nums):
    df = pd.read_csv(DATA_DIR + "train.csv")
    df["R_1"] = df["R"].values
    df["C_1"] = df["C"].values
    config = OmegaConf.load(config_path)
    df = create_feats(df)
    # if config.create_features:
    #     df = fc(df)
    #     # df = add_feature(df)
    #     # df = add_lag_feature(df)

    # df = pd.read_pickle("../data/v2_len_40_processed_train_regression.pkl")

    df = df.groupby("breath_id").head(config.seq_len)
    config.model.kwargs["output_dim"] = 1

    if config.normalization.is_norm:
        scl = RobustScaler()
        print(config.dataset.train.kwargs.numerical_columns)
        for col in config.dataset.train.kwargs.numerical_columns:
            if col != "preds":
                df[col] = scl.fit_transform(df[[col]])

    folds = GroupKFold(n_splits=15)
    folds = list(folds.split(df, groups=df["breath_id"]))

    for i in fold_nums:
        train = df.iloc[folds[i][0]]
        val = df.iloc[folds[i][1]]
        print(train.shape, val.shape)
        train = map_dataset(train)
        val = map_dataset(val)
        train_grp_dict = get_group_dict(train)
        val_grp_dict = get_group_dict(val)
        # train_grp_dict = load_dict(
        #     "../data/v2_len_40_regression_train_grp_dict_fold_{}.pkl".format(i)
        # )
        # val_grp_dict = load_dict(
        #     "../data/v2_len_40_regression_val_grp_dict_fold_{}.pkl".format(i)
        # )

        path = "../experiments/{}".format(config["experiment_name"])
        create_path(path)
        if not os.path.exists(path + "/config.yaml"):
            OmegaConf.save(config, path + "/config.yaml")
        path = path + "/fold_{}".format(i)

        train_df = getattr(datalib, config.dataset.train["class"])(
            **config.dataset.train["kwargs"], group_dict=train_grp_dict,
        )
        val_df = getattr(datalib, config.dataset.val["class"])(
            **config.dataset.val["kwargs"], group_dict=val_grp_dict,
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
            save_top_k=1,
            mode="min",
            filename="model-{epoch}-{val_MAE:.4f}-{val_loss:.4f}",
        )

        lr_callback = LearningRateMonitor(
            logging_interval=config.schedular.schedular_interval
        )

        num_gpus = torch.cuda.device_count()

        train_iter = math.ceil(len(train_dl) / num_gpus)
        num_steps = math.ceil((len(train_dl) * config.num_epochs) / num_gpus)
        lr_step_size = (config.last_lr / config.start_lr) ** (1 / config.num_epochs)
        print(lr_step_size)

        lit_model = Model(
            config,
            num_train_iter=train_iter,
            num_steps=num_steps,
            lr_step_size=lr_step_size,
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
            # gradient_clip_val=0.5,
        )
        train.fit(model=lit_model, train_dataloader=train_dl, val_dataloaders=val_dl)
        del val_grp_dict, train_grp_dict, train_df, val_df, train_dl, val_dl
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
