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
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm

import numpy as np
from utils import fc

DATA_DIR = "/mnt/disks/extra_data/kaggle/ventilator_prediction/"
R_MAP = {5: 0, 50: 1, 20: 2}
C_MAP = {20: 0, 50: 1, 10: 2}


def create_feats(df_tmp):
    c = "u_in"
    LAG_WINDOW_RANGE = range(3)
    df_tmp = df_tmp.assign(
        **{
            f"{c}_t-{t}": df_tmp.groupby("breath_id")[c].shift(t)
            for t in LAG_WINDOW_RANGE
        }
    )
    use_fts = [f"{c}_t-{t}" for t in LAG_WINDOW_RANGE]
    df_tmp["u_in_lag_1"] = df_tmp.groupby("breath_id")["u_in"].shift(1).fillna(0)
    df_tmp["u_in_lag_2"] = df_tmp.groupby("breath_id")["u_in"].shift(2).fillna(0)
    df_tmp["u_in_cumsum"] = df_tmp.groupby("breath_id")["u_in"].cumsum()
    df_tmp["u_in_cummean"] = df_tmp["u_in_cumsum"] / (
        df_tmp.groupby("breath_id")["u_in"].cumcount() + 1
    )
    df_tmp["u_in_cummax"] = df_tmp.groupby("breath_id")["u_in"].cummax()
    df_tmp["next_u_in"] = df_tmp.groupby("breath_id")["u_in"].shift(-1).fillna(0)

    df_tmp["roll_u_in_max"] = df_tmp[use_fts].max(axis=1).fillna(0)
    df_tmp["roll_u_in_min"] = df_tmp[use_fts].min(axis=1).fillna(0)

    df_tmp["time_lag_1"] = df_tmp.groupby("breath_id")["time_step"].shift(1).fillna(0)
    df_tmp["time_lag_2"] = df_tmp.groupby("breath_id")["time_step"].shift(2).fillna(0)

    df_tmp.drop(use_fts, axis=1, inplace=True)

    df_tmp["area"] = df_tmp["time_step"] * df_tmp["u_in"]
    df_tmp["area"] = df_tmp.groupby("breath_id")["area"].cumsum()

    df_tmp["u_out_lag_1"] = df_tmp.groupby("breath_id")["u_out"].shift(1).fillna(0)
    df_tmp["u_out_lag_2"] = df_tmp.groupby("breath_id")["u_out"].shift(2).fillna(0)

    df_tmp["time_step*u_out"] = df_tmp["time_step"] * df_tmp["u_out"]

    df_tmp["R+C"] = df_tmp["R"] + df_tmp["C"]
    df_tmp["R/C"] = df_tmp["R"] / df_tmp["C"]
    df_tmp["u_in/C"] = df_tmp["u_in"] / df_tmp["C"]
    df_tmp["u_in/R"] = df_tmp["u_in"] / df_tmp["R"]
    df_tmp["u_in_cumsum/C"] = df_tmp["u_in_cumsum"] / df_tmp["C"]
    df_tmp["u_in_cumsum/R"] = df_tmp["u_in_cumsum"] / df_tmp["R"]
    df_tmp["timestep_diff"] = (
        df_tmp["time_step"] - df_tmp.groupby("breath_id")["time_step"].shift(1)
    ).fillna(0)
    df_tmp["u_in_diff"] = (
        df_tmp["u_in"] - df_tmp.groupby("breath_id")["u_in"].shift(1)
    ).fillna(0)
    df_tmp["u_in_pct_change"] = (
        df_tmp["u_in_diff"] / (df_tmp["u_in_lag_1"] + 1e-4)
    ).fillna(0)
    df_tmp["u_in_diff_next"] = (
        df_tmp["u_in"] - df_tmp.groupby("breath_id")["u_in"].shift(-1)
    ).fillna(0)
    df_tmp["u_in_log"] = np.log1p(df_tmp["u_in"])
    df_tmp["u_in_cumsum_log"] = np.log1p(df_tmp["u_in_cumsum"])
    return df_tmp


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
    df["R_1"] = df["R"].values
    df["C_1"] = df["C"].values
    config = OmegaConf.load(config_path)
    # df = create_feats(df)
    df = fc(df)
    df = df.groupby("breath_id").head(config.seq_len)
    if config.normalization.is_norm:
        scl = RobustScaler()
        print(config.dataset.train.kwargs.numerical_columns)
        for col in config.dataset.train.kwargs.numerical_columns:
            df[col] = scl.fit_transform(df[[col]])

    folds = GroupKFold(n_splits=5)
    folds = list(folds.split(df, groups=df["breath_id"]))
    for i in fold_nums:
        train = df.iloc[folds[i][0]]
        val = df.iloc[folds[i][1]]
        print(train.shape, val.shape)
        train = map_dataset(train)
        val = map_dataset(val)
        train_grp_dict = get_group_dict(train)
        val_grp_dict = get_group_dict(val)
        path = "../experiments/{}".format(config["experiment_name"])
        create_path(path)
        if not os.path.exists(path + "/config.yaml"):
            OmegaConf.save(config, path + "/config.yaml")
        path = path + "/fold_{}".format(i)

        train_df = getattr(datalib, config.dataset.train["class"])(
            **config.dataset.train["kwargs"], group_dict=train_grp_dict
        )
        val_df = getattr(datalib, config.dataset.val["class"])(
            **config.dataset.val["kwargs"], group_dict=val_grp_dict
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

        train_iter = int(len(train_dl) / num_gpus)
        num_steps = int((len(train_dl) * config.num_epochs) / num_gpus)

        lit_model = Model(config, num_train_iter=train_iter, num_steps=num_steps)
        # state_dict = torch.load(
        #     "../experiments/RNN-simple-v4/fold_0/model-epoch=97-val_MAE=0.2041-val_loss=0.2041.ckpt"
        # )["state_dict"]
        # lit_model.load_state_dict(state_dict=state_dict)

        # precision = 16 if config.mp_training else 32

        # wandb_exp_name = "{}_fold_{}".format(config.experiment_name, fold_num)
        logger = WandbLogger(project=config.wandb_project, name=config.experiment_name)
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
