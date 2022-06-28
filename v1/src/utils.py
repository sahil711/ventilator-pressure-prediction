import numpy as np
import pandas as pd


def nm_features(df_tmp):
    df_tmp["R_cat"] = df_tmp["R"].map({20: 1, 50: 2, 5: 0})
    df_tmp["C_cat"] = df_tmp["C"].map({20: 1, 50: 2, 10: 0})
    df_tmp["RC_cat"] = df_tmp["R"].astype("str") + "-" + df_tmp["C"].astype("str")

    mapper = pd.Series(
        index=df_tmp["RC_cat"].unique(), data=np.arange(df_tmp["RC_cat"].nunique())
    )
    df_tmp["RC_cat"] = df_tmp["RC_cat"].map(mapper)
    df_tmp["bidc"] = df_tmp.groupby("breath_id").cumcount()
    df_tmp["u_in_lag_1"] = df_tmp.groupby("breath_id")["u_in"].shift(1).fillna(-999)
    df_tmp["u_in_lag_2"] = df_tmp.groupby("breath_id")["u_in"].shift(2).fillna(-999)
    df_tmp["u_in_lag_3"] = df_tmp.groupby("breath_id")["u_in"].shift(3).fillna(-999)
    df_tmp["u_in_lag_4"] = df_tmp.groupby("breath_id")["u_in"].shift(4).fillna(-999)

    df_tmp["u_in_cumsum"] = df_tmp.groupby("breath_id")["u_in"].cumsum()
    df_tmp["u_in_cumsum_lag_1"] = (
        df_tmp.groupby("breath_id")["u_in_cumsum"].shift(1).fillna(-999)
    )
    df_tmp["u_in_cumsum_lag_2"] = (
        df_tmp.groupby("breath_id")["u_in_cumsum"].shift(2).fillna(-999)
    )
    df_tmp["u_in_cumsum_lag_1-u_in_cumsum_lag_2"] = (
        df_tmp["u_in_cumsum_lag_1"] - df_tmp["u_in_cumsum_lag_2"]
    )

    df_tmp["u_in_cumsum_lag_3"] = (
        df_tmp.groupby("breath_id")["u_in_cumsum"].shift(3).fillna(0)
    )
    df_tmp["u_in_cumsum_lag_4"] = (
        df_tmp.groupby("breath_id")["u_in_cumsum"].shift(4).fillna(0)
    )

    df_tmp["u_in_cummean"] = df_tmp["u_in_cumsum"] / (
        df_tmp.groupby("breath_id")["u_in"].cumcount() + 1
    )
    df_tmp["u_in_cummax"] = df_tmp.groupby("breath_id")["u_in"].cummax()
    df_tmp["next_u_in"] = df_tmp.groupby("breath_id")["u_in"].shift(-1).fillna(0)

    df_tmp["area"] = df_tmp["time_step"] * df_tmp["u_in"]
    df_tmp["area"] = df_tmp.groupby("breath_id")["area"].cumsum()
    df_tmp["area_lag_1"] = df_tmp.groupby("breath_id")["area"].shift(1).fillna(0)
    df_tmp["area_lag_2"] = df_tmp.groupby("breath_id")["area"].shift(2).fillna(0)
    df_tmp["area_lead_1"] = df_tmp.groupby("breath_id")["area"].shift(-1).fillna(0)
    df_tmp["area_lead_2"] = df_tmp.groupby("breath_id")["area"].shift(-2).fillna(0)
    df_tmp["area_diff_lag_1"] = df_tmp["area"] - df_tmp["area_lag_1"]
    df_tmp["area_diff_lead_1"] = df_tmp["area"] - df_tmp["area_lead_1"]

    df_tmp["u_in_cumsum*time_step"] = df_tmp["u_in_cumsum"] * df_tmp["time_step"]
    df_tmp["u_in_cumsum*time_step_lag_1"] = (
        df_tmp.groupby("breath_id")["u_in_cumsum*time_step"].shift(1).fillna(0)
    )
    df_tmp["u_in_cumsum*time_step/c"] = df_tmp["u_in_cumsum*time_step"] / df_tmp["C"]
    df_tmp["u_in_cumsum*time_step/c_lag_1"] = (
        df_tmp.groupby("breath_id")["u_in_cumsum*time_step/c"].shift(1).fillna(0)
    )
    df_tmp["area/c"] = df_tmp["area"] / df_tmp["C"]
    df_tmp["area/c_lag_1"] = df_tmp.groupby("breath_id")["area/c"].shift(1).fillna(0)

    df_tmp["u_out_lag_1"] = df_tmp.groupby("breath_id")["u_out"].shift(1).fillna(0)

    df_tmp["time_step*u_out"] = df_tmp["time_step"] * df_tmp["u_out"]

    df_tmp["R+C"] = df_tmp["R"] + df_tmp["C"]
    df_tmp["R/C"] = df_tmp["R"] / df_tmp["C"]
    df_tmp["u_in/C"] = df_tmp["u_in"] / df_tmp["C"]
    df_tmp["u_in/R"] = df_tmp["u_in"] / df_tmp["R"]
    df_tmp["u_in_cumsum/C"] = df_tmp["u_in_cumsum"] / df_tmp["C"]
    df_tmp["u_in_cumsum/R"] = df_tmp["u_in_cumsum"] / df_tmp["R"]
    df_tmp["area*R/C"] = df_tmp["area"] * df_tmp["R/C"]
    df_tmp["u_in_cumsum*R/C"] = df_tmp["u_in_cumsum"] * df_tmp["R/C"]
    df_tmp["u_in_cumsum*R/C_lag_1"] = (
        df_tmp.groupby("breath_id")["u_in_cumsum*R/C"].shift(1).fillna(0)
    )

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
    df_tmp["u_in_lag_1_is_zero"] = df_tmp["u_in_lag_1"] == 0
    df_tmp["u_in_zero"] = df_tmp["u_in_lag_1"] == 0
    df_tmp["u_in_lead_1"] = df_tmp.groupby("breath_id")["u_in"].shift(-1)
    df_tmp["maop"] = df_tmp["bidc"] - df_tmp["breath_id"].map(
        df_tmp[df_tmp["u_in_zero"]].groupby("breath_id")["bidc"].min()
    )
    df_tmp["spike"] = (df_tmp["u_in"] > df_tmp["u_in_lag_1"]) & (
        df_tmp["u_in"] > df_tmp["u_in_lead_1"]
    )
    df_tmp["u_in_lag_1_is_zero_cumsum"] = df_tmp.groupby("breath_id")[
        "u_in_lag_1_is_zero"
    ].cumsum()
    df_tmp["is_max_u_in"] = df_tmp["u_in"] == df_tmp.groupby("breath_id")[
        "u_in"
    ].transform("max")
    df_tmp["nki"] = df_tmp["bidc"] - df_tmp["breath_id"].map(
        df_tmp.groupby("breath_id")["u_in"].apply(np.argmax)
    )
    df_tmp["nki2"] = df_tmp["bidc"] - df_tmp["breath_id"].map(
        df_tmp.groupby("breath_id")["u_in_cumsum"].apply(np.argmax)
    )
    df_tmp["nki3"] = df_tmp["nki"] - df_tmp["nki2"]
    df_tmp["nki4"] = df_tmp["bidc"] - df_tmp["breath_id"].map(
        df_tmp[df_tmp["u_in_lag_1_is_zero"]].groupby("breath_id")["bidc"].max()
    )
    df_tmp["u_in_cummax - u_in"] = df_tmp["u_in_cummax"] - df_tmp["u_in"]
    return df_tmp


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


def fc(df, include_R_C=True):
    feat = df[["breath_id", "u_in"]].groupby("breath_id")["u_in"]
    for i in [5, 10]:
        df["mean_{}_last_{}".format("u_in", i)] = (
            feat.rolling(window=i, min_periods=0).mean().values
        )
        df["min_{}_last_{}".format("u_in", i)] = (
            feat.rolling(window=i, min_periods=0).min().values
        )
        df["max_{}_last_{}".format("u_in", i)] = (
            feat.rolling(window=i, min_periods=0).max().values
        )
        df["std_{}_last_{}".format("u_in", i)] = (
            feat.rolling(window=i, min_periods=0).std().values
        )
        print(i)

    df = df.sort_values(["breath_id", "time_step"], ascending=[1, 0]).reset_index(
        drop=True
    )
    feat = df.groupby("breath_id")["u_in"]
    for i in [5, 10]:
        df["mean_{}_next_{}".format("u_in", i)] = (
            feat.rolling(window=i, min_periods=0).mean().values
        )
        df["min_{}_next_{}".format("u_in", i)] = (
            feat.rolling(window=i, min_periods=0).min().values
        )
        df["max_{}_next_{}".format("u_in", i)] = (
            feat.rolling(window=i, min_periods=0).max().values
        )
        df["std_{}_next_{}".format("u_in", i)] = (
            feat.rolling(window=i, min_periods=0).std().values
        )
        print(i)
    df.sort_values("id", inplace=True)
    df.fillna(0, inplace=True)
    df["u_in_cumsum"] = df.groupby("breath_id")["u_in"].cumsum()

    df["u_in_cummean"] = df["u_in_cumsum"] / (
        df.groupby("breath_id")["u_in"].cumcount() + 1
    )
    df["u_in_cummax"] = df.groupby("breath_id")["u_in"].cummax()

    if include_R_C:
        df["R+C"] = df["R"] + df["C"]
        df["R/C"] = df["R"] / df["C"]
        df["u_in/C"] = df["u_in"] / df["C"]
        df["u_in/R"] = df["u_in"] / df["R"]
        df["u_in_cumsum/C"] = df["u_in_cumsum"] / df["C"]
        df["u_in_cumsum/R"] = df["u_in_cumsum"] / df["R"]

    for i in [1, 2, 3, 4]:
        df["lag_{}_{}".format("u_in", i)] = (
            df.groupby("breath_id")["u_in"].shift(i).fillna(0)
        )
        df["lead_{}_{}".format("u_in", i)] = (
            df.groupby("breath_id")["u_in"].shift(i * -1).fillna(0)
        )
    df["auc"] = (
        (df["time_step"] - df.groupby("breath_id")["time_step"].shift(1))
        * (df["u_in"] + df["lag_u_in_1"])
        / 2
    ).fillna(
        0
    )  # area of a trapezium

    for i in [1, 2]:
        df["lag_{}_{}".format("auc", i)] = (
            df.groupby("breath_id")["auc"].shift(i).fillna(0)
        )
        df["lead_{}_{}".format("auc", i)] = (
            df.groupby("breath_id")["auc"].shift(i * -1).fillna(0)
        )

    for col in df.columns[df.columns.str.contains("lag_u_in|lead_u_in")]:
        df["per_change_u_in_{}".format(col)] = (df["u_in"] - df[col]) / (
            df["u_in"] + 1e-6
        )
    for col in df.columns[df.columns.str.contains("lag_auc|lead_auc")]:
        df["per_change_auc_{}".format(col)] = (df["auc"] - df[col]) / (df["auc"] + 1e-6)
    return df


def add_feature(df):
    df["R_val"] = df["R"]
    df["C_val"] = df["C"]
    df["u_out_val"] = df["u_out"]
    df["time_delta"] = df.groupby("breath_id")["time_step"].diff().fillna(0)
    df["delta"] = df["time_delta"] * df["u_in"]
    df["area"] = df.groupby("breath_id")["delta"].cumsum()

    df["cross"] = df["u_in"] * df["u_out"]
    df["cross2"] = df["time_step"] * df["u_out"]

    df["u_in_cumsum"] = (df["u_in"]).groupby(df["breath_id"]).cumsum()
    df["one"] = 1
    df["count"] = (df["one"]).groupby(df["breath_id"]).cumsum()
    df["u_in_cummean"] = df["u_in_cumsum"] / df["count"]

    df = df.drop(["count", "one"], axis=1)
    return df


def add_lag_feature(df):
    # https://www.kaggle.com/kensit/improvement-base-on-tensor-bidirect-lstm-0-173
    for lag in range(1, 5):
        df[f"breath_id_lag{lag}"] = df["breath_id"].shift(lag).fillna(0)
        df[f"breath_id_lag{lag}same"] = np.select(
            [df[f"breath_id_lag{lag}"] == df["breath_id"]], [1], 0
        )

        # u_in
        df[f"u_in_lag_{lag}"] = (
            df["u_in"].shift(lag).fillna(0) * df[f"breath_id_lag{lag}same"]
        )
        # df[f'u_in_lag_{lag}_back'] = df['u_in'].shift(-lag).fillna(0) * df[f'breath_id_lag{lag}same']
        df[f"u_in_time{lag}"] = df["u_in"] - df[f"u_in_lag_{lag}"]
        # df[f'u_in_time{lag}_back'] = df['u_in'] - df[f'u_in_lag_{lag}_back']
        df[f"u_out_lag_{lag}"] = (
            df["u_out"].shift(lag).fillna(0) * df[f"breath_id_lag{lag}same"]
        )
        # df[f'u_out_lag_{lag}_back'] = df['u_out'].shift(-lag).fillna(0) * df[f'breath_id_lag{lag}same']

    # breath_time
    df["time_step_lag"] = (
        df["time_step"].shift(1).fillna(0) * df[f"breath_id_lag{lag}same"]
    )
    df["breath_time"] = df["time_step"] - df["time_step_lag"]

    drop_columns = ["time_step_lag"]
    drop_columns += [f"breath_id_lag{i}" for i in range(1, 5)]
    drop_columns += [f"breath_id_lag{i}same" for i in range(1, 5)]
    df = df.drop(drop_columns, axis=1)

    # fill na by zero
    df = df.fillna(0)
    return df
