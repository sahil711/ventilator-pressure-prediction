def fc(df):
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
