from torch.utils.data import Dataset
import torch


class VentilaltorData(Dataset):
    def __init__(
        self, df, id_col, seq_feature_cols, identity_feature_cols, target_col=None
    ):
        self.df = df
        self.id_col = id_col
        self.seq_feature_cols = seq_feature_cols
        self.identity_feature_cols = identity_feature_cols
        self.target_col = target_col

    def __len__(self):
        return self.df[self.id_col].nunique()

    def __getitem__(self, idx):
        data = self.df[self.df[self.id_col] == idx]
        seq_data = torch.tensor(data[self.seq_feature_cols].values, dtype=torch.float32)
        identity_data = torch.tensor(
            data[self.identity_feature_cols].iloc[0].values, dtype=torch.long
        )
        if self.target_col is not None:
            target = torch.tensor(data[self.target_col].values, dtype=torch.float32)
            return {"seq": seq_data, "identity_data": identity_data, "target": target}
        else:
            return {"seq": seq_data, "identity_data": identity_data}
