from torch.utils.data import Dataset
import torch
import numpy as np


class VentilatorData(Dataset):
    def __init__(
        self, group_dict, categorical_columns, numerical_columns, target_column=None
    ):
        self.group_dict = group_dict
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.target_column = target_column

    def __len__(self):
        return len(self.group_dict)

    def __getitem__(self, idx):
        # data = self.group_df.get_group(idx)
        # num_data = torch.tensor(
        #     data[self.numerical_columns].values, dtype=torch.float32
        # )
        # cat_data = torch.tensor(data[self.categorical_columns].values, dtype=torch.long)

        data = self.group_dict[idx]
        cat_data = torch.tensor(
            np.array([data[k] for k in self.categorical_columns]).T, dtype=torch.long
        )
        num_data = torch.tensor(
            np.array([data[k] for k in self.numerical_columns]).T, dtype=torch.float32
        )
        if self.target_column is not None:
            # tar_data = torch.tensor(
            #     data[self.target_column].values, dtype=torch.float32
            # )
            tar_data = np.array(data[self.target_column])
            return {"num": num_data, "cat": cat_data, "target": tar_data}
        else:
            return {"num": num_data, "cat": cat_data}


class VentilatorData2(Dataset):
    def __init__(
        self, group_dict, categorical_columns, numerical_columns, target_column=None
    ):
        self.group_dict = group_dict
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.target_column = target_column

    def __len__(self):
        return len(self.group_dict)

    def __getitem__(self, idx):

        data = self.group_dict[idx]
        cat_data = torch.tensor(
            np.array([data[k] for k in self.categorical_columns]).T, dtype=torch.long
        )
        num_data = torch.tensor(
            np.array([data[k] for k in self.numerical_columns]).T, dtype=torch.float32
        )
        u_out = torch.tensor(np.array(data["u_out"]), dtype=torch.long)

        if self.target_column is not None:
            tar_data = torch.tensor(
                np.array(data[self.target_column]), dtype=torch.float32
            )
            return {
                "num": num_data,
                "cat": cat_data,
                "target": tar_data,
                "u_out": u_out,
            }
        else:
            return {"num": num_data, "cat": cat_data, "u_out": u_out}


class VentilatorDataClassification(Dataset):
    def __init__(
        self, group_dict, categorical_columns, numerical_columns, target_column=None
    ):
        self.group_dict = group_dict
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.target_column = target_column

    def __len__(self):
        return len(self.group_dict)

    def __getitem__(self, idx):

        data = self.group_dict[idx]
        cat_data = torch.tensor(
            np.array([data[k] for k in self.categorical_columns]).T, dtype=torch.long
        )
        num_data = torch.tensor(
            np.array([data[k] for k in self.numerical_columns]).T, dtype=torch.float32
        )
        u_out = torch.tensor(np.array(data["u_out"]), dtype=torch.long)

        if self.target_column is not None:
            tar_data = torch.tensor(
                np.array(data[self.target_column]), dtype=torch.long
            )
            return {
                "num": num_data,
                "cat": cat_data,
                "target": tar_data,
                "u_out": u_out,
            }
        else:
            return {"num": num_data, "cat": cat_data, "u_out": u_out}


class VentilatorDataClassificationV2(Dataset):
    def __init__(
        self,
        group_dict,
        breath_df,
        categorical_columns,
        numerical_columns,
        target_column=None,
    ):
        self.group_dict = group_dict
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.target_column = target_column
        self.breath_df = breath_df

    def __len__(self):
        return len(self.breath_df)

    def __getitem__(self, idx):
        breath_id = self.breath_df.iloc[idx]
        data = self.group_dict[breath_id]

        cat_data = torch.tensor(
            np.array([data[k] for k in self.categorical_columns]).T, dtype=torch.long
        )
        num_data = torch.tensor(
            np.array([data[k] for k in self.numerical_columns]).T, dtype=torch.float32
        )
        u_out = torch.tensor(np.array(data["u_out"]), dtype=torch.long)

        if self.target_column is not None:
            tar_data = torch.tensor(
                np.array(data[self.target_column]), dtype=torch.long
            )
            return {
                "num": num_data,
                "cat": cat_data,
                "target": tar_data,
                "u_out": u_out,
            }
        else:
            return {"num": num_data, "cat": cat_data, "u_out": u_out}
