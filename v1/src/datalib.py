from torch.utils.data import Dataset
import torch
import numpy as np


class VentilatorDataRegression(Dataset):
    def __init__(
        self, group_dict, categorical_columns, numerical_columns, target_column=None,
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


class VentilatorDataMultiLabel(Dataset):
    def __init__(
        self,
        group_dict,
        label_df,
        categorical_columns,
        numerical_columns,
        target_column=None,
    ):
        self.group_dict = group_dict
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.target_column = target_column
        self.label_df = label_df

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        _id = self.label_df.iloc[idx]
        breath_id, rc = _id["breath_id"], _id["RC"]
        data = self.group_dict[breath_id]

        cat_data = torch.tensor(
            np.array([data[k] for k in self.categorical_columns]).T, dtype=torch.long
        )
        num_data = torch.tensor(
            np.array([data[k] for k in self.numerical_columns]).T, dtype=torch.float32
        )
        u_out = torch.tensor(np.array(data["u_out"]), dtype=torch.long)
        rc = torch.tensor([rc], dtype=torch.long)

        if self.target_column is not None:
            tar_data = torch.tensor(
                np.array(data[self.target_column]), dtype=torch.long
            )
            return {
                "num": num_data,
                "cat": cat_data,
                "pressure": tar_data,
                "rc": rc,
                "u_out": u_out,
            }
        else:
            return {"num": num_data, "cat": cat_data, "u_out": u_out}


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


class VentilatorDataClassificationV3(Dataset):
    def __init__(
        self,
        group_dict,
        categorical_columns,
        numerical_columns,
        seq_len,
        target_column=None,
        flip_aug=0,
        shift_aug=0,
    ):
        self.group_dict = group_dict
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.target_column = target_column
        self.flip_aug = flip_aug
        self.shift_aug = shift_aug
        self.seq_len = seq_len

    def __len__(self):
        return len(self.group_dict)

    def __getitem__(self, idx):

        if torch.rand(1).item() > self.shift_aug:
            start_idx = 0
        else:
            start_idx = torch.multinomial(
                torch.tensor([0] + [1] * 9, dtype=torch.float32), 1
            ).item()
        end_idx = start_idx + self.seq_len

        data = self.group_dict[idx]
        cat_data = torch.tensor(
            np.array([data[k] for k in self.categorical_columns]).T[
                start_idx:end_idx, :
            ],
            dtype=torch.long,
        )
        num_data = torch.tensor(
            np.array([data[k] for k in self.numerical_columns]).T[start_idx:end_idx, :],
            dtype=torch.float32,
        )
        u_out = torch.tensor(
            np.array(data["u_out"])[start_idx:end_idx], dtype=torch.long
        )

        if self.target_column is not None:
            tar_data = torch.tensor(
                np.array(data[self.target_column])[start_idx:end_idx], dtype=torch.long
            )
            if torch.rand(1).item() > self.flip_aug:
                return {
                    "num": num_data,
                    "cat": cat_data,
                    "target": tar_data,
                    "u_out": u_out,
                }
            else:
                return {
                    "num": torch.flip(num_data, dims=[0]),
                    "cat": torch.flip(cat_data, dims=[0]),
                    "target": torch.flip(tar_data, dims=[0]),
                    "u_out": torch.flip(u_out, dims=[0]),
                }

        else:
            if torch.rand(1).item() > self.flip_aug:
                return {"num": num_data, "cat": cat_data, "u_out": u_out}
            else:
                return {
                    "num": torch.flip(num_data, dims=[0]),
                    "cat": torch.flip(cat_data, dims=[0]),
                    "u_out": torch.flip(u_out, dims=[0]),
                }


class VentilatorDataClassification(Dataset):
    def __init__(
        self,
        group_dict,
        categorical_columns,
        numerical_columns,
        target_column=None,
        flip_aug=0,
    ):
        self.group_dict = group_dict
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.target_column = target_column
        self.flip_aug = flip_aug

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
            if torch.rand(1).item() > self.flip_aug:
                return {
                    "num": num_data,
                    "cat": cat_data,
                    "target": tar_data,
                    "u_out": u_out,
                }
            else:
                return {
                    "num": torch.flip(num_data, dims=[0]),
                    "cat": torch.flip(cat_data, dims=[0]),
                    "target": torch.flip(tar_data, dims=[0]),
                    "u_out": torch.flip(u_out, dims=[0]),
                }

        else:
            if torch.rand(1).item() > self.flip_aug:
                return {"num": num_data, "cat": cat_data, "u_out": u_out}
            else:
                return {
                    "num": torch.flip(num_data, dims=[0]),
                    "cat": torch.flip(cat_data, dims=[0]),
                    "u_out": torch.flip(u_out, dims=[0]),
                }


class VentilatorDataMatrixClassification(Dataset):
    def __init__(
        self,
        group_dict,
        matrix_dict,
        categorical_columns,
        numerical_columns,
        target_column=None,
    ):
        self.group_dict = group_dict
        self.matrix_dict = matrix_dict
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

        u_in_matrix = self.matrix_dict[idx]

        if self.target_column is not None:
            tar_data = torch.tensor(
                np.array(data[self.target_column]), dtype=torch.long
            )
            return {
                "num": num_data,
                "cat": cat_data,
                "target": tar_data,
                "u_out": u_out,
                "u_in_matrix": u_in_matrix,
            }
        else:
            return {
                "num": num_data,
                "cat": cat_data,
                "u_out": u_out,
                "u_in_matrix": u_in_matrix,
            }


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
