from torch import nn
import torch

__all__ = ["VentNetV0"]


class LinBnReLu(nn.Module):
    def __init__(self, in_dim, out_dim, is_bn=True, is_act=True, dropout=0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.is_act = is_act
        if self.is_act:
            self.act = nn.ReLU()
        self.is_bn = is_bn
        if self.is_bn:
            self.bn = nn.BatchNorm1d(in_dim)
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        if self.is_bn:
            x = self.bn(x)
        x = self.dropout(x)
        if self.is_act:
            x = self.act(x)
        x = self.lin(x)
        return x


class RNNBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rnn = getattr(nn, self.config["class"])(**self.config["rnn_kwargs"])
        if self.config.is_residual:
            self.residual_proj_layer = LinBnReLu(
                in_dim=self.config.rnn_kwargs.input_size,
                out_dim=self.config.output_size,
                is_act=True,
                is_bn=False,
                dropout=self.config.residual_dropout,
            )

    def forward(self, x):
        x_seq, _ = self.rnn(x)
        if self.config.is_residual:
            x_res = self.residual_proj_layer(x)
            x_seq = x_seq + x_res
        return x_seq


class VentNetV0(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.init_proj_layer = LinBnReLu(
            self.config.input_features,
            self.config.init_proj_dimension,
            is_bn=False,
            is_act=False,
        )
        self.emb_R = nn.Embedding(**self.config.embedding_layers.identity_columns.R)
        self.emb_C = nn.Embedding(**self.config.embedding_layers.identity_columns.C)
        rnn_models = []
        rnn_blocks = [k for k in self.config.rnn_blocks if "block" in k]
        for k in rnn_blocks:
            rnn_models.append(RNNBlock(self.config.rnn_blocks[k]))
        self.rnn_models = nn.Sequential(*rnn_models)

        fc_input_size = (
            self.config.rnn_blocks.final_ouput_size
            + self.config.embedding_layers.identity_columns.final_output_size
        )
        self.fc = LinBnReLu(**self.config.fc_block, in_dim=fc_input_size)

    def forward(self, x):
        x_seq, x_identity = x["seq"], x["identity_data"]
        x_seq = self.init_proj_layer(x_seq)
        x_seq = self.rnn_models(x_seq)
        seq_len = x_seq.size(1)
        x_identity = (
            torch.cat(
                [self.emb_R(x_identity[:, 0]), self.emb_C(x_identity[:, 1])], axis=-1
            )
            .unsqueeze(1)
            .repeat(1, seq_len, 1)
        )
        x = torch.cat([x_seq, x_identity], axis=-1)
        x = self.fc(x)
        return x.squeeze(-1)
