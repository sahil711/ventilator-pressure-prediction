from torch import nn
import torch

__all__ = ["RNNBlock", "Conv1DBnRelu", "Conv1DBasicBlock", "LinBnReLu", "RNNBlockV2"]


class Conv1DBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class Conv1DBasicBlock(nn.Module):
    # TODO: add se block to this layer
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.conv1 = Conv1DBnRelu(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.conv2 = Conv1DBnRelu(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x_res = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + x_res
        x = self.act2(x)
        return x


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
        self.act = nn.ReLU()

    def forward(self, x):
        x_seq, _ = self.rnn(x)
        if self.config.is_residual:
            x_res = self.residual_proj_layer(x)
            x_seq = x_seq + x_res

        return x_seq


class RNNBlockV2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rnn = getattr(nn, self.config["rnn_class"])(**self.config["rnn_kwargs"])
        self.act = nn.ReLU()

    def forward(self, x):
        x_seq, _ = self.rnn(x)
        if self.config.is_residual:
            x_seq = torch.cat([x_seq, x], dim=-1)
        x_seq = self.act(x_seq)
        return x_seq
