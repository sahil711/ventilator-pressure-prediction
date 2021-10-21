from torch import nn
import torch
from torch.nn import TransformerEncoderLayer

__all__ = ["TransformerEncoderLayer", "InitRNNWeights", "LSTMDpReLu", "LinBnReLu"]


class WaveBlock(nn.Module):
    """
    Wavenet implementation
    paper: https://arxiv.org/abs/1609.03499
    reference: https://github.com/vincentherrmann/pytorch-wavenet/blob/master/wavenet_model.py
    blog: https://medium.com/@kion.kim/wavenet-a-network-good-to-know-7caaae735435
    """

    def __init__(
        self,
        kernel_size,
        num_layers,
        input_dim,
        output_dim,
        dilation_dim,
        residual_dim,
        skip_dim,
    ):

        super(WaveBlock, self).__init__()

        self.num_layers = num_layers

        self.start_conv = nn.Conv1d(input_dim, residual_dim, kernel_size=1)

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_scaling_convs = nn.ModuleList()
        self.skip_scaling_convs = nn.ModuleList()

        self.final_conv1 = nn.Conv1d(
            in_channels=skip_dim, out_channels=skip_dim, kernel_size=1
        )
        self.final_conv2 = nn.Conv1d(
            in_channels=skip_dim, out_channels=output_dim, kernel_size=1
        )

        for i in range(num_layers):
            dilation_rate = 2 ** i
            padding_size = int((kernel_size - 1) * dilation_rate / 2)
            self.filter_convs.append(
                nn.Conv1d(
                    in_channels=residual_dim,
                    out_channels=dilation_dim,
                    dilation=dilation_rate,
                    kernel_size=kernel_size,
                    padding=padding_size,
                )
            )

            self.gate_convs.append(
                nn.Conv1d(
                    in_channels=residual_dim,
                    out_channels=dilation_dim,
                    dilation=dilation_rate,
                    kernel_size=kernel_size,
                    padding=padding_size,
                )
            )
            self.residual_scaling_convs.append(
                nn.Conv1d(
                    in_channels=dilation_dim, out_channels=residual_dim, kernel_size=1
                )
            )

            self.skip_scaling_convs.append(
                nn.Conv1d(
                    in_channels=dilation_dim, out_channels=skip_dim, kernel_size=1
                )
            )

    def forward(self, x):

        x = self.start_conv(x)
        for i in range(self.num_layers):
            _res = x
            _filter = self.filter_convs[i](x)
            _gate = self.gate_convs[i](x)
            _val = nn.Tanh()(_filter) * nn.Sigmoid()(_gate)

            x = torch.add(self.residual_scaling_convs[i](_val), _res)

            _skip = self.skip_scaling_convs[i](_val)

            if i == 0:

                output = _skip
            else:
                output = torch.add(_skip, output)

        output = nn.ReLU()(output)

        output = self.final_conv1(output)
        output = nn.ReLU()(output)

        output = self.final_conv2(output)
        return output


class InitRNNWeights:
    def __init__(self, init_type):
        self.init_type = init_type

    def __call__(self, module):
        if self.init_type == "yakama":
            self._yakama_init(module)
        elif self.init_type == "xavier":
            self._xavier_init(module)
        elif self.init_type == "None":
            pass

    def _yakama_init(self, module):
        for name, param in module.named_parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)

    def _xavier_init(self, module):
        for name, param in module.named_parameters():
            if "bias" not in name:
                nn.init.xavier_normal_(param)


class LSTMDpReLu(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        bidirectional,
        dropout,
        batch_first,
        num_layers,
        is_activation=False,
    ):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size,
            hidden_size,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=batch_first,
            num_layers=num_layers,
        )
        self.is_activation = is_activation
        if is_activation:
            self.act = nn.ReLU(True)
        self.dp = nn.Dropout(dropout)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.dp(x)
        if self.is_activation:
            x = self.act(x)
        return x


class LSTMUnit(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        bidirectional,
        dropout,
        batch_first,
        num_layers,
    ):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size,
            hidden_size,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=batch_first,
            num_layers=num_layers,
        )
        lstm_out_size = 2 * hidden_size if bidirectional else hidden_size
        self.fc = nn.Linear(in_features=lstm_out_size, out_features=output_size)
        self.act = nn.ReLU(True)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.fc(x)
        x = self.act(x)
        return x


class Conv1DBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, is_bn=False):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.is_bn = is_bn
        if self.is_bn:
            self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.is_bn:
            x = self.bn(x)
        x = self.act(x)
        return x


class Conv1DBasicBlock(nn.Module):
    # TODO: add se block to this layer
    def __init__(self, in_channels, out_channels, kernel_size, padding, is_bn=False):
        super().__init__()
        self.is_bn = is_bn
        self.conv1 = Conv1DBnRelu(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            is_bn=is_bn,
        )
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        if self.is_bn:
            self.bn2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x_res = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.is_bn:
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
