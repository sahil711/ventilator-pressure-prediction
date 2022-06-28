from torch import nn
import torch
from torch.nn import TransformerEncoderLayer, LSTM, Conv1d

__all__ = [
    "TransformerEncoderLayer",
    "InitRNNWeights",
    "LSTMDpReLu",
    "LinBnReLu",
    "LSTM",
    "CustomConformer",
    "Conv1d",
    "CustomTransformerEncoderLayer",
]


class CustomConformerV2(nn.Module):
    def __init__(
        self, d_model, num_heads, dim_ff, dropout, kernel_size, num_kernels, apply_cnn
    ):
        super().__init__()
        # self.encoder = nn.TransformerEncoderLayer(
        #     d_model=d_model, nhead=num_heads, dim_feedforward=dim_ff, dropout=dropout
        # )
        self.encoder = CustomTransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=dim_ff, dropout=dropout
        )
        self.apply_cnn = apply_cnn
        if self.apply_cnn:
            self.cnn = nn.Conv1d(
                in_channels=d_model,
                out_channels=num_kernels,
                kernel_size=kernel_size,
                padding=(kernel_size // 2),
            )

    def forward(self, x):
        x = self.encoder(x)
        if self.apply_cnn:
            x_cnn = self.cnn(x.permute(0, 2, 1)).permute(0, 2, 1)
            x = torch.cat([x, x_cnn], dim=-1)
        return x


class CustomConformer(nn.Module):
    def __init__(
        self, in_dim, d_model, num_heads, dim_ff, dropout, kernel_size, num_kernels
    ):
        super().__init__()
        self.proj_layer = nn.Linear(
            in_features=in_dim, out_features=d_model, bias=False
        )
        self.reverse_proj_layer = nn.Linear(
            in_features=d_model, out_features=in_dim, bias=False
        )
        # self.encoder = nn.TransformerEncoderLayer(
        #     d_model=d_model, nhead=num_heads, dim_feedforward=dim_ff, dropout=dropout
        # )
        self.encoder = CustomTransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=dim_ff, dropout=dropout
        )
        self.cnn = nn.Conv1d(
            in_channels=in_dim,
            out_channels=num_kernels,
            kernel_size=kernel_size,
            padding=(kernel_size // 2),
        )

    def forward(self, x):
        x = self.proj_layer(x)
        x = self.encoder(x)
        x = self.reverse_proj_layer(x)
        x_cnn = self.cnn(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = torch.cat([x, x_cnn], dim=-1)
        return x


class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout
        )
        self.linear1 = nn.Linear(in_features=d_model, out_features=dim_feedforward)
        self.linear2 = nn.Linear(in_features=dim_feedforward, out_features=d_model)
        self.act = nn.ReLU(inplace=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dp1 = nn.Dropout(dropout)
        self.dp2 = nn.Dropout(dropout)

    def forward(self, x):
        attn, _ = self.self_attn(x, x, x)
        res = x + attn
        res = self.norm1(res)
        x = self.linear1(attn)
        x = self.act(x)
        x = self.dp1(x)
        x = self.linear2(x)
        x = self.dp2(x)
        x = x + res
        x = self.norm2(x)
        return x


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
    def __init__(
        self, in_channels, out_channels, kernel_size, padding, is_bn=False, bias=True
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
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
            # self.act = Swish()
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


class Swish(nn.Module):
    """
    https://github.com/sooftware/conformer/blob/main/conformer/activation.py
    Swish is a smooth, non-monotonic function that consistently matches or outperforms ReLU on deep networks applied
    to a variety of challenging domains such as Image classification and Machine translation.
    """

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, inputs):
        return inputs * inputs.sigmoid()


class GLU(nn.Module):
    """
    https://github.com/sooftware/conformer/blob/main/conformer/activation.py
    The gating mechanism is called Gated Linear Units (GLU), which was first introduced for natural language processing
    in the paper “Language Modeling with Gated Convolutional Networks”
    """

    def __init__(self, dim):
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, inputs):
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()


class ConformerMHA(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout
        )
        self.norm = nn.LayerNorm(d_model)
        self.dp = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm(x)
        x, _ = self.mha(x, x, x)
        x = self.dp(x)
        return x


class ConformerFFN(nn.Module):
    def __init__(self, in_dim, dim_multiplier, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.linear1 = nn.Linear(
            in_features=in_dim, out_features=in_dim * dim_multiplier
        )
        self.linear2 = nn.Linear(
            in_features=in_dim * dim_multiplier, out_features=in_dim
        )
        self.act = Swish()
        self.dp1 = nn.Dropout(dropout)
        self.dp2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm(x)
        x = self.linear1(x)
        x = self.act(x)
        x = self.dp1(x)
        x = self.linear2(x)
        x = self.dp2(x)
        return x


class ConformerConv(nn.Module):
    def __init__(self, in_channels, proj_factor, kernel_size, dropout):
        super().__init__()
        self.pointconv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels * proj_factor,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )
        self.conv = nn.Conv1d(
            in_channels=in_channels * proj_factor,
            out_channels=in_channels * proj_factor,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )
        self.pointconv2 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels * proj_factor,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )
        self.batchnorm = nn.BatchNorm1d(in_channels * proj_factor)
        self.layernorm = nn.LayerNorm(in_channels)
        self.act = Swish()
        self.dp = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layernorm(x)
        x = self.pointconv1(x.permute(0, 2, 1))
        x = self.act(x)
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.act(x)
        x = self.pointconv2(x)
        x = self.dp(x)
        return x.permute(0, 2, 1)


class ConformerBlock(nn.Module):
    def __init__(
        self, d_model, conv_multiplier, ffn_multiplier, dropout, kernel_size, num_heads
    ):
        super().__init__()
        self.mha = ConformerMHA(d_model, num_heads, dropout)
        self.conv = ConformerConv(
            in_channels=d_model,
            proj_factor=conv_multiplier,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.ffn1 = ConformerFFN(
            in_dim=d_model, dim_multiplier=ffn_multiplier, dropout=dropout
        )
        self.ffn2 = ConformerFFN(
            in_dim=d_model, dim_multiplier=ffn_multiplier, dropout=dropout
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x1 = x + 0.5 * self.ffn1(x)
        x2 = x1 + self.mha(x1)
        x3 = x2 + self.conv(x2)
        output = x3 + 0.5 * self.ffn2(x3)
        output = self.norm(output)
        return output
