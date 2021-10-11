from torch import nn


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
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        # self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        # x = self.bn(x)
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
        # self.bn2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x_res = x
        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.bn2(x)
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
