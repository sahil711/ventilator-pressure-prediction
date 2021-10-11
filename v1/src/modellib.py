from torch import nn
from model import LinBnReLu, LSTMUnit, Conv1DBnRelu, InitRNNWeights
import torch
import model


class LSTMModelV2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        emb_layers = []
        for k in self.config.embedding_layer:
            emb_layers.append(nn.Embedding(**self.config.embedding_layer[k]))
        self.embedding = nn.Sequential(*emb_layers)

        self.conv = nn.ModuleList(
            [
                Conv1DBnRelu(28, 32, 3, padding=1),
                Conv1DBnRelu(28, 32, 5, padding=2),
                Conv1DBnRelu(28, 32, 1, padding=0),
            ]
        )
        self.rnn1 = LSTMUnit(
            input_size=512,
            hidden_size=512,
            bidirectional=True,
            dropout=0,
            batch_first=True,
            output_size=512,
            num_layers=1,
        )
        self.rnn2 = LSTMUnit(
            input_size=512,
            hidden_size=512,
            bidirectional=True,
            dropout=0,
            batch_first=True,
            output_size=512,
            num_layers=1,
        )
        self.rnn3 = LSTMUnit(
            input_size=512,
            hidden_size=512,
            bidirectional=True,
            dropout=0,
            batch_first=True,
            output_size=512,
            num_layers=1,
        )
        self.linear_proj = nn.Linear(192, 512)
        self.fc = nn.Sequential(
            *[
                LinBnReLu(in_dim=512, out_dim=1024, is_bn=False, dropout=0),
                LinBnReLu(in_dim=1024, out_dim=1, is_bn=False, dropout=0),
            ]
        )
        wt_init = InitRNNWeights(init_type="xavier")
        wt_init(self.rnn1)
        wt_init(self.rnn2)
        wt_init(self.rnn3)

    def forward(self, x):
        x_cat, x_num = x["cat"], x["num"]
        x_cat = torch.cat(
            [self.embedding[i](x_cat[:, :, i]) for i in range(len(self.embedding))],
            dim=-1,
        )
        feats = []
        for mod in self.conv:
            feats.append(mod(x_num.permute(0, 2, 1)))
        feats = torch.cat(feats, dim=1).permute(0, 2, 1)
        x = torch.cat([feats, x_cat], dim=-1)
        x = self.linear_proj(x)
        x_res = x
        x = self.rnn1(x)
        x = x + x_res
        x_res = x
        x = self.rnn2(x)
        x = x + x_res
        x_res = x
        x = self.rnn3(x)
        x = self.fc(x)
        return x.squeeze(-1)


class LSTMModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        emb_layers = []
        for k in self.config.embedding_layer:
            emb_layers.append(nn.Embedding(**self.config.embedding_layer[k]))
        self.embedding = nn.Sequential(*emb_layers)
        self.rnn = getattr(nn, self.config.rnn_layer["class"])(
            **self.config.rnn_layer["kwargs"]
        )
        self.fc = nn.Sequential(
            *[
                LinBnReLu(
                    in_dim=2 * self.rnn.hidden_size,
                    out_dim=512,
                    is_bn=False,
                    dropout=0.1,
                ),
                LinBnReLu(in_dim=512, out_dim=1, is_bn=False, dropout=0),
            ]
        )
        wt_init = getattr(model, self.config.rnn_init["class"])(
            **self.config.rnn_init["kwargs"]
        )
        print(wt_init.__dict__)
        print(torch.tensor([x.sum() for name, x in self.rnn.named_parameters()]).sum())
        wt_init(self.rnn)
        print(torch.tensor([x.sum() for name, x in self.rnn.named_parameters()]).sum())

    def forward(self, x):
        x_cat, x_num = x["cat"], x["num"]
        x_cat = torch.cat(
            [self.embedding[i](x_cat[:, :, i]) for i in range(len(self.embedding))],
            dim=-1,
        )
        x = torch.cat([x_num, x_cat], dim=-1)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x.squeeze(-1)


class LSTMModelClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        emb_layers = []
        for k in self.config.embedding_layer:
            emb_layers.append(nn.Embedding(**self.config.embedding_layer[k]))
        self.embedding = nn.Sequential(*emb_layers)
        self.rnn = getattr(nn, self.config.rnn_layer["class"])(
            **self.config.rnn_layer["kwargs"]
        )
        self.fc = nn.Sequential(
            *[
                LinBnReLu(
                    in_dim=2 * self.rnn.hidden_size,
                    out_dim=512,
                    is_bn=False,
                    dropout=0.1,
                ),
                LinBnReLu(
                    in_dim=512, out_dim=self.config.output_dim, is_bn=False, dropout=0
                ),
            ]
        )
        wt_init = getattr(model, self.config.rnn_init["class"])(
            **self.config.rnn_init["kwargs"]
        )
        print(wt_init.__dict__)
        print(torch.tensor([x.sum() for name, x in self.rnn.named_parameters()]).sum())
        wt_init(self.rnn)
        print(torch.tensor([x.sum() for name, x in self.rnn.named_parameters()]).sum())

    def forward(self, x):
        x_cat, x_num = x["cat"], x["num"]
        x_cat = torch.cat(
            [self.embedding[i](x_cat[:, :, i]) for i in range(len(self.embedding))],
            dim=-1,
        )
        x = torch.cat([x_num, x_cat], dim=-1)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x


class CNNLSTMModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        emb_layers = []
        for k in self.config.embedding_layer:
            emb_layers.append(nn.Embedding(**self.config.embedding_layer[k]))
        self.embedding = nn.Sequential(*emb_layers)

        layers = []
        for k in self.config.feature_extractor:
            layers.append(
                getattr(model, config.feature_extractor[k]["class"])(
                    **config.feature_extractor[k]["kwargs"]
                )
            )
        self.cnn = nn.Sequential(*layers)

        self.rnn = getattr(nn, self.config.rnn_layer["class"])(
            **self.config.rnn_layer["kwargs"]
        )
        self.fc = nn.Sequential(
            *[
                LinBnReLu(
                    in_dim=2 * self.rnn.hidden_size, out_dim=512, is_bn=False, dropout=0
                ),
                LinBnReLu(in_dim=512, out_dim=1, is_bn=False, dropout=0),
            ]
        )

    def forward(self, x):
        x_cat, x_num = x["cat"], x["num"]
        x_cat = torch.cat(
            [self.embedding[i](x_cat[:, :, i]) for i in range(len(self.embedding))],
            dim=-1,
        )
        x = torch.cat([x_num, x_cat], dim=-1)
        x = self.cnn(x.permute(0, 2, 1)).permute(0, 2, 1)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x.squeeze(-1)
