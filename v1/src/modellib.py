from torch import nn
from model import LinBnReLu, LSTMUnit, Conv1DBnRelu, InitRNNWeights
import torch
import model


class WaveNetModelClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        emb_layers = []
        for k in self.config.embedding_layer:
            emb_layers.append(nn.Embedding(**self.config.embedding_layer[k]))
        self.embedding = nn.Sequential(*emb_layers)
        self.wavenet = getattr(model, self.config.wave_layer["class"])(
            **self.config.wave_layer["kwargs"]
        )
        self.fc = nn.Sequential(
            *[
                LinBnReLu(
                    in_dim=self.config.wave_layer.kwargs.output_dim,
                    out_dim=512,
                    is_bn=False,
                    dropout=0.1,
                ),
                LinBnReLu(
                    in_dim=512, out_dim=self.config.output_dim, is_bn=False, dropout=0
                ),
            ]
        )

    def forward(self, x):
        x_cat, x_num = x["cat"], x["num"]
        x_cat = torch.cat(
            [self.embedding[i](x_cat[:, :, i]) for i in range(len(self.embedding))],
            dim=-1,
        )
        x = torch.cat([x_num, x_cat], dim=-1)
        x = self.wavenet(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.fc(x)
        return x


class LSTMModelMultiLabelClassification(nn.Module):
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
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.fc_pressure = nn.Sequential(
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
        self.fc_RC = nn.Sequential(
            *[
                LinBnReLu(
                    in_dim=2 * self.rnn.hidden_size,
                    out_dim=512,
                    is_bn=False,
                    dropout=0.1,
                ),
                LinBnReLu(in_dim=512, out_dim=9, is_bn=False, dropout=0),
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
        out_pressure = self.fc_pressure(x)
        x_pool = self.pool(x.permute(0, 2, 1)).squeeze(-1)
        out_rc = self.fc_RC(x_pool)
        return {"pred_pressure": out_pressure, "pred_RC": out_rc}


class LSTMCNNClassfier(nn.Module):
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
        layers = []
        for k in self.config.cnn_layer:
            layers.append(
                getattr(model, self.config.cnn_layer[k]["class"])(
                    **self.config.cnn_layer[k]["kwargs"]
                )
            )
        self.cnn = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            *[
                LinBnReLu(in_dim=512, out_dim=512, is_bn=False, dropout=0.1,),
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
        x = self.cnn(x.permute(0, 2, 1)).permute(0, 2, 1)
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
        x = self.cnn(x.permute(0, 2, 1)).permute(0, 2, 1)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x.squeeze(-1)


class LSTMAttnClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        emb_layers = []
        for k in self.config.embedding_layer:
            emb_layers.append(nn.Embedding(**self.config.embedding_layer[k]))
        self.embedding = nn.Sequential(*emb_layers)

        seq_layers = []
        for k in self.config.rnn_layer:
            seq_layers.append(
                getattr(nn, self.config.rnn_layer[k]["class"])(
                    **self.config.rnn_layer[k]["kwargs"]
                )
            )
        self.rnn = nn.Sequential(*seq_layers)
        self.attn_conv = nn.Sequential(
            *[
                nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1),
                nn.ReLU(),
            ]
        )
        self.fc = nn.Sequential(
            *[
                LinBnReLu(in_dim=2 * 512, out_dim=512, is_bn=False, dropout=0.1,),
                LinBnReLu(
                    in_dim=512, out_dim=self.config.output_dim, is_bn=False, dropout=0
                ),
            ]
        )
        wt_init = getattr(model, self.config.rnn_init["class"])(
            **self.config.rnn_init["kwargs"]
        )
        print(wt_init.__dict__)
        for layer in self.rnn:
            wt_init(layer)

    def forward(self, x):
        x_cat, x_num = x["cat"], x["num"]
        attn_matrix = x["u_in_matrix"]
        x_cat = torch.cat(
            [self.embedding[i](x_cat[:, :, i]) for i in range(len(self.embedding))],
            dim=-1,
        )
        x_in = torch.cat([x_num, x_cat], dim=-1)
        attn = self.attn_conv(attn_matrix).squeeze(1)
        for layer in self.rnn:
            x_in, _ = layer(x_in)
            # wt_x_in = torch.matmul(x_in.permute(0, 2, 1), attn_matrix.float()).permute(
            #     0, 2, 1
            # )
            # x_in = x_in + wt_x_in
        x_in = torch.matmul(x_in.permute(0, 2, 1), attn).permute(0, 2, 1)
        output = self.fc(x_in)
        return output


class LSTMModelClassificationV2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        emb_layers = []
        for k in self.config.embedding_layer:
            emb_layers.append(nn.Embedding(**self.config.embedding_layer[k]))
        self.embedding = nn.Sequential(*emb_layers)

        seq_layers = []
        for k in self.config.rnn_layer:
            seq_layers.append(
                getattr(model, self.config.rnn_layer[k]["class"])(
                    **self.config.rnn_layer[k]["kwargs"]
                )
            )
        self.rnn = nn.Sequential(*seq_layers)

        self.fc = nn.Sequential(
            *[
                LinBnReLu(
                    in_dim=self.config.fc_input_size,
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
        for layer in self.rnn:
            print(layer)
            wt_init(layer)

    def forward(self, x):
        x_cat, x_num = x["cat"], x["num"]
        x_cat = torch.cat(
            [self.embedding[i](x_cat[:, :, i]) for i in range(len(self.embedding))],
            dim=-1,
        )
        x = torch.cat([x_num, x_cat], dim=-1)
        for layer in self.rnn:
            x_res = x
            x = layer(x)
            x = torch.cat([x, x_res], dim=-1)
            x = nn.ReLU()(x)
        x = self.fc(x)
        return x


class LSTMTransformerModelClassificationV2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        emb_layers = []
        for k in self.config.embedding_layer:
            emb_layers.append(nn.Embedding(**self.config.embedding_layer[k]))
        self.embedding = nn.Sequential(*emb_layers)

        seq_layers = []
        for k in self.config.rnn_layer:
            seq_layers.append(
                getattr(model, self.config.rnn_layer[k]["class"])(
                    **self.config.rnn_layer[k]["kwargs"]
                )
            )
        self.rnn = nn.Sequential(*seq_layers)
        self.proj_layer = nn.Linear(self.config.fc_input_size, 512)
        encoder_layer = getattr(model, self.config.transformer_layer["class"])(
            **self.config.transformer_layer["kwargs"]
        )
        self.encoder = nn.TransformerEncoder(
            num_layers=self.config.transformer_layer.num_layers,
            encoder_layer=encoder_layer,
        )
        self.fc = nn.Sequential(
            *[
                LinBnReLu(
                    in_dim=self.config.transformer_layer.kwargs.d_model,
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
        for layer in self.rnn:
            print(layer)
            wt_init(layer)

    def forward(self, x):
        x_cat, x_num = x["cat"], x["num"]
        x_cat = torch.cat(
            [self.embedding[i](x_cat[:, :, i]) for i in range(len(self.embedding))],
            dim=-1,
        )
        x = torch.cat([x_num, x_cat], dim=-1)
        for layer in self.rnn:
            x_res = x
            x = layer(x)
            x = torch.cat([x, x_res], dim=-1)
            x = nn.ReLU()(x)
        x = self.proj_layer(x)
        x = nn.ReLU()(x)
        x = self.encoder(x)
        x = self.fc(x)
        return x


class LSTMModelRegression(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        emb_layers = []
        for k in self.config.embedding_layer:
            emb_layers.append(nn.Embedding(**self.config.embedding_layer[k]))
        self.embedding = nn.Sequential(*emb_layers)

        seq_layers = []
        for k in self.config.rnn_layer:
            seq_layers.append(
                getattr(model, self.config.rnn_layer[k]["class"])(
                    **self.config.rnn_layer[k]["kwargs"]
                )
            )
        self.rnn = nn.Sequential(*seq_layers)

        self.fc1 = nn.Sequential(
            *[
                LinBnReLu(
                    in_dim=self.config.fc_input_size,
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
        for layer in self.rnn:
            print(layer)
            wt_init(layer)

    def forward(self, x):
        x_cat, x_num = x["cat"], x["num"]
        x_cat = torch.cat(
            [self.embedding[i](x_cat[:, :, i]) for i in range(len(self.embedding))],
            dim=-1,
        )
        x = torch.cat([x_num, x_cat], dim=-1)
        for layer in self.rnn:
            x = layer(x)
        x1 = self.fc1(x).squeeze(-1)
        return x1


class LSTMModelDualRegression(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        emb_layers = []
        for k in self.config.embedding_layer:
            emb_layers.append(nn.Embedding(**self.config.embedding_layer[k]))
        self.embedding = nn.Sequential(*emb_layers)

        seq_layers = []
        for k in self.config.rnn_layer:
            seq_layers.append(
                getattr(model, self.config.rnn_layer[k]["class"])(
                    **self.config.rnn_layer[k]["kwargs"]
                )
            )
        self.rnn = nn.Sequential(*seq_layers)

        self.fc1 = nn.Sequential(
            *[
                LinBnReLu(
                    in_dim=self.config.fc_input_size,
                    out_dim=512,
                    is_bn=False,
                    dropout=0.1,
                ),
                LinBnReLu(
                    in_dim=512, out_dim=self.config.output_dim, is_bn=False, dropout=0
                ),
            ]
        )
        self.fc2 = nn.Sequential(
            *[
                LinBnReLu(
                    in_dim=self.config.fc_input_size,
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
        for layer in self.rnn:
            print(layer)
            wt_init(layer)

    def forward(self, x):
        x_cat, x_num = x["cat"], x["num"]
        x_cat = torch.cat(
            [self.embedding[i](x_cat[:, :, i]) for i in range(len(self.embedding))],
            dim=-1,
        )
        x = torch.cat([x_num, x_cat], dim=-1)
        for layer in self.rnn:
            x = layer(x)
        x1 = self.fc1(x).squeeze(-1)
        x2 = self.fc2(x).squeeze(-1)
        return x1, x2


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
