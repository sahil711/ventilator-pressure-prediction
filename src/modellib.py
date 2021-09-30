from model import LinBnReLu, RNNBlock
import model
import torch
from torch import nn

__all__ = ["VentNetV0", "VentNetV1"]


class VentNetV1(nn.Module):
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

        layers = []
        for k in self.config.rnn_blocks:
            layers.append(
                getattr(model, self.config.rnn_blocks[k]["class"])(
                    self.config.rnn_blocks[k]["kwargs"]
                )
            )
        self.rnn = nn.Sequential(*layers)

        self.fc = LinBnReLu(**self.config.fc_block)

    def forward(self, x):
        x_cat, x_num = x["cat"], x["num"]
        x_cat = torch.cat(
            [self.embedding[i](x_cat[:, :, i]) for i in range(len(self.embedding))],
            dim=-1,
        )
        x = torch.cat([x_num, x_cat], dim=-1)
        x = self.cnn(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.rnn(x)
        x = self.fc(x)

        return x.squeeze(-1)


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
