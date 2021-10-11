import pytorch_lightning as pl
import modellib
from torch import nn
from torch import optim
from torchmetrics.regression import MeanAbsoluteError
import torch
from transformers import get_linear_schedule_with_warmup
import joblib

# import time


class Model(pl.LightningModule):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.save_hyperparameters(self.config)
        self.model = getattr(modellib, self.config.model["class"])(
            self.config.model["kwargs"]
        )
        self.criteria = getattr(nn, self.config.loss["class"])()
        self.num_train_iter = kwargs.get("num_train_iter")
        self.num_steps = kwargs.get("num_steps")
        self.train_metric = MeanAbsoluteError(compute_on_step=False)
        self.val_metric = MeanAbsoluteError(compute_on_step=False)

    def training_step(self, batch, batch_idx):
        y = batch["target"].view(-1)
        preds = self.model(batch).view(-1)
        if self.config.is_u_out:
            idx = torch.where(batch["u_out"].view(-1) == 0)[0]
        else:
            idx = torch.where(batch["cat"][:, :, 0].view(-1) == 0)[0]
        y_trunc = y[idx]
        preds_trunc = preds[idx]
        if self.config.training_type.loss == "truncated":
            loss = self.criteria(preds_trunc, y_trunc)
            self.train_metric(preds_trunc, y_trunc)
        else:
            loss = self.criteria(preds, y)
            self.train_metric(preds, y)

        self.log(
            name="train_MAE",
            value=self.train_metric,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            rank_zero_only=True,
            sync_dist=True,
        )

        self.log(
            name="train_loss",
            value=loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            rank_zero_only=True,
        )
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        y = batch["target"].view(-1)
        preds = self.model(batch).view(-1)
        if self.config.is_u_out:
            idx = torch.where(batch["u_out"].view(-1) == 0)[0]
        else:
            idx = torch.where(batch["cat"][:, :, 0].view(-1) == 0)[0]
        y_trunc = y[idx]
        preds_trunc = preds[idx]
        if self.config.training_type.loss == "truncated":
            loss = self.criteria(preds_trunc, y_trunc)
            self.val_metric(preds_trunc, y_trunc)
        else:
            loss = self.criteria(preds, y)
            self.val_metric(preds, y)

        self.log(
            name="val_MAE",
            value=self.val_metric,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            rank_zero_only=True,
            sync_dist=True,
        )

        self.log(
            name="val_loss",
            value=loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            rank_zero_only=True,
        )

    def test_step(self, batch, batch_idx):
        preds = self.model(batch).view(-1)
        return {"preds": preds}

    def test_epoch_end(self, outputs):
        preds = torch.cat([x["preds"] for x in outputs])
        self.write_prediction("preds", preds, filename="prediction.pt")

    def configure_optimizers(self):
        optimizer = getattr(optim, self.config.optimizer.optim_class)(
            **self.config.optimizer.optim_kwargs, params=self.model.parameters()
        )
        if self.config.schedular.schedular_class == "OneCycleLR":
            sched = getattr(optim.lr_scheduler, self.config.schedular.schedular_class)(
                **self.config.schedular.scheduler_kwargs,
                optimizer=optimizer,
                steps_per_epoch=self.num_train_iter
            )
        elif self.config.schedular.schedular_class == "Linear":
            sched = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=0, num_training_steps=self.num_steps
            )
        else:
            sched = getattr(optim.lr_scheduler, self.config.schedular.schedular_class)(
                **self.config.schedular.scheduler_kwargs, optimizer=optimizer
            )

        lr_dict = {
            "scheduler": sched,
            "interval": self.config.schedular.schedular_interval,
            "monitor": "val_MAE",
            "strict": True,
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_dict}


def get_true_val(mapping, tensor):
    return torch.tensor([mapping[x.item()] for x in tensor])


class ClassifcationModel(pl.LightningModule):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.save_hyperparameters(self.config)
        self.model = getattr(modellib, self.config.model["class"])(
            self.config.model["kwargs"]
        )
        self.criteria = getattr(nn, self.config.loss["class"])()
        self.num_train_iter = kwargs.get("num_train_iter")
        self.num_steps = kwargs.get("num_steps")
        self.mapping = joblib.load(kwargs.get("mapping"))
        self.topk = kwargs.get("topk")
        self.train_metric = MeanAbsoluteError(compute_on_step=False)
        self.val_metric = MeanAbsoluteError(compute_on_step=False)

    def training_step(self, batch, batch_idx):
        y = batch["target"].view(-1)
        preds = self.model(batch).view(-1, 950)
        if self.config.is_u_out:
            idx = torch.where(batch["u_out"].view(-1) == 0)[0]
        else:
            idx = torch.where(batch["cat"][:, :, 0].view(-1) == 0)[0]
        y_trunc = y[idx]
        preds_trunc = preds[idx]
        if self.config.training_type.loss == "truncated":
            loss = self.criteria(preds_trunc, y_trunc)
        else:
            loss = self.criteria(preds, y)

        # true_y = get_true_val(self.mapping, y_trunc)
        # topk = preds_trunc.topk(k=self.topk, dim=1).indices
        # # true_preds = torch.tensor(
        # #     [torch.median(get_true_val(self.mapping, x)).item() for x in topk]
        # # )
        # true_preds = (
        #     get_true_val(self.mapping, topk.view(-1))
        #     .view(-1, self.topk)
        #     .median(dim=1)
        #     .values
        # )

        # self.train_metric(true_preds, true_y)

        # self.log(
        #     name="train_MAE",
        #     value=self.train_metric,
        #     prog_bar=True,
        #     on_step=True,
        #     on_epoch=True,
        #     rank_zero_only=True,
        #     sync_dist=True,
        # )

        self.log(
            name="train_loss",
            value=loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            rank_zero_only=True,
        )
        # return {"loss": loss, "target": y_trunc.detach(), "preds": preds_trunc.detach()}
        return {"loss": loss}

    # def training_epoch_end(self, outputs):
    #     st = time.time()
    #     out = self.all_gather(outputs)
    #     y = torch.cat([x["target"].squeeze(0).cpu() for x in out], dim=0)
    #     preds = torch.cat([x["preds"].squeeze(0).cpu() for x in out], dim=0)
    #     true_y = get_true_val(self.mapping, y)
    #     topk = preds.topk(k=self.topk, dim=1).indices
    #     true_preds = (
    #         get_true_val(self.mapping, topk.view(-1))
    #         .view(-1, self.topk)
    #         .median(dim=1)
    #         .values
    #     )
    #     print(time.time() - st, y.shape, preds.shape)
    #     self.train_metric(true_preds, true_y)
    #     self.log(
    #         name="train_MAE",
    #         value=self.train_metric,
    #         prog_bar=True,
    #         on_step=False,
    #         on_epoch=True,
    #         rank_zero_only=True,
    #         sync_dist=True,
    #     )

    def validation_step(self, batch, batch_idx):
        y = batch["target"].view(-1)
        preds = self.model(batch).view(-1, 950)
        if self.config.is_u_out:
            idx = torch.where(batch["u_out"].view(-1) == 0)[0]
        else:
            idx = torch.where(batch["cat"][:, :, 0].view(-1) == 0)[0]
        y_trunc = y[idx]
        preds_trunc = preds[idx]

        true_y = get_true_val(self.mapping, y_trunc)
        topk = preds_trunc.topk(k=self.topk, dim=1).indices
        true_preds = (
            get_true_val(self.mapping, topk.view(-1))
            .view(-1, self.topk)
            .median(dim=1)
            .values
        )
        true_preds = torch.tensor(
            [torch.median(get_true_val(self.mapping, x)).item() for x in topk]
        )
        self.val_metric(true_preds, true_y)

        self.log(
            name="val_MAE",
            value=self.val_metric,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            rank_zero_only=True,
            sync_dist=True,
        )
        # return y_trunc, preds_trunc

    # def validation_epoch_end(self, outputs):
    #     out = self.all_gather(outputs)
    #     print(out)
    #     print([x[0].shape for x in out])
    #     print([x[1].shape for x in out])
    #     y = torch.hstack([x for x in out[0]])
    #     y = torch.cat([x for x in out[0]])
    #     # y = torch.cat([x[0].squeeze(0).cpu() for x in out], dim=0)
    #     # preds = torch.cat([x[1].squeeze(0).cpu() for x in out], dim=0)
    #     # true_y = get_true_val(self.mapping, y)
    #     # topk = preds.topk(k=self.topk, dim=1).indices
    #     # true_preds = (
    #     #     get_true_val(self.mapping, topk.view(-1))
    #     #     .view(-1, self.topk)
    #     #     .median(dim=1)
    #     #     .values
    #     # )
    #     # self.val_metric(true_preds, true_y)
    #     # self.log(
    #     #     name="val_MAE",
    #     #     value=self.val_metric,
    #     #     prog_bar=True,
    #     #     on_step=False,
    #     #     on_epoch=True,
    #     #     rank_zero_only=True,
    #     #     sync_dist=True,
    #     # )

    def test_step(self, batch, batch_idx):
        preds = self.model(batch).view(-1, 950)
        topk = preds.topk(k=self.topk, dim=1).indices
        true_preds = (
            get_true_val(self.mapping, topk.view(-1))
            .view(-1, self.topk)
            .median(dim=1)
            .values
        )
        true_preds = torch.tensor(
            [torch.median(get_true_val(self.mapping, x)).item() for x in topk]
        )

        return {"preds": true_preds}

    def test_epoch_end(self, outputs):
        preds = torch.cat([x["preds"] for x in outputs])
        self.write_prediction("preds", preds, filename="prediction.pt")

    def configure_optimizers(self):
        optimizer = getattr(optim, self.config.optimizer.optim_class)(
            **self.config.optimizer.optim_kwargs, params=self.model.parameters()
        )
        if self.config.schedular.schedular_class == "OneCycleLR":
            sched = getattr(optim.lr_scheduler, self.config.schedular.schedular_class)(
                **self.config.schedular.scheduler_kwargs,
                optimizer=optimizer,
                steps_per_epoch=self.num_train_iter
            )
        elif self.config.schedular.schedular_class == "Linear":
            sched = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=0, num_training_steps=self.num_steps
            )
        else:
            sched = getattr(optim.lr_scheduler, self.config.schedular.schedular_class)(
                **self.config.schedular.scheduler_kwargs, optimizer=optimizer
            )

        lr_dict = {
            "scheduler": sched,
            "interval": self.config.schedular.schedular_interval,
            "monitor": "val_MAE",
            "strict": True,
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_dict}
