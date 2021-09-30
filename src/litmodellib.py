import pytorch_lightning as pl
import modellib
from torch import nn
from torch import optim
from torchmetrics.regression import MeanAbsoluteError
import torch


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
        self.train_metric = MeanAbsoluteError(compute_on_step=True)
        self.val_metric = MeanAbsoluteError(compute_on_step=False)

    def training_step(self, batch, batch_idx):
        y = batch["target"].view(-1)
        preds = self.model(batch).view(-1)
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
