import pytorch_lightning as pl
import modellib
from torch import nn
from torch import optim
from torchmetrics.regression import MeanAbsoluteError
import torch
from transformers import get_linear_schedule_with_warmup
import joblib
import losslib

# import time


def smoothing_regression_loss(criterion, y_pred, y_true):
    loss = criterion(y_pred, y_true)
    step = 0.07030214545121005

    for lag, w in [(1, 8 / 15), (2, 4 / 15), (3, 2 / 15), (4, 1 / 15)]:
        neg_lag_target = y_true - lag * step
        neg_lag_target = neg_lag_target
        neg_lag_loss = criterion(y_pred, neg_lag_target)
        pos_lag_target = y_true + lag * step
        pos_lag_target = pos_lag_target
        pos_lag_loss = criterion(y_pred, pos_lag_target)
        loss += (neg_lag_loss + pos_lag_loss) * w
    return loss


class Model_regression(pl.LightningModule):
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
        self.lr_step_size = kwargs.get("lr_step_size")

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
        elif self.config.schedular.schedular_class == "MultiplicativeLR":
            fun = lambda epoch: self.lr_step_size
            sched = getattr(optim.lr_scheduler, self.config.schedular.schedular_class)(
                optimizer, lr_lambda=fun
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


class Model(pl.LightningModule):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.save_hyperparameters(self.config)
        self.model = getattr(modellib, self.config.model["class"])(
            self.config.model["kwargs"]
        )
        # model_path = "../experiments/LSTM-Regression-v7-long-run/fold_1/model-epoch=97-val_MAE=0.1642-val_loss=0.1620.ckpt"
        # wt_dict = torch.load(model_path)["state_dict"]
        # self.load_state_dict(wt_dict)

        self.criteria = getattr(losslib, self.config.loss["class"])(
            **self.config.loss["kwargs"]
        )
        self.num_train_iter = kwargs.get("num_train_iter")
        self.num_steps = kwargs.get("num_steps")
        self.lr_step_size = kwargs.get("lr_step_size")
        self.train_metric = MeanAbsoluteError(compute_on_step=False)
        self.val_metric = MeanAbsoluteError(compute_on_step=False)

    def training_step(self, batch, batch_idx):
        y = batch["target"].view(-1)
        if self.config.is_dual_head:
            preds1, preds2 = self.model(batch)
            preds1 = preds1.view(-1)
            preds2 = preds2.view(-1)
        else:
            preds1 = self.model(batch).view(-1)
        idx1 = torch.where(batch["u_out"].view(-1) == 0)[0]
        y_trunc1 = y[idx1]
        preds_trunc1 = preds1[idx1]
        if self.config.is_dual_head:
            idx2 = torch.where(batch["u_out"].view(-1) == 1)[0]
            y_trunc2 = y[idx2]
            preds_trunc2 = preds1[idx2]

        if self.config.training_type.loss == "truncated":
            if self.config.is_dual_head:
                if self.config.training_type.is_smoothing:
                    loss1 = smoothing_regression_loss(
                        self.criteria, preds_trunc1, y_trunc1
                    )
                    loss2 = smoothing_regression_loss(
                        self.criteria, preds_trunc2, y_trunc2
                    )
                else:
                    loss1 = self.criteria(preds_trunc1, y_trunc1)
                    loss2 = self.criteria(preds_trunc2, y_trunc2)
                loss = loss1 + loss2
            else:
                if self.config.training_type.is_smoothing:
                    loss = smoothing_regression_loss(
                        self.criteria, preds_trunc1, y_trunc1
                    )
                else:
                    loss = self.criteria(preds_trunc1, y_trunc1)
            self.train_metric(preds_trunc1, y_trunc1)
        else:
            if self.config.is_dual_head:
                if self.config.training_type.is_smoothing:
                    loss1 = smoothing_regression_loss(self.criteria, preds1, y)
                    loss2 = smoothing_regression_loss(self.criteria, preds2, y)
                else:
                    loss1 = self.criteria(preds1, y)
                    loss2 = self.criteria(preds2, y)
                loss = loss1 + loss2
            else:
                if self.config.training_type.is_smoothing:
                    loss = smoothing_regression_loss(self.criteria, preds1, y)
                else:
                    loss = self.criteria(preds1, y)
            self.train_metric(preds1, y)

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
        if self.config.is_dual_head:
            preds, _ = self.model(batch)
            preds = preds.view(-1)
        else:
            preds = self.model(batch).view(-1)

        idx = torch.where(batch["u_out"].view(-1) == 0)[0]
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
        if self.config.is_dual_head:
            preds, _ = self.model(batch)
            preds = preds.view(-1)
        else:
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
        elif self.config.schedular.schedular_class == "MultiplicativeLR":
            fun = lambda epoch: self.lr_step_size
            sched = getattr(optim.lr_scheduler, self.config.schedular.schedular_class)(
                optimizer, lr_lambda=fun
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


def smoothing_loss(criterion, y_pred, y_true):
    loss = criterion(y_pred, y_true)

    for lag, w in [(1, 8 / 15), (2, 4 / 15), (3, 2 / 15), (4, 1 / 15)]:
        neg_lag_target = nn.ReLU()(y_true - lag)
        neg_lag_target = neg_lag_target.long()
        neg_lag_loss = criterion(y_pred, neg_lag_target)
        pos_lag_target = 949 - nn.ReLU()((949 - (y_true + lag)))
        pos_lag_target = pos_lag_target.long()
        pos_lag_loss = criterion(y_pred, pos_lag_target)
        loss += (neg_lag_loss + pos_lag_loss) * w
    return loss


def oridnal_cross_entropy(x, y):
    wts = (
        1 + (y - x.argmax(dim=1)).abs() * 0.0703
    )  # added one so that even when the class is same,there is some loss because of CE
    log_prob = -1.0 * nn.LogSoftmax(dim=1)(x)
    loss = log_prob.gather(1, y.unsqueeze(1))
    loss = loss.squeeze(1)
    loss *= wts
    loss = loss.mean()
    return loss


def smoothing_oridnal_cross_entropy(y_pred, y_true):
    loss = oridnal_cross_entropy(y_pred, y_true)

    for lag, w in [(1, 8 / 15), (2, 4 / 15), (3, 2 / 15), (4, 1 / 15)]:
        neg_lag_target = nn.ReLU()(y_true - lag)
        neg_lag_target = neg_lag_target.long()
        neg_lag_loss = oridnal_cross_entropy(y_pred, neg_lag_target)
        pos_lag_target = 949 - nn.ReLU()((949 - (y_true + lag)))
        pos_lag_target = pos_lag_target.long()
        pos_lag_loss = oridnal_cross_entropy(y_pred, pos_lag_target)
        loss += (neg_lag_loss + pos_lag_loss) * w
    return loss


class ClassifcationModel(pl.LightningModule):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.save_hyperparameters(self.config)
        self.model = getattr(modellib, self.config.model["class"])(
            self.config.model["kwargs"]
        )
        # model_path = "../experiments/LSTMDpRelu-Transformer-concat-skip-classify-smooth-CE-dp-0.4-10-folds/fold_1/model-epoch=98-val_MAE=0.1473-val_loss=0.0000.ckpt"
        # wt_dict = torch.load(model_path)["state_dict"]
        # self.load_state_dict(wt_dict)
        self.criteria = getattr(nn, self.config.loss["class"])()
        self.criteria = getattr(losslib, self.config.loss["class"])(
            **self.config.loss["kwargs"]
        )
        self.num_train_iter = kwargs.get("num_train_iter")
        self.num_steps = kwargs.get("num_steps")
        self.lr_step_size = kwargs.get("lr_step_size")
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
            if self.config.training_type.is_smoothing:
                loss = smoothing_loss(self.criteria, preds_trunc, y_trunc)
            elif self.config.training_type.is_ordinal:
                loss = smoothing_oridnal_cross_entropy(preds_trunc, y_trunc)
            else:
                loss = self.criteria(preds_trunc, y_trunc)
        else:
            if self.config.training_type.is_smoothing:
                loss = smoothing_loss(self.criteria, preds, y)
            elif self.config.training_type.is_ordinal:
                loss = smoothing_oridnal_cross_entropy(preds, y)
            else:
                loss = self.criteria(preds, y)

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
        elif self.config.schedular.schedular_class == "MultiplicativeLR":
            fun = lambda epoch: self.lr_step_size
            sched = getattr(optim.lr_scheduler, self.config.schedular.schedular_class)(
                optimizer, lr_lambda=fun
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


class ClassifcationMultiLabelModel(pl.LightningModule):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.save_hyperparameters(self.config)
        self.model = getattr(modellib, self.config.model["class"])(
            self.config.model["kwargs"]
        )
        self.criteria_pressure = getattr(nn, self.config.loss["class"])()
        self.criteria_rc = getattr(nn, self.config.loss["class"])()
        self.num_train_iter = kwargs.get("num_train_iter")
        self.num_steps = kwargs.get("num_steps")
        self.mapping = joblib.load(kwargs.get("mapping"))
        self.topk = kwargs.get("topk")
        self.train_metric = MeanAbsoluteError(compute_on_step=False)
        self.val_metric = MeanAbsoluteError(compute_on_step=False)

    def training_step(self, batch, batch_idx):
        y_pressure = batch["pressure"].view(-1)
        y_rc = batch["rc"].view(-1)
        preds = self.model(batch)
        preds_pressure = preds["pred_pressure"].view(-1, 950)
        preds_rc = preds["pred_RC"].view(-1, 9)
        if self.config.is_u_out:
            idx = torch.where(batch["u_out"].view(-1) == 0)[0]
        else:
            idx = torch.where(batch["cat"][:, :, 0].view(-1) == 0)[0]
        y_pressure_trunc = y_pressure[idx]
        preds_pressure_trunc = preds_pressure[idx]
        wt_pressure = self.config.loss.weight.pressure
        wt_rc = self.config.loss.weight.rc

        if self.config.training_type.loss == "truncated":
            loss = (
                self.criteria_pressure(preds_pressure_trunc, y_pressure_trunc)
                * wt_pressure
                + self.criteria_rc(preds_rc, y_rc) * wt_rc
            )
        else:
            loss = (
                self.criteria_pressure(preds_pressure, y_pressure) * wt_pressure
                + self.criteria_rc(preds_rc, y_rc) * wt_rc
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
        y_pressure = batch["pressure"].view(-1)
        preds = self.model(batch)
        preds_pressure = preds["pred_pressure"].view(-1, 950)
        if self.config.is_u_out:
            idx = torch.where(batch["u_out"].view(-1) == 0)[0]
        else:
            idx = torch.where(batch["cat"][:, :, 0].view(-1) == 0)[0]
        y_pressure_trunc = y_pressure[idx]
        preds_pressure_trunc = preds_pressure[idx]

        true_y = get_true_val(self.mapping, y_pressure_trunc)
        topk = preds_pressure_trunc.topk(k=self.topk, dim=1).indices
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
