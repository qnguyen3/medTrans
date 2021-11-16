import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from .supervised import models as supervised
import torch
import torch.nn as nn
import torchmetrics

import sys

class GetModel(LightningModule):
    def __init__(self,
                arch: str,
                pretrained: bool,
                learning_rate: float,
                optimizer: str = 'sgd',
                fine_tune: bool = False,
                loss_function: nn.Module = nn.CrossEntropyLoss(),
                num_classes: int = 1000,
                 **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Init model
        if num_classes is None:
            self.representation = True
        else:
            self.representation = False

        self.fine_tune = fine_tune
        self.model = self.init_model(arch=arch, pretrained=pretrained, num_classes=num_classes)
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.loss_function = loss_function

        self.accuracy = torchmetrics.Accuracy()
        self.train_loss = []
        self.validation_loss = []
    
    def forward(self, x):
        if self.representation:
            return self.model.forward_features(x)
        else:
            if self.fine_tune == True:
                with torch.no_grad():
                    x = self.model.forward_features(x)
                return self.model.head(x)
            else:
                x = self.model(x)
                return x

    def init_model(self, arch: str, num_classes: int, pretrained: bool):
        model = supervised.__dict__[arch](pretrained=pretrained, num_classes=num_classes)
        return model

    def configure_optimizers(self, optimizer: str = 'sgd'):
        params = self.parameters()
        if optimizer == 'sgd':
            return torch.optim.SGD(params=params, lr = self.learning_rate)
        elif optimizer == 'adam':
            return torch.optim.Adam(params=params, lr = self.learning_rate)
        elif optimizer == 'adamw':
            return torch.optim.AdamW(params=params, lr = self.learning_rate)
    

    def training_step(self, batch, batch_idx):
        x, y = batch
        predict = self(x)
        loss = self.loss_function(predict, y)
        self.train_loss.append(loss)
        self.log('train_acc_step', self.accuracy(predict, y), prog_bar=True)
        self.log("train_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        predict = self(x)
        loss = self.loss_function(predict, y)
        self.log('val_acc_step', self.accuracy(predict, y), prog_bar=True)
        self.log("val_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        predict = self(x)
        loss = self.loss_function(predict, y)
        self.log("test_loss", loss)
        return loss

    # def validation_epoch_end(self, outputs):
    #     print(outputs)
    #     # avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
    #     # avg_loss = torch.stack([x["val_loss"].item() for x in outputs]).mean()
    #     self.logger.experiment.add_scalar('avg_val_loss',avg_loss, self.current_epoch)

    # def on_train_epoch_end(self):
    #     print('training end: \n')
    #     avg_train_loss = sum(self.train_loss) / len(self.train_loss)
    #     print("average train loss",avg_train_loss)
    #     self.log('avg_train_loss', avg_train_loss, on_step=False, on_epoch=True)
    #     self.train_loss = []

    # def on_validation_epoch_end(self):
    #     print('validation end: \n')
    #     avg_validation_loss = sum(self.validation_loss) / len(self.validation_loss)
    #     print("average validation loss",avg_validation_loss)
    #     self.log('avg_validation_loss', avg_validation_loss, on_step=False, on_epoch=True)
    #     self.validation_loss = []

    