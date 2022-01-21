import os

import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Callback
from optuna.integration import PyTorchLightningPruningCallback
import torch
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from pb1.model import ConvModel
import torchvision

import pb1.optuna_config as cfg
import joblib
import optuna

from torch.utils.data.dataset import Dataset
import os



exec_config = cfg.ExecutionConfig()
optuna_config = cfg.OptunaConfig()
net_config = cfg.NetConfig()


def dump_study_callback(study, trial):
    joblib.dump(study, 'study.pkl')


class LightningNet(pl.LightningModule):

    def __init__(self, trial):
        super(LightningNet, self).__init__()
        self.trial = trial
        self.setup_model()
        self.setup_loss()
        self.setup_datasets()

    def setup_model(self):
        # Net architecture
        if net_config.suggest_activations is not None:
            chosen_act = self.trial.suggest_categorical("Activation", net_config.suggest_activations)
        else:
            chosen_act = net_config.default_activations

        if self.n_conv_layers
        self.model = ConvModel(activation=chosen_act)

    def setup_loss(self):
        # Loss choice
        if optuna_config.suggest_loss is not None:
            chosen_loss = self.trial.suggest_categorical("Loss", optuna_config.suggest_loss)
        else:
            chosen_loss = optuna_config.default_loss
        if chosen_loss == 'bce':
            self.loss = nn.BCELoss()
        elif chosen_loss == 'l1':
            self.loss = nn.L1Loss()
        elif chosen_loss == 'cross_entropy':
            self.loss = nn.CrossEntropyLoss()
        elif chosen_loss == 'mse':
            self.loss = nn.MSELoss()
        elif chosen_loss == 'nll':
            self.loss = nn.NLLLoss()

    def setup_datasets(self):
        # Define Train, Val and Test Datasets
        self.dataset = torchvision.datasets.USPS(root='USPS/',
                                            train=True,
                                            transform=transforms.ToTensor(),
                                            download=False)
        self.train_set, self.val_set = random_split(self.dataset, [6000, 1291])

        self.test_set = torchvision.datasets.USPS(root='USPS/',
                                                     train=False,
                                                     transform=transforms.ToTensor(),
                                                     download=False)

    def forward(self, data):
        return self.model.forward(data)

    def configure_optimizers(self):
        # Learning rate suggestion
        if optuna_config.suggest_learning_rate is not None:  # choosing lr in the given interval
            chosen_lr = self.trial.suggest_loguniform('learning-rate',
                                                      optuna_config.suggest_learning_rate[0],
                                                      optuna_config.suggest_learning_rate[1])
        else:
            chosen_lr = optuna_config.default_learning_rate

        # Weight decay suggestion
        if optuna_config.suggest_weight_decay is not None:  # choosing wd in the given interval
            chosen_weight_decay = self.trial.suggest_uniform('weight-decay',
                                                             optuna_config.suggest_weight_decay[0],
                                                             optuna_config.suggest_weight_decay[1])
        else:
            chosen_weight_decay = optuna_config.default_weight_decay

        # Optimiser suggestion
        if optuna_config.suggest_optimiser is not None:  # choosing optimiser in the given list
            chosen_optimiser = self.trial.suggest_categorical("optimizer", optuna_config.suggest_optimiser)
            if chosen_optimiser == 'Adam':
                return Adam(self.model.parameters(), lr=chosen_lr, weight_decay=chosen_weight_decay)
            elif chosen_optimiser == 'SDG':
                return SGD(self.model.parameters(), lr=chosen_lr, weight_decay=chosen_weight_decay)
        else:  # hard-coded default to Adam
            return Adam(self.model.parameters(), lr=chosen_lr, weight_decay=chosen_weight_decay)


    def train_loader(self):
        return DataLoader(self.train_set, self.batch_size, shuffle=True, num_workers=2)

    def val_loader(self):
        return DataLoader(self.val_set, self.batch_size, shuffle=False, num_workers=2)

    def test_loader(self):
        return DataLoader(self.test_set, self.batch_size, shuffle=False, num_workers=2)


    def training_step(self, batch, batch_idx):
        image, target = batch
        output = self.model(image)
        return self.loss(output, target)

    def validation_step(self, batch, batch_idx):
        image, target = batch
        output = self.model(image)
        pred = output.argmax(dim=1, keepdim=True)
        accuracy = pred.eq(target.view_as(pred)).float().mean()
        self.log("val_acc", accuracy)
        self.log("hp_metric", accuracy, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        acc = self.accuracy(logits, y)
        return acc

    def accuracy(self, logits, y):
        # currently IPU poptorch doesn't implicit convert bools to tensor
        # hence we use an explicit calculation for accuracy here. Once fixed in poptorch
        # we can use the accuracy metric.
        acc = torch.sum(torch.eq(torch.argmax(logits, -1), y).to(torch.float32)) / len(y)
        return acc

    def test_epoch_end(self, outputs):
        self.log("test_acc", torch.stack(outputs).mean())



# defines the [hyperparameters] -> objective value mapping for Optuna optimisation
def objective(trial):
    model = LightningNet(trial)  # this initialisation depends on the trial argument

    metrics_callback = MetricsCallback()
    trainer = pl.Trainer(
        logger=True,
        checkpoint_callback=False,
        max_epochs=exec_config.epochs,
        gpus=1 if torch.cuda.is_available() else None,
        callbacks=PyTorchLightningPruningCallback(trial, monitor="val_acc")]
    )

    trainer.fit(model)

    # Write all the hyper parameters
    #hyperparameters = dict(n_layers=n_layers, , output_dims=output_dims)
    #trainer.logger.log_hyperparams(hyperparameters)

    return trainer.callback_metrics["val_acc"].item()
