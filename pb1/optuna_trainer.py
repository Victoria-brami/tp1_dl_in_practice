import torch.nn as nn
import pytorch_lightning as pl
from optuna.integration import PyTorchLightningPruningCallback
import torch
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from model import Net, build_config
import torchvision
from arg_parser import parser
import optuna_config as cfg
import joblib


args = parser()


exec_config = cfg.ExecutionConfig()
optuna_config = cfg.OptunaConfig(args=args)
net_config = cfg.NetConfig(args=args)


def dump_study_callback(study, trial):
    joblib.dump(study, 'study.pkl')


class LightningNet(pl.LightningModule):

    def __init__(self, trial):
        super(LightningNet, self).__init__()
        self.trial = trial
        self.configure_batch_size()
        self.configure_model()
        self.configure_loss()
        self.configure_datasets()

    def configure_batch_size(self):
        self.batch_size = optuna_config.default_batch_size

    def configure_model(self):
        # Net architecture
        if net_config.suggest_activations is not None:
            chosen_act = self.trial.suggest_categorical("Activation", net_config.suggest_activations)
        else:
            chosen_act = net_config.default_activations
        if net_config.suggest_activations is not None:
            chosen_conv_layers = self.trial.suggest_categorical("Conv Layers", net_config.suggest_conv_layers)
        else:
            chosen_conv_layers = net_config.default_conv_layers
        if net_config.suggest_linear_layers is not None:
            chosen_linear_layers = self.trial.suggest_categorical("Linear layers", net_config.suggest_linear_layers)
        else:
            chosen_linear_layers = net_config.default_linear_layers
        config = build_config(chosen_act, chosen_conv_layers, chosen_linear_layers)
        self.model = Net(config=config)

    def configure_loss(self):
        # Loss choice
        if optuna_config.suggest_loss is not None:
            chosen_loss = self.trial.suggest_categorical("Loss", optuna_config.suggest_loss)
        else:
            chosen_loss = optuna_config.default_loss
        if chosen_loss.lower() == 'bce':
            self.loss = nn.BCELoss()
        elif chosen_loss.lower() == 'l1':
            self.loss = nn.L1Loss()
        elif chosen_loss.lower() == 'cross_entropy':
            self.loss = nn.CrossEntropyLoss()
        elif chosen_loss.lower() == 'mse':
            self.loss = nn.MSELoss()
        elif chosen_loss.lower() == 'nll':
            self.loss = nn.NLLLoss()

    def configure_datasets(self):
        # Define Train, Val and Test Datasets
        self.dataset = torchvision.datasets.USPS(root=args.data_dir,
                                            train=True,
                                            transform=transforms.ToTensor(),
                                            download=False)
        self.train_set, self.val_set = random_split(self.dataset, [6000, 1291])

        self.test_set = torchvision.datasets.USPS(root=args.data_dir,
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
        if optuna_config.suggest_optimizer is not None:  # choosing optimiser in the given list
            chosen_optimiser = self.trial.suggest_categorical("optimizer", optuna_config.suggest_optimizer)
            if chosen_optimiser == 'Adam':
                return Adam(self.model.parameters(), lr=chosen_lr, weight_decay=chosen_weight_decay)
            elif chosen_optimiser == 'SGD':
                return SGD(self.model.parameters(), lr=chosen_lr, weight_decay=chosen_weight_decay)
        else:  # hard-coded default to Adam
            return AdamW(self.model.parameters(), lr=chosen_lr, weight_decay=chosen_weight_decay)


    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_set, self.batch_size, shuffle=False, num_workers=4)


    def training_step(self, batch, batch_idx):
        image, target = batch
        output = self.model(image)
        batch_size = target.shape[0]
        labels_one_hot = torch.FloatTensor(batch_size, self.model.num_classes)
        labels_one_hot.zero_()
        labels_one_hot.scatter_(1, target.view(-1, 1), 1)
        return self.loss(output, labels_one_hot)

    def validation_step(self, batch, batch_idx):
        image, target = batch
        output = self.model(image)
        pred = output.argmax(dim=1, keepdim=True)
        accuracy = pred.eq(target.view_as(pred)).float().mean()
        return accuracy

    def validation_epoch_end(self, outputs) -> None:
        # since the training step/validation step and test step are run on the IPU device
        # we must log the average loss outside the step functions.
        self.log("val_acc", torch.stack(outputs).mean(), prog_bar=True)

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

    #metrics_callback = MetricsCallback()
    trainer = pl.Trainer(
        logger=True,
        checkpoint_callback=False,
        max_epochs=exec_config.epochs,
        gpus=1 if torch.cuda.is_available() else None,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_acc")]
    )

    trainer.fit(model)

    # Write all the hyper parameters
    #hyperparameters = dict(n_layers=n_layers, , output_dims=output_dims)
    #trainer.logger.log_hyperparams(hyperparameters)

    return trainer.callback_metrics["val_acc"].item()
