import optuna
import numpy as np
import torch.nn as nn


class ExecutionConfig(object):
    """ Configuration of the training loop"""

    def __init__(self):
        self.epochs = 10 # 5
        self.chkp_folder = '../experiment/optuna_training/'
        self.gpus = 1 #1
        self.num_validation_sanity_steps = 0

class NetConfig(object):

    def __init__(self, num_classes=10):
        self.suggest_activations = None  #
        self.default_activations = 'relu'

        self.suggest_conv_layers = None # [1,4]
        self.default_conv_layers = 0

        self.suggest_linear_layers= None
        self.default_linear_layers = [100, num_classes]


class OptunaConfig(object):
    """ Configuration of the Optuna study: what to optimise and by what means"""

    def __init__(self):
        self.timeout = 24 * 3600  # 2*3600       # seconds, if None is in both limits, use CTRL+C to stop
        self.n_trials = 10  # 500          # will stop whenever the time or number of trials is reached
        # Computations for HyperBand configuration
        #n_iters = int(TrainDatasetConfig().num_data * ExecutionConfig().epochs / TrainDatasetConfig().batch_size)
        n_iters = 60000 // 64
        reduction_factor = int(round(np.exp(np.log(n_iters) / 4)))
        #reduction_factor = int(round(np.exp(np.log(n_iters) / 4)))  # for 5 brackets (see Optuna doc)

        self.n_jobs = 1  # number of parallel optimisations
        self.n_iters = n_iters
        self.reduction_factor = reduction_factor
        self.pruner = 'Hyperband'  # options: Hyperband, Median, anything else -> no pruner

        self.suggest_optimiser = None  # ['SGD', 'Adam', 'AdamW]default is hardcoded to Adam
        self.default_optimiser = 'SGD'

        self.suggest_learning_rate = None #[1e-5, 1e-2]
        self.default_learning_rate = 0.01

        self.suggest_weight_decay = None #[0, 1e-5]
        self.default_weight_decay = 0.00

        self.suggest_loss = ['bce', 'mse', 'l1']
        self.default_loss = 'mse'

        self.suggest_batch_size = None
        self.default_batch_size = 64