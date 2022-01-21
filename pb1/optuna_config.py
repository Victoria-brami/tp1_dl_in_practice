import optuna
import numpy as np
import torch.nn as nn


class ExecutionConfig(object):
    """ Configuration of the training loop"""

    def __init__(self):
        self.epochs = 5 # 5
        self.chkp_folder = '../experiment/optuna_training/'
        self.gpus = 1 #1
        self.num_validation_sanity_steps = 0

class NetConfig(object):

    def __init__(self):
        self.suggest_activations = None  #
        self.default_activations = 'relu'

        self.suggest_n_conv_layers = None # [1,4]
        self.default_n_conv_layers = 4

        self.suggest_linear_dim = None
        self.default_linear_dim = 1000


class OptunaConfig(object):
    """ Configuration of the Optuna study: what to optimise and by what means"""

    def __init__(self):
        self.timeout = 24 * 3600  # 2*3600       # seconds, if None is in both limits, use CTRL+C to stop
        self.n_trials = 5  # 500          # will stop whenever the time or number of trials is reached
        # Computations for HyperBand configuration
        #n_iters = int(TrainDatasetConfig().num_data * ExecutionConfig().epochs / TrainDatasetConfig().batch_size)
        #reduction_factor = int(round(np.exp(np.log(n_iters) / 4)))  # for 5 brackets (see Optuna doc)

        self.n_jobs = 1  # number of parallel optimisations
        #self.n_iters = n_iters
        #self.reduction_factor = reduction_factor
        self.pruner = 'Hyperband'  # options: Hyperband, Median, anything else -> no pruner

        self.suggest_optimiser = None  # ['SGD', 'Adam', 'AdamW]default is hardcoded to Adam
        self.default_optimiser = 'Adam'

        self.suggest_learning_rate = None #[1e-5, 1e-2]
        self.default_learning_rate = 0.0001

        self.suggest_weight_decay = None #[0, 1e-5]
        self.default_weight_decay = 0.00

        self.suggest_loss = None
        self.default_loss = nn.BCELoss()

        self.suggest_batch_size = None
        self.default_batch_size = 64