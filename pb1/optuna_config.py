import json
import numpy as np
from model import *

with open("net_configs.json", 'r') as json_file:
    LIST_MODEL_CONFIGS = json.load(json_file)

class ExecutionConfig(object):
    """ Configuration of the training loop"""

    def __init__(self):
        self.epochs = 10 # 5
        self.chkp_folder = '../experiment/optuna_training/'
        self.gpus = 1 #1
        self.num_validation_sanity_steps = 0

class NetConfig(object):

    def __init__(self, args, num_classes=10):

        if args.model:
            self.suggest_model = ["11", "12", "21", "22", "31", "32"]
        else:
            self.suggest_model  = None
        self.default_model = "12"
        if args.activations:
            self.suggest_activations = ['relu', 'prelu', 'tanh', 'sigmoid']
        else:
            self.suggest_activations = None  #
        self.default_activations = 'relu'

        if args.conv_layers_archi:
            self.suggest_conv_layers = LIST_MODEL_CONFIGS['conv_layers']
        else:
            self.suggest_conv_layers = None # [1,4]
        self.default_conv_layers = 0 #[16, "M", 16, "M"]

        if args.linear_layers_archi:
            self.suggest_linear_layers = LIST_MODEL_CONFIGS['linear_layers']
        else:
            self.suggest_linear_layers= None
        self.default_linear_layers = [100, num_classes]

        if args.dropout:
            self.suggest_dropout = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        else:
            self.suggest_dropout= None
        self.default_dropout= 0


class OptunaConfig(object):
    """ Configuration of the Optuna study: what to optimise and by what means"""

    def __init__(self, args):
        self.timeout = 24 * 3600  # 2*3600       # seconds, if None is in both limits, use CTRL+C to stop
        self.n_trials = 20  # 500          # will stop whenever the time or number of trials is reached
        # Computations for HyperBand configuration
        #n_iters = int(TrainDatasetConfig().num_data * ExecutionConfig().epochs / TrainDatasetConfig().batch_size)
        n_iters = 60000 // 64
        reduction_factor = int(round(np.exp(np.log(n_iters) / 4)))
        #reduction_factor = int(round(np.exp(np.log(n_iters) / 4)))  # for 5 brackets (see Optuna doc)

        self.n_jobs = 1  # number of parallel optimisations
        self.n_iters = n_iters
        self.reduction_factor = reduction_factor
        self.pruner = 'Hyperband'  # options: Hyperband, Median, anything else -> no pruner

        if args.optimizer:
            self.suggest_optimizer = ['SGD', 'Adam', 'AdamW', 'RmsProp']
        else:
            self.suggest_optimizer = None  # ['SGD', 'Adam', 'AdamW]default is hardcoded to Adam
        self.default_optimizer = 'Adam'

        if args.lr:
            self.suggest_learning_rate = [1e-5, 1e-1]
        else:
            self.suggest_learning_rate = None #[1e-5, 1e-2]
        self.default_learning_rate = 0.01

        if args.wd:
            self.suggest_weight_decay = [0, 1e-5]
        else:
            self.suggest_weight_decay = None #[0, 1e-5]
        self.default_weight_decay = 0.00

        if args.loss:
            self.suggest_loss = ['bce', 'mse', 'cross_entropy']
        else:
            self.suggest_loss = None
        self.default_loss = 'cross_entropy'

        if args.batch_size:
            self.suggest_batch_size = [4, 16, 32, 64, 128]
        else:
            self.suggest_batch_size = None
        self.default_batch_size = 64