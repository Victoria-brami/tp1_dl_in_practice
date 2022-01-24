# TP 1: Deep Learning in Practive

## Requirements
Install all the packages needed to automatize hyperparameters tuning:
```
conda create -n [NAME OF CONDA ENV] 
conda activate [NAME OF CONDA ENV] 
pip install -r requirements.txt
```
Then replace the path to the project in ```pb1/optuna_train.py``` line 2.

## Search of best Hyperparameters for Mnist classification

The list of different options for each parameter is defined in ```optuna_config.py``` in variables starting with ```self.suggest_xxxxxx```.

When you choose tuning a given hyperparameter, the training runs for 10 epochs for each tested value.

1. Find the best Loss:
``` python optuna_train.py --data_dir [PATH TO USPS directory] --loss```

2. Find the best optimizer:
``` python optuna_train.py --data_dir [PATH TO USPS directory] --optimizer```

3. Find the best learning rate:
``` python optuna_train.py --data_dir [PATH TO USPS directory] --lr```

4. Find the best activation function in the network:
``` python optuna_train.py --data_dir [PATH TO USPS directory] --activations```

5. Find the best network architecture (last linear layers):
``` python optuna_train.py --data_dir [PATH TO USPS directory] --linear_layers_archi```

6. Find the best network architecture (first convolution layers):
``` python optuna_train.py --data_dir [PATH TO USPS directory] --conv_layers_archi```

You can also make combinations:
7. Find best global architecture:
``` python optuna_train.py --data_dir [PATH TO USPS directory] --conv_layers_archi --linear_layers_archi --activations```

etc.

## Visualization of the optimization process

While running, the script generates a file called ```study.pkl```. You can visualize the best parameters found on a dashboard with:
```
python dash_study.py
```

