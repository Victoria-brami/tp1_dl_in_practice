import sys
sys.path.append("/c/Users/Victoria/Documents/Victoria/enpc/3A/MVA/S2/deep_learning_in_practice/tp1_dl_in_practice/pb1")

import ray
import logging
import optuna
from optuna_trainer import *

def main():
    if optuna_config.pruner == 'Hyperband':
        print('Hyperband pruner')
        pruner = optuna.pruners.HyperbandPruner(max_resource=optuna_config.n_iters,
                                                reduction_factor=optuna_config.reduction_factor)
    elif optuna_config.pruner == 'Median':
        print('Median pruner')
        pruner = optuna.pruners.MedianPruner()
    else:
        print('No pruner (or invalid pruner name)')
        pruner = optuna.pruners.NopPruner()

    # initialise the multiprocessing handler
    ray.init(num_cpus=2, num_gpus=exec_config.gpus, logging_level=logging.CRITICAL,
             ignore_reinit_error=True)

    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=optuna_config.n_trials, timeout=optuna_config.timeout,
                   callbacks=[dump_study_callback], n_jobs=optuna_config.n_jobs)

    # displays a study summary
    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # dumps the study for use with dash_study.py
    joblib.dump(study, 'study.pkl')


if __name__ == "__main__":
    main()