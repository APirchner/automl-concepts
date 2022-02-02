from typing import (
    Any, Dict, Optional
)

import hydra
from omegaconf import DictConfig
import lightgbm as lgb
from ray import tune
from optuna.trial import Trial
import mlflow.lightgbm
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune.schedulers.hyperband import HyperBandScheduler
from sklearn.metrics import mean_squared_error
from ray.tune.integration.mlflow import mlflow_mixin

from utils.data_loading import load_wine_quality


@mlflow_mixin
def objective(config):
    mlflow.lightgbm.autolog(log_models=True)
    data_train, data_val = load_wine_quality('lightgbm')  # type: lgb.Dataset
    config.pop('mlflow')

    model = lgb.train(
        config,
        train_set=data_train,
        valid_sets=[data_val],
        valid_names='val',
        verbose_eval=False
    )
    mlflow.lightgbm.log_model(model, 'model')
    preds = model.predict(data_val.get_data())
    val_loss = mean_squared_error(data_val.get_label(), preds)
    tune.report(val_mse=val_loss, done=True)


def optuna_space(trial: Trial) -> Optional[Dict[str, Any]]:
    trial.suggest_loguniform('eta', 1e-2, 1)
    trial.suggest_int('num_leaves', 100, 250)
    trial.suggest_int('num_trees', 100, 250)
    trial.suggest_categorical('boosting_type', ['gbdt', 'dart'])
    return {
        'objective': 'regression',
        'metric': 'mse',
        'verbose': -1,
    }


def tune_lightgbm(experiment_name: str, mlflow_uri: str, num_trials: int = 10):
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name=experiment_name)

    search_algo = OptunaSearch(
        space=optuna_space,
        metric='val_mse',
        mode='min'
    )
    scheduler = HyperBandScheduler()
    results = tune.run(
        objective,
        search_alg=search_algo,
        mode='min',
        scheduler=scheduler,
        num_samples=num_trials,
        config={
            'mlflow': {
                'experiment_name': experiment_name,
                'tracking_uri': mlflow.get_tracking_uri()
            }
        }
    )
    best_model = results.get_best_checkpoint(metric='mse')



@hydra.main('conf', 'config_lightgbm')
def main(cfg: DictConfig) -> None:
    tune_lightgbm(cfg.mlflow_conf.experiment_name, cfg.mlflow_conf.tracking_uri, cfg.tune_conf.trials)


if __name__ == '__main__':
    main()
