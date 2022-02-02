from typing import (
    Any, Dict, Optional
)

import hydra
from omegaconf import DictConfig
import xgboost as xgb
from ray import tune
from optuna.trial import Trial
from ray.tune.integration.xgboost import TuneReportCallback
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.mlflow import mlflow_mixin
import mlflow.xgboost

from utils.data_loading import load_higgs_challenge


@mlflow_mixin
def objective(config):
    mlflow.xgboost.autolog()
    data_train, data_val = load_higgs_challenge('xgboost', split=0.75)
    config.pop('mlflow')

    xgb.train(
        config,
        data_train,
        num_boost_round=250,
        early_stopping_rounds=25,
        evals=[(data_val, 'eval')],
        verbose_eval=False,
        callbacks=[TuneReportCallback()]
    )


def optuna_space(trial: Trial) -> Optional[Dict[str, Any]]:
    trial.suggest_loguniform('eta', 1e-2, 1)
    trial.suggest_int('max_depth', 3, 9)
    trial.suggest_uniform('subsample', 0.5, 1.0)
    trial.suggest_int('min_child_weight', 1, 20)
    trial.suggest_categorical('booster', ['dart', 'gbtree'])
    return {
        'objective': 'binary:logistic',
        'eval_metric': ['auc', 'error', 'logloss']
    }


def tune_xgboost(experiment_name: str, mlflow_uri: str, num_trials: int = 10):
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name=experiment_name)

    search_algo = OptunaSearch(
        space=optuna_space,
        metric='eval-logloss',
        mode='min'
    )
    scheduler = ASHAScheduler()
    tune.run(
        objective,
        metric='eval-logloss',
        mode='min',
        search_alg=search_algo,
        scheduler=scheduler,
        num_samples=num_trials,
        config={
            'mlflow': {
                'experiment_name': experiment_name,
                'tracking_uri': mlflow.get_tracking_uri()
            }
        }
    )
    best_model = mlflow.search_runs(order_by=['metrics.auc DESC']).iloc[0]
    mlflow.register_model(best_model.artifact_uri, name='xgboost-demo')


@hydra.main('conf', 'config_xgboost')
def main(cfg: DictConfig) -> None:
    tune_xgboost(cfg.mlflow_conf.experiment_name, cfg.mlflow_conf.tracking_uri, cfg.tune_conf.trials)


if __name__ == '__main__':
    main()
