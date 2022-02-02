from typing import Any, Tuple
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import Dataset
import xgboost as xgb



def load_wine_quality(format: str, split: float = 0.8, seed: int = 1234) -> Tuple[Any, Any]:
    path = Path('/home/andreas/projects/automl-concepts/data/winequality-red.csv')
    if not path.exists():
        raise MissingDataSetException('winequality-red.csv not found in data folder')
    data = pd.read_csv(path)
    data_train, data_test = train_test_split(data, train_size=split, random_state=seed)
    if format == 'pandas':
        return data_train, data_test
    label_train = data_train.quality
    label_test = data_test.quality
    data_train.drop(['quality'], axis=1, inplace=True)
    data_test.drop(['quality'], axis=1, inplace=True)
    if format == 'lightgbm':
        return Dataset(data_train, label=label_train), Dataset(data_test, label=label_test, free_raw_data=False)
    raise NotImplementedError(f'Dataset format {format} is not implemented')

def load_higgs_challenge(format: str, split: float = 0.8, seed: int = 1234) -> Tuple[Any, Any]:
    path = Path('/home/andreas/projects/automl-concepts/data/atlas-higgs-challenge-2014-v2.csv')
    if not path.exists():
        raise MissingDataSetException('atlas-higgs-challenge-2014-v2.csv not found in data folder')
    data = pd.read_csv(path)
    data = data.drop(['EventId', 'KaggleSet', 'Weight', 'KaggleWeight'], axis=1)
    missing_vals = data.apply(lambda x: x.loc[x == -999.0].shape[0] / x.shape[0])
    data = data.loc[:, missing_vals.loc[missing_vals <= 0.1].index.to_list()]
    label_encoder = LabelEncoder().fit(data.Label)
    data_train, data_test = train_test_split(data, train_size=split, random_state=seed)
    if format == 'pandas':
        return data_train, data_test
    if format == 'xgboost':
        label_train = label_encoder.transform(data_train.Label)
        label_test = label_encoder.transform(data_test.Label)
        return (
            xgb.DMatrix(data_train.drop('Label', axis=1), label=label_train),
            xgb.DMatrix(data_test.drop('Label', axis=1), label=label_test)
        )
    raise NotImplementedError(f'Dataset format {format} is not implemented')


class MissingDataSetException(Exception):
    pass

if __name__ == '__main__':
    load_higgs_challenge('xgboost')