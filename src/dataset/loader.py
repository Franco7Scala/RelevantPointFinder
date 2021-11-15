import torch
import numpy

from arff2pandas import a2p
from pandas.tests.groupby.test_value_counts import df


def get_dataset():
    with open("/raw_data/ECG5000/ECG5000_TRAIN.arff") as f:
        train = a2p.load(f)

    with open("/raw_data/ECG5000/ECG5000_TEST.arff") as f:
        test = a2p.load(f)

    return train, test


def get_classes():
    return ["Normal", "R on T", "PVC", "SP", "UB"]
