import os
import torch
import torch.utils.data as data_utils

from arff2pandas import a2p
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import random_split
from src.dataset.ECG_dataset import ECGDataset


def get_dataset():
    with open(os.path.dirname(os.path.realpath(__file__)) + "/raw_data/ECG5000/ECG5000_TRAIN.arff") as f:
        train = a2p.load(f)

    with open(os.path.dirname(os.path.realpath(__file__)) + "/raw_data/ECG5000/ECG5000_TEST.arff") as f:
        test = a2p.load(f)

    return train, test


def get_dataloader(batch_size, train=True):
    dataset = ECGDataset(ref_file='training2017/REFERENCE.csv', data_dir='training2017')
    train_size = int(0.7 * len(dataset))
    train_set, val_set = random_split(dataset, [train_size, len(dataset) - train_size])
    if train:
        return DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)

    else:
        return DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=4)


def get_classes():
    return ["Normal", "R on T", "PVC", "SP", "UB"]
