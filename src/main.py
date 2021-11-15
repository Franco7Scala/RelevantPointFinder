import torch
import numpy

from arff2pandas import a2p

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("ECG5000_TRAIN.arff") as f:
    train = a2p.load(f)

with open("ECG5000_TEST.arff") as f:
    test = a2p.load(f)

    data_frame = train.append(test)
    data_frame = data_frame.sample(frac=1.0)

    CLASS_NORMAL = 1
    class_names = ["Normal", "R on T", "PVC", "SP", "UB"]

    new_columns = list(data_frame.columns)
    new_columns[-1] = "target"
    data_frame.columns = new_columns

    normal_df = data_frame[data_frame.target == str(CLASS_NORMAL)].drop(labels="target", axis=1)
    anomaly_df = data_frame[data_frame.target != str(CLASS_NORMAL)].drop(labels="target", axis=1)

    train_data_frame, val_data_frame = train_test_split(normal_df, test_size=0.15, random_state=RANDOM_SEED)
    val_data_frame, test_df = train_test_split(val_data_frame, test_size=0.33, random_state=RANDOM_SEED)

def create_dataset(data_frame):
    sequences = data_frame.astype(numpy.float32).to_numpy().tolist()
    dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]
    quantity_sequences, sequences_length, quantity_features = torch.stack(dataset).shape
    return dataset, sequences_length, quantity_features

train_dataset, sequences_length, quantity_features = create_dataset(train_data_frame)
val_dataset, _, _ = create_dataset(val_data_frame)
test_normal_dataset, _, _ = create_dataset(test_df)
test_anomaly_dataset, _, _ = create_dataset(anomaly_df)
