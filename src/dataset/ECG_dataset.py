import os
import signal
import numpy
import torch
import scipy.io
import pandas

from torch.utils.data import Dataset

#TODO usare generatore da riscrivere la classe

class ECGDataset(Dataset):

    def __init__(self, ref_file, data_dir, max_length=61, freq=300):
        self.ref_file = ref_file
        self.data_dir = data_dir
        self.max_length = max_length
        self.freq = freq
        self.ref_frame = pandas.read_csv(ref_file, names=['mat', 'label'])
        self.ref_frame['label'] = pandas.Categorical(self.ref_frame['label'])
        self.ref_frame['label_code'] = self.ref_frame['label'].cat.codes

    def __len__(self):
        return len(self.ref_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        mat_name = os.path.join(self.data_dir, self.ref_frame.iloc[idx, 0])
        # extend and create spectrogram
        mat_val = self._zero_pad(scipy.io.loadmat(mat_name)['val'][0], length=self.max_length * self.freq)
        sx = self._spectrogram(numpy.expand_dims(mat_val, axis=0))[2]
        # normalize the spectrogram
        sx_norm = (sx - numpy.mean(sx)) / numpy.std(sx)
        sample = {'sx': sx_norm, 'label': self.ref_frame.iloc[idx, 2]}
        return sample

    def _spectrogram(self, data, fs=300, nperseg=64, noverlap=32):
        f, t, Sxx = signal.spectrogram(data, fs=fs, nperseg=nperseg, noverlap=noverlap)
        Sxx = numpy.transpose(Sxx, [0, 2, 1])
        Sxx = numpy.abs(Sxx)
        mask = Sxx > 0
        Sxx[mask] = numpy.log(Sxx[mask])
        return f, t, Sxx

    def _zero_pad(self, data, length):
        extended = numpy.zeros(length)
        siglength = numpy.min([length, data.shape[0]])
        extended[:siglength] = data[:siglength]
        return extended
