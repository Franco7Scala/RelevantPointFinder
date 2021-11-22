import torch

from src.neural_network.abstract_neural_network import AbstractNeuralNetwork
from src.neural_network.convolutional_neural_network import ConvolutionalNeuralNetwork
from src.neural_network.lstm import LSTM


class CombinedNeuralNetwork(AbstractNeuralNetwork):

    def __init__(self, device):
        super(CombinedNeuralNetwork, self).__init__(device)
        self.lstm = LSTM(device)
        self.cnn = ConvolutionalNeuralNetwork(device)
        self.to(device)

    def forward(self, x):
        x = self.cnn(x)
        x = self.lstm(x)
        return x

    def save_lstm(self, path):
        torch.save(self.lstm.state_dict(), path)

    def save_cnn(self, path):
        torch.save(self.cnn.state_dict(), path)

    def load_lstm(self, path):
        self.lstm = torch.load(path, map_location=torch.device("cpu"))
        return self.lstm

    def load_cnn(self, path):
        self.cnn = torch.load(path, map_location=torch.device("cpu"))
        return self.cnn
