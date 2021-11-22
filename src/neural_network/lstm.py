from torch import nn
from src.neural_network.abstract_neural_network import AbstractNeuralNetwork


class LSTM(AbstractNeuralNetwork):

    def __init__(self, device):
        super(LSTM, self).__init__(device)
        self.lstm = nn.LSTM(1, 1, 1)
        self.to(device)

    def forward(self, x):
        return self.lstm(x)
