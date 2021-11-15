from torch import nn
from src.neural_network.convolutional_neural_network import ConvolutionalNeuralNetwork
from src.neural_network.lstm import LSTM


class CombinedNeuralNetwork(nn.Module):

    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(CombinedNeuralNetwork, self).__init__()
        self.lstm = LSTM(seq_len, n_features, embedding_dim).to(device)
        self.cnn = ConvolutionalNeuralNetwork(seq_len, embedding_dim, n_features).to(device)

    def forward(self, x):
        x = self.lstm(x)
        x = self.cnn(x)
        return x
