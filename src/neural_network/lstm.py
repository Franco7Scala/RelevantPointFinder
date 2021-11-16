from torch import nn
from src.neural_network.abstract_neural_network import AbstractNeuralNetwork


class LSTM(AbstractNeuralNetwork):

    def __init__(self, device):
        super(LSTM, self).__init__(device)
        self.seq_len = 10
        self.n_features = 10
        self.embedding_dim = 64
        # building lstm
        self.embedding_dim, self.hidden_dim = self.embedding_dim, 2 * self.embedding_dim
        self.rnn1 = nn.LSTM(input_size=self.n_features, hidden_size=self.hidden_dim, num_layers=1, batch_first=True)
        self.rnn2 = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.embedding_dim, num_layers=1, batch_first=True)
        self.to(device)

    def forward(self, x):
        x = x.reshape((1, self.seq_len, self.n_features))
        x, (_, _) = self.rnn1(x)
        x, (hidden_n, _) = self.rnn2(x)
        return hidden_n.reshape((self.n_features, self.embedding_dim))
