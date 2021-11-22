import torch
import torch.nn as nn

from torch.optim.adam import Adam
from src.neural_network.abstract_neural_network import AbstractNeuralNetwork
from src.neural_network.layers.conv_2d_block import Conv2dBlock
from src.neural_network.layers.lambda_layer import LambdaLayer
from src.neural_network.layers.view_layer import View


class ConvolutionalNeuralNetwork(AbstractNeuralNetwork):

    def __init__(self, device):
        super(ConvolutionalNeuralNetwork, self).__init__(device)
        # building nn
        filters_start = 32
        layer_filters = filters_start
        filters_growth = 32
        strides_start = 1
        strides_end = 2
        depth = 4
        n_blocks = 6
        n_channels = 1
        input_shape = (n_channels, 33, 570)
        layers = []
        for block in range(n_blocks):
            if block == 0:
                provide_input = True
            else:
                provide_input = False

            layers.append(Conv2dBlock(depth, layer_filters, filters_growth, strides_start, strides_end, input_shape, first_layer=provide_input))
            layer_filters += filters_growth

        layers.append(View((-1, 9, 224)))
        layers.append(LambdaLayer(lambda x: torch.mean(x, axis=1)))
        layers.append(nn.Linear(224, 1))
        self.net = nn.Sequential(*layers)
        self.to(device)

    def forward(self, x):
        x = self.net(x)
        x = torch.sigmoid(x)
        return x

    def fit(self, train_dataloader, test_dataloader):
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.parameters(), lr=0.001)
        train_loss, val_loss = [], []
        train_acc, val_acc = [], []
        for epoch in range(500):
            train_running_loss = 0.0
            val_running_loss = 0.0
            train_correct = 0
            val_correct = 0
            # train
            self.train()
            for i, data in enumerate(train_dataloader):
                inputs = data['sx'].to(self.device)
                labels = data['label'].to(self.device).long()
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_running_loss += loss.item() * outputs.shape[0]
                _, predicted = torch.max(outputs, 1)
                train_correct += torch.sum(predicted == labels.data)

            train_loss.append(train_running_loss / len(train_dataloader.dataset.indices))
            train_acc.append(train_correct.float().item() / len(train_dataloader.dataset.indices))
            # evaluate
            with torch.no_grad():
                self.eval()
                for i, data in enumerate(test_dataloader):
                    inputs = data['sx'].to(self.device)
                    labels = data['label'].to(self.device).long()
                    outputs = self(inputs)
                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item() * outputs.shape[0]
                    _, predicted = torch.max(outputs, 1)
                    val_correct += torch.sum(predicted == labels.data)

                val_loss.append(val_running_loss / len(test_dataloader.dataset.indices))
                val_acc.append(val_correct.float().item() / len(test_dataloader.dataset.indices))
