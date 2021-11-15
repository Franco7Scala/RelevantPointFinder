import numpy
import torch
import torch.nn as nn


class AbstractNeuralNetwork(nn.Module):

    def __init__(self, device):
        super(AbstractNeuralNetwork, self).__init__()
        self.optimizer = None
        self.criterion = None
        self.device = device

    def forward(self, x):
        pass

    def _accuracy(self, predictions, labels):
        classes = torch.argmax(predictions, dim=1)
        return torch.mean((classes == labels).float())

    def fit(self, dataloader, epochs):
        for epoch in range(epochs):
            dataiter = iter(dataloader)
            for batch in dataiter:
                self.optimizer.zero_grad()
                output = self(batch["x"])
                loss = self.criterion(output, batch["y"].to(self.device))
                loss.backward()
                # updating nn weights
                self.optimizer.step()

    def get_statistics(self, dataloader):
        losses = []
        accuracies = []
        data_iterator = iter(dataloader)
        for batch in data_iterator:
            self.optimizer.zero_grad()
            output = self(batch["x"])
            losses.append(self.criterion(output, batch["y"].to(self.device)).cpu().data)
            accuracies.append(self._accuracy(output.to("cpu"), batch["y"].to("cpu")))

        return numpy.average(losses), numpy.average(accuracies)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self = torch.load(path, map_location=torch.device('cpu'))
