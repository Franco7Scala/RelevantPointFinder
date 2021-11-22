from torch import nn


class View(nn.Module):

    def __init__(self, shape):
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)
