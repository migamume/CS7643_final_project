#imports
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import random

#build encoder
class Autoencoder(nn.Module):
    def __init__(self, num_hidden_nodes):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(400, num_hidden_nodes), #Linear function (input size, hidden size)
            nn.Sigmoid() #Apply sigmoid function to linear output
        )
        self.decoder = nn.Sequential(
            nn.Linear(num_hidden_nodes, 400),
            nn.Tanh()
        )
        self.softmax = nn.Linear(num_hidden_nodes, 4)

    def forward(self, x):
        encoded = self.encoder(x) #hidden level for auto-encoder (this will be the input for the supervised learning model)
        decoded = self.decoder(encoded)
        softmax_output = self.softmax(encoded)
        return softmax_output





