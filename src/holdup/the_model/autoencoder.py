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
        self.num_hidden_nodes = num_hidden_nodes
        self.encoder = nn.Sequential(
            nn.Linear(400, self.num_hidden_nodes), #Linear function (input size, hidden size)
            nn.Sigmoid() #Apply sigmoid function to linear output
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.num_hidden_nodes, 400),
            nn.Tanh()
        )
        self.classifier = nn.Sequential(nn.Linear(self.num_hidden_nodes, 4))

    def forward(self, x):
        encoded = self.encoder(x) #hidden level for auto-encoder (this will be the input for the supervised learning model)
        decoded = self.decoder(encoded)
        softmax_output = self.classifier(encoded.view(-1, self.num_hidden_nodes))
        return softmax_output






