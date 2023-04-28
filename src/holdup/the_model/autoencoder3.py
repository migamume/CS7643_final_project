#imports
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import random

#build encoder
class Encoder(nn.Module):
    def __init__(self, num_hidden_nodes):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(400, num_hidden_nodes),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, num_hidden_nodes):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(num_hidden_nodes, 400),
            # nn.Tanh()
        )

    def forward(self, x):
        return self.decoder(x)

class Classifier(nn.Module):
    def __init__(self, num_hidden_nodes, num_classes=4):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(num_hidden_nodes, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)






