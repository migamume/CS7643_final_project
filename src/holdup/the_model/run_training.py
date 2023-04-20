import pickle

from holdup.the_model.autoencoder import Autoencoder
import numpy as np
# import pandas as pd
import torch
from torch import nn
import torch.optim as optim
# from sklearn.model_selection import train_test_split
import random
import os
from holdup.parser.replayable_hand import ReplayableHand, Streets
import functools
from typing import Tuple, List
import matplotlib.pyplot as plt
from get_datasets import *
import pandas as pd
from sklearn.model_selection import train_test_split

preflop = "preflop"
flop = "flop"
turn = "turn"
river = "river"

def get_stage(dataset, stage):
    if stage == preflop:
        return dataset[0]
    if stage == flop:
        return dataset[1]
    if stage == turn:
        return dataset[2]
    if stage == river:
        return dataset[3]

def flatten_streets(dataset):
    streets = [[], [], [], []]
    for logfile in dataset:
        for index, street in enumerate(logfile):
            streets[index] = streets[index] + street
    return streets


def get_data(dataset, stage):
    flattened_data = flatten_streets(dataset)
    stage_data = get_stage(flattened_data, stage)
    return [(x[0], x[1][1]) for x in stage_data]


with open('last_possible.pickle', 'rb') as last_possible_pickle:
    last_possible_dataset = pickle.load(last_possible_pickle)

preflop_data = get_data(last_possible_dataset, "preflop")
flop_data = get_data(last_possible_dataset, "flop")
turn_data = get_data(last_possible_dataset, "turn")
river_data = get_data(last_possible_dataset, "river")

def separate_train_test(street_data):
    n_train = int(len(street_data)*0.6)
    train_set = street_data[:n_train]
    test_set = street_data[n_train:]
    return train_set,test_set

train_preflop, test_preflop =separate_train_test(preflop_data)
print("preflop_train_data_size: {}".format(len(train_preflop)))
print("preflop_test_data_size: {}".format(len(test_preflop)))

train_flop, test_flop=separate_train_test(flop_data)
print("flop_train_data_size: {}".format(len(train_flop)))
print("flop_test_data_size: {}".format(len(test_flop)))

train_turn, test_turn=separate_train_test(turn_data)
print("turn_train_data_size: {}".format(len(train_turn)))
print("turn_test_data_size: {}".format(len(test_turn)))

train_river, test_river=separate_train_test(river_data)
print("river_train_data_size: {}".format(len(train_river)))
print("river_test_data_size: {}".format(len(test_river)))


# Set the device to use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create an instance of the autoencoder
# model = Autoencoder(num_hidden_nodes).to(device)

def train(model, train_loader, num_epochs, weight_decay, pftr='stage_name'):
    criterion = nn.CrossEntropyLoss() #changed to cross entropy loss for classification based tasks (semi-supervised)
    optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay)
    train_losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for data in train_loader:
            inputs, labels = data
            inputs = inputs.float()
            optimizer.zero_grad()
            batch_size, _, _ = inputs.size()
            inputs = inputs.view(batch_size, -1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    # Autoencoder.save(save_the_model, bbox_inches='tight')
    print("Training finished!")
    # plot the learning curve
    plt.plot(range(1, num_epochs + 1), train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    s = str(pftr) + '_lc_epoch_loss'
    plt.savefig(s)
    plt.clf()

def quick_test(model,test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.float()
            batch_size, _, _ = inputs.size()
            inputs = inputs.view(batch_size, -1)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy on test set: {correct / total * 100:.2f}%")


def train_and_quick_test(num_hidden_nodes, num_epochs, weight_decay,train_data,test_data, pftr):
    # Define the model
    model = Autoencoder(num_hidden_nodes).to(device)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=20, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=20, shuffle=False)
    train(model, train_loader, num_epochs, weight_decay,pftr)
    # Test the model
    quick_test(model,test_loader)


# Define hyperparameters for each stage
'''For flop stage prediction:
Input: 400 (data matrix vectorized)
Hidden: 20 - hidden level for auto-encoder (this will be the input for the supervised learning model)
Output: 400 (same size as input)
Epochs: 20 for softmax
Loss: MSE function -> L2 norm between input and output plus regularization term
-> ||input - output||**2(base 2) + lambda||w||**2 (base 2)
Optimizer: SCG - used for faster time (maybe we can use SGD?)
'''
flop_hidden_nodes = 20
flop_epochs = 20
flop_weight_decay = 0.001

'''For turn and rivers stage prediction:
Input: 400 (data matrix vectorized) 
Hidden: 40 - hidden level for auto-encoder (this will be the input for the supervised learning model) 
Output: 400 (same size as input) 
Epochs: 40 for softmax
Loss: MSE function -> L2 norm between input and output plus regularization term 
-> ||input - output||**2(base 2) + lambda||w||**2 (base 2) 
Optimizer: SCG - used for faster time (maybe we can use SGD?) '''

turn_hidden_nodes = 40
turn_epochs = 40
turn_weight_decay = 0.001

river_hidden_nodes = 40
river_epochs = 40
river_weight_decay = 0.001

# Define train_data and test_data for each stage
# train_data and test_data should be torch.utils.data.TensorDataset objects



if __name__ == "__main__":
    #the authors didn't visualize results from preflop, so the data from this would be new
    train_and_quick_test(flop_hidden_nodes, flop_epochs, flop_weight_decay,train_preflop,test_preflop,pftr='preflop_last_possible')

    #train/test models on flop, turn, river datasets
    train_and_quick_test(flop_hidden_nodes, flop_epochs, flop_weight_decay,train_flop,test_flop,pftr='flop_last_possible')
    train_and_quick_test(turn_hidden_nodes, turn_epochs, turn_weight_decay,train_turn,test_turn,pftr='turn_last_possible')
    train_and_quick_test(river_hidden_nodes, river_epochs, river_weight_decay,train_river,test_river,pftr='river_last_possible')