import pickle

from holdup.the_model.autoencoder import Autoencoder
from holdup.the_model.autoencoder3 import Encoder, Decoder, Classifier
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


with open('last_possible_new_bins.pickle', 'rb') as last_possible_pickle:
    last_possible_dataset = pickle.load(last_possible_pickle)

preflop_data = get_data(last_possible_dataset, "preflop")
flop_data = get_data(last_possible_dataset, "flop")
turn_data = get_data(last_possible_dataset, "turn")
river_data = get_data(last_possible_dataset, "river")

#training set 60%, validation set 20%, testing set 20%
def separate_train_test_val(street_data):
    n_train = int(len(street_data)*0.6)
    n_val = int(len(street_data)*0.2)
    train_set = street_data[:n_train]
    val_set = street_data[n_train:n_train+n_val]
    test_set = street_data[n_train+n_val:]
    return train_set, val_set, test_set

train_preflop, val_preflop, test_preflop =separate_train_test_val(preflop_data)
print("preflop_train_data_size: {}".format(len(train_preflop)))
print("preflop_test_data_size: {}".format(len(test_preflop)))

train_flop,val_flop, test_flop=separate_train_test_val(flop_data)
print("flop_train_data_size: {}".format(len(train_flop)))
print("flop_test_data_size: {}".format(len(test_flop)))

train_turn,val_turn, test_turn=separate_train_test_val(turn_data)
print("turn_train_data_size: {}".format(len(train_turn)))
print("turn_test_data_size: {}".format(len(test_turn)))

train_river,val_river, test_river=separate_train_test_val(river_data)
print("river_train_data_size: {}".format(len(train_river)))
print("river_test_data_size: {}".format(len(test_river)))


# Set the device to use CUDA if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

# Create an instance of the autoencoder
# model = Autoencoder(num_hidden_nodes).to(device)
def train_autoencoder(encoder, decoder, train_loader, num_epochs, learning_rate, weight_decay):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(decoder.parameters(), lr = learning_rate, weight_decay=weight_decay)
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for data in train_loader:
            inputs, labels = data
            inputs = inputs.float()
            optimizer.zero_grad()
            batch_size, _, _ = inputs.size()
            inputs = inputs.view(batch_size, -1)
            outputs = encoder(inputs)
            outputs = decoder(outputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        print(f"Unsupervised Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}")        
    # Autoencoder.save(save_the_model, bbox_inches='tight')
    print("Unsupervised Training finished!")

def train_classifier(encoder, classifier, train_loader, val_loader, num_epochs, learning_rate, weight_decay):
    # Your classifier training code here (use the modified train function)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(encoder.parameters(), weight_decay=weight_decay, lr=learning_rate)
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for data in train_loader:
            inputs, labels = data
            inputs = inputs.float()
            optimizer.zero_grad()
            batch_size, _, _ = inputs.size()
            inputs = inputs.view(batch_size, -1)
            outputs = encoder(inputs)
            outputs = classifier(outputs)
            # print(outputs)
            # print(outputs)
            # print(labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        print(f"Supervised Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}")
        with torch.no_grad():
            running_loss = 0.0
            for data in val_loader:
                inputs, labels = data
                inputs = inputs.float()
                batch_size, _, _ = inputs.size()
                inputs = inputs.view(batch_size, -1)
                outputs = encoder(inputs)
                outputs = classifier(outputs)
                # print(outputs)
                # print(torch.argmax(outputs, dim=1))
                # print(labels)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / len(val_loader.dataset)
            val_losses.append(epoch_loss)
            print(f"Supervised Epoch [{epoch+1}/{num_epochs}], Validation Loss: {epoch_loss:.4f}")
    # Autoencoder.save(save_the_model, bbox_inches='tight')
    print("Supervised Training finished!")
    # plot the learning curve
    # plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    # plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    # plt.title('Training and Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    # s = str(pftr) + '_lc_epoch_loss'
    # plt.savefig(s)
    # plt.clf()
    
def quick_test3(encoder, classifier, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.float()
            batch_size, _, _ = inputs.size()
            inputs = inputs.view(batch_size, -1)
            outputs = encoder(inputs)
            outputs = classifier(outputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total * 100
    print(f"Accuracy on test set: {accuracy:.2f}%")
    return accuracy


# Re. Batch sizes Typically, we should test for [8,16,32,64,128,256,512,1024]

    
def train_and_quick_test3(num_hidden_nodes, num_epochs, weight_decay, train_data, val_data, test_data, batch_size,learning_rate, pftr):
    # Define the models
    encoder = Encoder(num_hidden_nodes).to(device)
    decoder = Decoder(num_hidden_nodes).to(device)
    classifier = Classifier(num_hidden_nodes).to(device)

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Unsupervised pre-training
    train_autoencoder(encoder, decoder, train_loader, num_epochs[0], learning_rate, weight_decay)

    # Supervised training
    train_classifier(encoder, classifier, train_loader, val_loader, num_epochs[1], learning_rate, weight_decay)

    # Test the model
    accuracy = quick_test3(encoder, classifier, test_loader)
    return accuracy



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
flop_epochs = (5, 20)
flop_weight_decay = 0#0.001

'''For turn and rivers stage prediction:
Input: 400 (data matrix vectorized) 
Hidden: 40 - hidden level for auto-encoder (this will be the input for the supervised learning model) 
Output: 400 (same size as input) 
Epochs: 40 for softmax
Loss: MSE function -> L2 norm between input and output plus regularization term 
-> ||input - output||**2(base 2) + lambda||w||**2 (base 2) 
Optimizer: SCG - used for faster time (maybe we can use SGD?) '''

hidden_nodes_values = [20, 40, 60]
num_epochs_values = [(5, 20), (10, 40)]
weight_decay_values = [0, 0.001, 0.01]
batch_size_values = [8, 16, 32, 64, 128]
learning_rate_values = [0.001, 0.01, 0.1]

# turn_hidden_nodes = 40
# turn_epochs = 40
# turn_weight_decay = 0.001

# river_hidden_nodes = 40
# river_epochs = 40
# river_weight_decay = 0.001

# Define train_data and test_data for each stage
# train_data and test_data should be torch.utils.data.TensorDataset objects


def set_random_seeds(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    
if __name__ == "__main__":
    best_parameters = None
    best_accuracy = -1

    for hidden_nodes in hidden_nodes_values:
        for num_epochs in num_epochs_values:
            for weight_decay in weight_decay_values:
                for batch_size in batch_size_values:
                    for learning_rate in learning_rate_values:
                        print(f"Training with: hidden_nodes={hidden_nodes}, num_epochs={num_epochs}, weight_decay={weight_decay}, batch_size={batch_size}, learning_rate={learning_rate}")
                        
                        set_random_seeds()
                        accuracy = train_and_quick_test3(hidden_nodes, num_epochs, weight_decay, train_flop, val_flop, test_flop, batch_size, learning_rate, pftr='flop_last_possible')
                        
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_parameters = (hidden_nodes, num_epochs, weight_decay, batch_size, learning_rate)

    print("Best parameters found:")
    print(f"hidden_nodes={best_parameters[0]}, num_epochs={best_parameters[1]}, weight_decay={best_parameters[2]}, batch_size={best_parameters[3]}, learning_rate={best_parameters[4]}")
    print(f"Best accuracy: {best_accuracy:.2f}%")
    #the authors didn't visualize results from preflop, so the data from this would be new
    # train_and_quick_test3( flop_hidden_nodes, flop_epochs, flop_weight_decay,train_preflop,val_preflop, test_preflop,32,pftr='preflop_last_possible')

    #train/test models on flop, turn, river datasets
    # train_and_quick_test3(flop_hidden_nodes, flop_epochs, flop_weight_decay,train_flop,val_flop,test_flop,64,pftr='flop_last_possible')
    # train_and_quick_test3(turn_hidden_nodes, turn_epochs, turn_weight_decay,train_turn,val_turn,test_turn,32,pftr='turn_last_possible')
    # train_and_quick_test3(river_hidden_nodes, river_epochs, river_weight_decay,train_river,val_river,test_river,32,pftr='river_last_possible')