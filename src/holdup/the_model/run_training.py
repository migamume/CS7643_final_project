from holdup.the_model.autoencoder import Autoencoder
import numpy as np
# import pandas as pd
import torch
from torch import nn
import torch.optim as optim
# from sklearn.model_selection import train_test_split
import random
import os
'''Data prep
Reconstruct 20x20 matrix
np or torch.reshape((20,20))
Look at this line: https://github.com/migamume/CS7643_final_project/blob/main/src/holdup/parser/data_prepper.py#L95

Split dataset -> 60 train, 40 test'''

# Set up paths and filenames
data_dir = 'data_dir/'
# Set up train-test split
num_train = 60000
num_test = 40000
train_data = []
test_data = []

# Iterate over CSV files
csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
random.shuffle(csv_files)
for i, csv_file in enumerate(csv_files):
    # Load CSV file
    data = np.loadtxt(os.path.join(data_dir, csv_file), delimiter=',')

    # Reshape matrix to 20x20
    data = np.reshape(data, (20, 20))

    # Decide whether to use file for training or testing
    if i < num_train:
        train_data.append(data)
    else:
        test_data.append(data)

# Convert data to PyTorch tensors
train_tensor = torch.tensor(train_data)
test_tensor = torch.tensor(test_data)

# Define constants for indexing matrix
FLOP_ROWS = [1, 5]
TURN_ROWS = [2, 6]
RIVER_ROWS = [3, 7]
BEHAVIOR_COLS = [12, 13, 14, 15]

# Define function to extract specific stages from tensor
def extract_stages(tensor, rows):
    return tensor[rows[0]:rows[1]+1, :]

# Define function to extract specific behaviors from tensor
def extract_behaviors(tensor, cols):
    return tensor[:, cols]

# Create empty tensors for training and testing data
num_train = train_tensor.shape[0]
num_test = test_tensor.shape[0]

train_flop = torch.zeros((num_train, 4, len(BEHAVIOR_COLS)))
train_turn = torch.zeros((num_train, 4, len(BEHAVIOR_COLS)))
train_river = torch.zeros((num_train, 4, len(BEHAVIOR_COLS)))
test_flop = torch.zeros((num_test, 4, len(BEHAVIOR_COLS)))
test_turn = torch.zeros((num_test, 4, len(BEHAVIOR_COLS)))
test_river = torch.zeros((num_test, 4, len(BEHAVIOR_COLS)))

# Iterate over train data
for i in range(num_train):
    # Load tensor
    tensor = train_tensor[i]

    # Extract flop, turn, and river stages and specific behaviors
    train_flop[i] = extract_behaviors(extract_stages(tensor, FLOP_ROWS), BEHAVIOR_COLS)
    train_turn[i] = extract_behaviors(extract_stages(tensor, TURN_ROWS), BEHAVIOR_COLS)
    train_river[i] = extract_behaviors(extract_stages(tensor, RIVER_ROWS), BEHAVIOR_COLS)

# Iterate over test data
for i in range(num_test):
    # Load tensor
    tensor = test_tensor[i]

    # Extract flop, turn, and river stages and specific behaviors
    test_flop[i] = extract_behaviors(extract_stages(tensor, FLOP_ROWS), BEHAVIOR_COLS)
    test_turn[i] = extract_behaviors(extract_stages(tensor, TURN_ROWS), BEHAVIOR_COLS)
    test_river[i] = extract_behaviors(extract_stages(tensor, RIVER_ROWS), BEHAVIOR_COLS)



# Set the device to use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create an instance of the autoencoder
model = Autoencoder().to(device)

def train(model, train_loader, num_epochs, weight_decay):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay)
    for epoch in range(num_epochs):
        running_loss = 0.0
        for data in train_loader:
            inputs, _ = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    Autoencoder.save('autoencoder_model.h5')
    print("Training finished!")

def quick_test(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy on test set: {correct / total * 100:.2f}%")


def train_and_quick_test(num_hidden_nodes, num_epochs, weight_decay,train_data,test_data):
    # Define the model
    model = Autoencoder(num_hidden_nodes)
    # Load data
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)
    # Train the model
    train(model, train_loader, num_epochs, weight_decay)
    # Test the model
    quick_test(model, test_loader)

# Define hyperparameters for each stage
'''For flop stage prediction:
Input: 400 (data matrix vectorized)
Hidden: 20 - hidden level for auto-encoder (this will be the input for the supervised learning model)
Output: 400 (same size as input)
Epochs: 10
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
Epochs: 30 
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

# Train the model on Flop stage
if __name__ == "__main__":
    train_and_quick_test(flop_hidden_nodes, flop_epochs, flop_weight_decay,train_flop,test_flop)

# Train the model on Turn stage
if __name__ == "__main__":
    train_and_quick_test(turn_hidden_nodes, turn_epochs, turn_weight_decay,train_turn,test_turn)

# Train the model on River stage
if __name__ == "__main__":
    train_and_quick_test(river_hidden_nodes, river_epochs, river_weight_decay,train_river,test_river)