"""
Description: This module defines a simple feedforward neural network model using PyTorch.

Author: Nisal Hemadasa
Date: 18-10-2024
Version: 1.0
"""
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from federated_network.client import DEVICE

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        # Define the layers for MNIST dataset (28 x 28 dimensional images)
        self.dense1 = nn.Linear(in_features=28 * 28, out_features=10)  # Dense layer with 28 * 28 inputs and 10 outputs
        self.relu = nn.ReLU()  # ReLU activation function
        self.dense2 = nn.Linear(10, 1)  # Output layer with 10 inputs and 1 output
        # self.sigmoid = nn.Sigmoid()  # Sigmoid activation function

    def forward(self, x):
        """Forward pass through the network"""
        # Flatten the input to (batch_size, 784)
        x = x.view(x.size(0), -1)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        # x = self.sigmoid(x)
        return x


# for MNIST dataset (28 x 28 dimensional images)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 64 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)


def train(_model: nn.Module, _dataloader: DataLoader, epochs: int, verbose=False) -> None:
    """
    Train the network on the training set.
    :param _model: The model to train
    :param _dataloader: The dataloader containing training dataset
    :param epochs: The number of epochs to train for
    :param verbose: Whether to print training progress
    :return: None
    """
    criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.BCELoss()
    # criterion = torch.nn.CrossEntropyLoss()
    _optimizer = torch.optim.Adam(_model.parameters(), lr=0.001)
    _model.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        # this loop is added because _dataset is dictionary like and torch.from_numpy() expects only Dataloader types.
        # Also takes batches of data from the dataset and trains the model
        for _x, _y in _dataloader:
            inputs = _x.float()  # _x is already a tensor, no need for conversion
            labels = _y.unsqueeze(1).float()  # Ensure labels are in the right format

            inputs = inputs.to(DEVICE)  # move inputs to device
            labels = labels.to(DEVICE)  # move labels to device

            # Clear gradients for each batch
            _optimizer.zero_grad()

            # Forward pass
            outputs = _model(inputs)

            # Calculate loss
            _loss = criterion(outputs, labels)

            # Backward pass and optimization
            _loss.backward()
            _optimizer.step()

            # Metrics
            epoch_loss += _loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

        epoch_loss /= len(_dataloader)
        epoch_acc = correct / total
        if verbose:
            print(f"Epoch {epoch + 1}: train loss {epoch_loss}, accuracy {epoch_acc}")


def test(_model, _dataset) -> Tuple[float, float]:
    """
    Evaluate the network on the entire test set.
    :param _model: The model to evaluate
    :param _dataset: The test dataset
    :return: Tuple of loss and accuracy
    """
    criterion = nn.BCEWithLogitsLoss()
    correct, total, loss = 0, 0, 0.0
    _model.eval()
    with torch.no_grad():
        # this loop is added because _dataset is dictionary like and torch.from_numpy() expects only Dataloader types
        for _x, _y in _dataset:
            inputs = _x.float()  # _x is already a tensor, no need for conversion
            labels = _y.unsqueeze(1).float()  # Ensure labels are in the right format

            # forward pass
            outputs = _model(inputs)

            # compute loss
            loss += criterion(outputs, labels).item()

            # compute accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # average loss over all samples
    loss /= len(_dataset)
    accuracy = correct / total
    return loss, accuracy
