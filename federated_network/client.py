"""
Description: This module defines a client of the federated network.

Author: Nisal Hemadasa
Date: 18-10-2024
Version: 1.0
"""
from collections import OrderedDict
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

import constants
from data.dataset_loader import load_datasets
from models.model import train, test, SimpleModel

DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__}"
)


class Client:
    def __init__(self, client_id, model, epochs, trainloader, valloader):
        self.client_id = client_id
        self.model = model
        self.epochs = epochs
        self.trainloader = trainloader
        self.valloader = valloader

    def get_model_weights(self):
        """ Get the model weights and biases """
        return self.model.state_dict()

    def fit(self, server_model_parameters):
        """ Train the client model using new data and server parameters and return the updated model weights and
        biases"""
        # Do not set the server weights and biases if the server aggregation is not done (e.g. initial round)
        if server_model_parameters is not None:
            set_parameters(self.model, server_model_parameters)  # Set the aggregated weights server to the client model

        # Train the client model using new data and server parameters
        train(self.model, self.trainloader, epochs=self.epochs)

        return get_parameters(self.model), len(self.trainloader), {}

    def evaluate(self):
        """ Evaluate the client model on the validation data and return the loss and accuracy """
        loss, accuracy = test(self.model, self.valloader)
        return float(loss), float(accuracy)


def set_parameters(_model, parameters: OrderedDict):
    """ Set the model weights and biases """
    _model.load_state_dict(parameters, strict=True)


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def client_fn(client_id: int, num_local_epochs: int, _dataset_name: str) -> Client:
    """
    Create a client instances on demand for the optimal use of resources.
    :param client_id: client id
    :param num_local_epochs: number of local epochs, before being aggregation ready
    :param _dataset_name: dataset name to load
    :returns Client: A Client instance.
    """
    # Load model
    _model = SimpleModel().to(DEVICE)

    # Each client gets a different dataloaders, so each client will train and evaluate on their own unique data
    train_set, test_set = load_datasets(64, False, _dataset_name)

    # Create a  single Flower client representing a single organization
    return Client(client_id=client_id, model=_model, epochs=num_local_epochs, trainloader=train_set, valloader=test_set)
