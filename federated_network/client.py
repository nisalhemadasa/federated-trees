"""
Description: This module defines a client of the federated network.

Author: Nisal Hemadasa
Date: 18-10-2024
Version: 1.0
"""
import random
from collections import OrderedDict
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from data.utils import convert_dataset_to_loader
from models.model import train, test, SimpleModel, CNN

DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__}"
)


class Client:
    def __init__(self, client_id, model, epochs, mini_batch_size, local_trainset, testset):
        self.client_id = client_id
        self.model = model
        self.epochs = epochs
        self.local_trainset = local_trainset
        self.testset = testset
        self.mini_batch_size = mini_batch_size
        self.trainloader = None  # initialized only when sample_data() is called
        self.testloader = None  # initialized only when sample_data() is called

    def get_model_weights(self):
        """ Get the model weights and biases """
        return self.model.state_dict()

    def sample_data(self):
        """ Sample data from the train and test datasets unique to each client and create DataLoaders"""
        # Create a DataLoader using a randomly sampled subset(fraction_of_data%) from the local training data
        fraction_of_data = 0.1
        subset_size = int(len(self.local_trainset) * fraction_of_data)
        indices = random.sample(range(len(self.local_trainset)), subset_size)
        subset = Subset(self.local_trainset, indices)

        self.trainloader = convert_dataset_to_loader(_dataset=subset,
                                                     _batch_size=self.mini_batch_size)
        self.testloader = convert_dataset_to_loader(_dataset=self.testset, _batch_size=self.mini_batch_size,
                                                    _is_shuffle=False)

    def fit(self, server_model_parameters):
        """ Train the client model using new data and server parameters and return the updated model weights and
        biases"""
        # Do not set the server weights and biases if the server aggregation is not done (e.g. initial round)
        if server_model_parameters is not None:
            set_parameters(self.model, server_model_parameters)  # Set the aggregated weights server to the client model

        # Train the client model using new data and server parameters
        train(self.model, self.trainloader, epochs=self.epochs)

        return get_parameters(self.model), len(self.trainloader)

    def evaluate(self):
        """ Evaluate the client model on the validation data and return the loss and accuracy """
        loss, accuracy = test(self.model, self.testloader)
        return float(loss), float(accuracy)


def set_parameters(_model, parameters: OrderedDict):
    """ Set the model weights and biases """
    _model.load_state_dict(parameters, strict=True)


def get_parameters(net) -> List[np.ndarray]:
    """ Set the model weights and biases """
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

    """ Get the train and test dataloaders """
    trainloader = convert_dataset_to_loader(_dataset=trainset, _batch_size=mini_batch_size)
    testloader = convert_dataset_to_loader(_dataset=testset, _batch_size=mini_batch_size, _is_shuffle=False)
    return trainloader, testloader


def client_fn(client_id: int, num_local_epochs: int, mini_batch_size: int, _dataset: List[Dataset]) -> Client:
    """
    Create a client instances on demand for the optimal use of resources.
    :param client_id: client id
    :param num_local_epochs: number of local epochs, before being aggregation ready
    :param mini_batch_size: size of the batches for the clients to train on
    :param _dataset: train and test datasets
    :returns Client: A Client instance.
    """
    # Load model
    # _model = SimpleModel().to(DEVICE)
    _model = CNN().to(DEVICE)

    # Upacking _dataset (which contains a subset of the complete training set (e.g., MNIST) and the global test set)
    local_trainset, testset = _dataset

    # Create a  single Flower client representing a single organization
    return Client(client_id=client_id, model=_model, epochs=num_local_epochs, mini_batch_size=mini_batch_size,
                  local_trainset=local_trainset, testset=testset)
