"""
Description: This module defines a server of the federated network.

Author: Nisal Hemadasa
Date: 19-10-2024
Version: 1.0
"""
from typing import List, OrderedDict

from torch.utils.data import DataLoader

import strategy
from federated_network.client import DEVICE
from models.model import SimpleModel, test, CNN


class Server:
    def __init__(self, _server_id, _strategy, _model, _client_ids=None):
        self.server_id = _server_id
        self.strategy = _strategy
        self.server_model = _model
        self.client_ids = []  # List of client IDs the server is connected to in the federated network

    def train(self, client_model_parameters: List[OrderedDict]) -> None:
        """
        Train the server model using the client model parameters.
        :param client_model_parameters: List of client model parameters
        :return: None
        """
        self.server_model = self.strategy.aggregate_models(self.server_model, client_model_parameters)

    def evaluate(self, _test_set: DataLoader) -> (float, float):
        """
        Evaluate the server model using the validation data.
        :param _test_set: test data
        :return: loss and accuracy
        """
        loss, accuracy = test(self.server_model, _test_set)
        return float(loss), float(accuracy)


def server_fn(server_id: int) -> Server:
    """
    Create a server instances on demand for the optimal use of resources.
    :param server_id: Server ID
    :returns Server: A Server instance.
    """
    aggregator_strategy = strategy.FedAvg.aggregator_fn()
    # model = SimpleModel().to(DEVICE)
    model = CNN().to(DEVICE)
    return Server(_server_id=server_id, _strategy=aggregator_strategy, _model=model)
