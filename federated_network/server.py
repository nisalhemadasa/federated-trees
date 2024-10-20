"""
Description: This module defines a server of the federated network.

Author: Nisal Hemadasa
Date: 19-10-2024
Version: 1.0
"""
from typing import List, OrderedDict

import strategy


class Server:
    def __init__(self, _server_id, _strategy):
        self.server_id = _server_id
        self.strategy = _strategy
        self.server_model = None

    def train(self, client_model_parameters: List[OrderedDict]) -> None:
        """
        Train the server model using the client model parameters.
        :param client_model_parameters: List of client model parameters
        :return: None
        """
        self.server_model = self.strategy.aggregate_models(self.server_model, client_model_parameters)


def server_fn(server_id: int) -> Server:
    """
    Create a server instances on demand for the optimal use of resources.

    :returns Server: A Server instance.
    """

    aggregator_strategy = strategy.FedAvg.aggregator_fn()

    return Server(_server_id=server_id, _strategy=strategy.FedAvg.fedavg())
