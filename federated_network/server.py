"""
Description: This module defines a server of the federated network.

Author: Nisal Hemadasa
Date: 19-10-2024
Version: 1.0
"""
from typing import List, OrderedDict

from torch.utils.data import DataLoader

import constants
import strategy
from federated_network.client import DEVICE
from models.model import SimpleModel, test, CNNMNIST, CNNCIFAR10


class Server:
    def __init__(self, _server_id, _strategy, _model, _client_ids=None):
        self.server_id = _server_id
        self.strategy = _strategy
        self.model = _model
        self.client_ids = []  # List of client IDs the server is connected to in the federated network
        self.child_server_ids = []  # List of child server IDs in the server hierarchy
        self.parent_server_id = None  # Parent server ID in the server hierarchy

    def train(self, client_model_parameters: List[OrderedDict]) -> None:
        """
        Train the server model using the client model parameters.
        :param client_model_parameters: List of client model parameters
        :return: None
        """
        self.model = self.strategy.aggregate_models(self.model, client_model_parameters)

    def evaluate(self, _test_set: DataLoader) -> (float, float):
        """
        Evaluate the server model using the validation data.
        :param _test_set: test data
        :return: loss and accuracy
        """
        loss, accuracy = test(self.model, _test_set)
        return float(loss), float(accuracy)


def aggregate_client_models(server_hierarchy: List[List[Server]], sampled_clients_model_parameters: List[OrderedDict],
                            server_test_set: DataLoader) -> List:
    """
    Aggregate the models of the clients to the server model.
    :param server_hierarchy: List of servers in the hierarchy
    :param sampled_clients_model_parameters: List of client model parameters
    :param server_test_set: List of test data for server model evaluation, once the aggregation is done
    :return: List of loss and accuracy at each level of the server hierarchy
    """
    # Store the loss and accuracy at each level of the server model hierarchy
    server_loss_and_accuracy = []

    # Aggregate the models of the clients to the server model
    for depth_level in range(len(server_hierarchy)):
        loss_and_accuracy_at_level = []

        for server in server_hierarchy[depth_level]:
            if depth_level == 0:
                # Leaf nodes: Aggregate client models
                client_model_parameters = [sampled_clients_model_parameters[client_id] for client_id in
                                           server.client_ids]

                # Aggregate client models
                server.train(client_model_parameters)
            else:
                # Internal nodes: Aggregate models from child servers
                child_server_model_parameters = [child_server.model.state_dict() for child_server in
                                                 server_hierarchy[depth_level - 1]]

                # Aggregate child server models
                server.train(child_server_model_parameters)

            # Evaluate the server model
            loss, accuracy = server.evaluate(server_test_set)
            loss_and_accuracy_at_level.append((loss, accuracy))

        server_loss_and_accuracy.append(loss_and_accuracy_at_level)

    return server_loss_and_accuracy


def downward_link_aggregate_server_models(server_hierarchy: List[List[Server]], server_test_set: DataLoader) -> List:
    """
    Aggregate the models of the child servers to the parent server model, along the download link. i.e. parameters of
    the root server gets aggregated with the parameters of its child servers, and gets assigned to the child servers.
    :param server_hierarchy: List of servers in the hierarchy
    :param server_test_set: List of test data for server model evaluation, once the aggregation is done
    :return: None
    """
    # Store the loss and accuracy at each level of the server model hierarchy
    server_loss_and_accuracy = []

    # Aggregate the parent server to the child server down the hierarchy starting from the leaf nodes
    for depth_level in range(len(server_hierarchy) - 1):
        loss_and_accuracy_at_level = []

        for server in server_hierarchy[depth_level]:
            # Get the server parameters and the parent server parameters
            parent_server = server_hierarchy[depth_level + 1][server.parent_server_id]
            server_parameters = [server.model.state_dict(), parent_server.model.state_dict()]

            # Aggregate child server models
            server.train(server_parameters)

            # Evaluate the server model
            loss, accuracy = server.evaluate(server_test_set)
            loss_and_accuracy_at_level.append((loss, accuracy))

        server_loss_and_accuracy.append(loss_and_accuracy_at_level)

    return server_loss_and_accuracy


def server_fn(server_id: int, dataset_name: str) -> Server:
    """
    Create a server instances on demand for the optimal use of resources.
    :param server_id: Server ID
    :param dataset_name: Name of the dataset
    :returns Server: A Server instance.
    """
    aggregator_strategy = strategy.FedAvg.aggregator_fn()
    # model = SimpleModel().to(DEVICE)
    if dataset_name == constants.DatasetNames.CIFAR_10:
        model = CNNCIFAR10().to(DEVICE)
    else:
        model = CNNMNIST().to(DEVICE)

    return Server(_server_id=server_id, _strategy=aggregator_strategy, _model=model)
