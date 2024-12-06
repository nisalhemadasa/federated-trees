"""
Description: This module defines a federated network.

Author: Nisal Hemadasa
Date: 19-10-2024
Version: 1.0
"""
import random
import time
from typing import List, OrderedDict

from torch.utils.data import DataLoader

import constants
from data.dataset_loader import load_datasets
from data.utils import split_dataset, convert_dataset_to_loader, equal_distribution
from drift_concepts.drift import drift_fn, apply_drift, Drift
from federated_network.client import client_fn, Client, set_parameters
from federated_network.server import server_fn, Server
from plots.plotting import plot_client_performance_vs_rounds, plot_server_performance_vs_rounds


def distribute_clients_to_servers(_servers: List[Server], _sampled_clients_model_parameters: List[OrderedDict]) -> None:
    """
    Determines how the distribution of the clients to the servers at the leaves of the hierarchy should be done.
    :param _servers: List of servers at the leaves of the hierarchy
    :param _sampled_clients_model_parameters: List of client model parameters
    :return: None
    """
    # Distribute the clients to the servers according to a given ratio (e.g., equally, etc.)
    num_clients = len(_sampled_clients_model_parameters)
    num_servers = len(_servers)

    # Get the distribution based on the strategy
    client_distribution = equal_distribution(num_clients, num_servers)

    if sum(client_distribution) != num_clients:
        raise ValueError("The distribution strategy must allocate all clients.")

    # Distribute clients to servers
    client_id = 0
    for i, server in enumerate(_servers):
        server.client_ids = list(range(client_id, client_id + client_distribution[i]))
        client_id += client_distribution[i]


def aggregate_client_models(_server_hierarchy: List[List[Server]], _sampled_clients_model_parameters: List[OrderedDict],
                            _server_test_set: DataLoader) -> List:
    """
    Aggregate the models of the clients to the server model.
    :param _server_hierarchy: List of servers in the hierarchy
    :param _sampled_clients_model_parameters: List of client model parameters
    :param _server_test_set: List of test data for server model evaluation, once the aggregation is done
    :return: List of loss and accuracy at each level of the server hierarchy
    """
    # Store the loss and accuracy at each level of the server model hierarchy
    _server_loss_and_accuracy = []

    # Aggregate the models of the clients to the server model
    for depth_level in range(len(_server_hierarchy)):
        loss_and_accuracy_at_level = []

        for server in _server_hierarchy[depth_level]:
            # Aggregate the models of the sampled clients to the server model
            server.train(_sampled_clients_model_parameters)

            # Evaluate server models on the test set
            loss, accuracy = server.evaluate(_server_test_set)
            loss_and_accuracy_at_level.append((loss, accuracy))

        _server_loss_and_accuracy.append(loss_and_accuracy_at_level)

    return _server_loss_and_accuracy


def client_initial_training(_clients: List[Client]) -> List:
    """
    Train the clients initially using their local data.
    :param _clients: List of client instances
    :return:  List of loss and accuracy of each client after the initial training
    """
    initial_client_loss_and_accuracy = []
    # All the clients are trained individually using local data initially
    for client in _clients:
        client.sample_data()
        client.fit(None)
        initial_client_loss_and_accuracy.append(client.evaluate())

    return initial_client_loss_and_accuracy


def train_client_models(_all_clients, _sampled_client_ids, _server: Server, _drift: Drift) -> List:
    """
    Train the client models.
    :param _all_clients: List of all client instances
    :param _sampled_client_ids: List of sampled client IDs
    :param _server: Server instance
    :param _drift: Drift instance
    :return: List of loss and accuracy of each client after training
    """
    round_client_loss_and_accuracy = []

    # Apply drift to the clients
    if _drift.is_drift:
        # Sample data from the drift applied datasets
        apply_drift(_all_clients, _drift)
    else:
        for client in _all_clients:
            # Sample data from the original datasets
            client.sample_data()

    for client in _all_clients:
        # client.sample_data()
        if client.client_id in _sampled_client_ids:
            set_parameters(client.model, _server.server_model.state_dict())
            # round_client_loss_and_accuracy.append(client.evaluate())

            # If the client is sampled in this global training round, then train using the server aggregated parameters
            client.fit(_server.server_model.state_dict())
        else:
            # If the client is not sampled, perform local training without server parameters
            client.fit(None)

            # round_client_loss_and_accuracy.append(client.evaluate())

        # Evaluate the client model after training
        round_client_loss_and_accuracy.append(client.evaluate())

    return round_client_loss_and_accuracy


def update_progress(_round, _num_training_rounds) -> None:
    """
    Update the progress of the simulation
    :param _round: Current simulation iteration number
    :param _num_training_rounds: Total number of training rounds
    :return: None
    """
    progress = (_round / _num_training_rounds) * 100
    print(f"\rSimulation Percentage completed: {progress:.2f}%", end="")


class FederatedNetwork:
    def __init__(self, num_client_instances, server_tree_layout, num_training_rounds, dataset_name, drift_specs,
                 client_select_fraction=0.5, minibatch_size=32, num_local_epochs=4):
        # Dataset name
        self.dataset_name = dataset_name

        # Fraction of clients to be selected for each round (represented by C in originally by McMahan et al. 2017)
        self.client_select_fraction = client_select_fraction

        # Minibatch size for each client (represented by B in originally by McMahan et al. 2017)
        self.minibatch_size = minibatch_size

        # Number of local epochs for each client (represented by E in originally by McMahan et al. 2017)
        self.num_local_epochs = num_local_epochs

        # Number of training rounds
        self.num_training_rounds = num_training_rounds

        # Number of client instances
        self.num_client_instances = num_client_instances

        # Load the dataset
        self.trainset, self.testset = load_datasets(dataset_name)

        # Partition the data set into subsets for each client
        partitioned_trainsets = split_dataset(self.trainset, self.num_client_instances)
        partitioned_testsets = split_dataset(self.testset, self.num_client_instances)

        # Create client instances
        self.clients = [
            client_fn(i, self.num_local_epochs, self.minibatch_size,
                      [partitioned_trainsets[i], partitioned_testsets[i]]) for i in range(num_client_instances)]

        # Concept drift properties
        self.drift = drift_fn(num_client_instances, num_training_rounds, drift_specs)

        # Create instances for servers at each level of the server tree
        server_hierarchy = []
        for depth_level in range(len(server_tree_layout)):
            # For each level in the tree, create a list of server instances
            servers_at_level = [server_fn(server_id) for server_id in range(server_tree_layout[depth_level])]
            server_hierarchy.append(servers_at_level)
        self.server_hierarchy = server_hierarchy

    def sample_clients(self) -> List[Client]:
        """ Sample clients from the client pool and returns a list of client instances """
        return random.sample(self.clients, int(self.client_select_fraction * len(self.clients)))

    def run_simulation(self) -> None:
        """ Run the simulation for the specified number of rounds """
        clients_loss_and_accuracy = []  # Store the loss and accuracy of the all the clients at each round
        sampled_clients_in_each_round = []  # To keep track of the client IDs sampled in each round
        server_loss_and_accuracy = []  # Store the loss and accuracy at each level of the server hierarchy

        # Start the timer
        start_time = time.time()

        # Train the clients initially using their local data
        initial_client_loss_and_accuracy = client_initial_training(self.clients)
        clients_loss_and_accuracy.append(initial_client_loss_and_accuracy)

        # Load the test set for server evaluation
        server_test_set = convert_dataset_to_loader(_dataset=self.testset, _batch_size=self.minibatch_size)

        for _round in range(self.num_training_rounds):
            # # Add drift to the clients, if within the drift period
            # if self.drift.drift_start_round < _round < self.drift.drift_end_round:
            #     self.drift.current_round = _round
            #     self.drift.is_drift = True
            # else:
            #     self.drift.is_drift = False

            # Clients sampled for a single round
            sampled_clients = self.sample_clients()

            # Extract the sampled client IDs and store them
            sampled_client_ids = [client.client_id for client in sampled_clients]
            sampled_clients_in_each_round.append(sampled_client_ids)

            # Implement the clustering algorithm here. That is, which clients to be selected for each server

            # As an example, only one server is considered
            sampled_clients_model_parameters = [sampled_client.model.state_dict() for sampled_client in sampled_clients]
            distribute_clients_to_servers(self.server_hierarchy[0], sampled_clients_model_parameters)

            # Aggregate the models of the clients to the server model
            round_server_loss_and_accuracy = aggregate_client_models(self.server_hierarchy,
                                                                     sampled_clients_model_parameters,
                                                                     server_test_set)
            server_loss_and_accuracy.append(round_server_loss_and_accuracy)

            # Implement local training for every client and evaluate the client models
            round_client_loss_and_accuracy = train_client_models(self.clients, sampled_client_ids,
                                                                 self.server_hierarchy[0][0], self.drift)
            clients_loss_and_accuracy.append(round_client_loss_and_accuracy)

            # Update the progress of the simulation
            update_progress(_round=_round + 1, _num_training_rounds=self.num_training_rounds)

        # Stop the timer
        end_time = time.time()

        print(f"Runtime: {end_time - start_time} seconds")

        # Plot the performance of the clients
        plot_client_performance_vs_rounds(clients_loss_and_accuracy)

        # Plot the performance of the server hierarchy
        plot_server_performance_vs_rounds(server_loss_and_accuracy)
