"""
Description: This script contains utility functions required for operations of the federated network.

Author: Nisal Hemadasa
Date: 09-12-2024
Version: 1.0
"""
from typing import List, OrderedDict

import constants
from drift_concepts.drift import apply_drift, Drift
from federated_network.client import set_parameters
from federated_network.server import Server


def equal_distribution(num_clients: int, num_servers: int) -> List[int]:
    """
    Distribute clients as evenly as possible across servers.
    :param num_clients: Number of clients.
    :param num_servers: Number of servers.
    :return: List of integers representing the number of clients assigned to each server.
    """
    base_clients = num_clients // num_servers
    extra_clients = num_clients % num_servers

    # Distribute extra clients to the first few servers
    return [base_clients + (1 if i < extra_clients else 0) for i in range(num_servers)]


def link_server_hierarchy(server_hierarchy: List[List[Server]]) -> None:
    """
    Link the servers in the hierarchy using a flexible-binary tree structure.
    :param server_hierarchy: List of servers in the hierarchy, where each list represents a level.
    :return: None
    """
    for depth_level in range(len(server_hierarchy) - 1, 0, -1):  # Start from the second-last level
        child_servers = server_hierarchy[depth_level - 1]
        parent_servers = server_hierarchy[depth_level]

        # Divide child servers evenly among parent servers
        num_parents = len(parent_servers)
        num_children = len(child_servers)
        children_per_parent = num_children // num_parents
        extra_children = num_children % num_parents  # Distribute extra children

        child_index = 0  # Track current child server index
        for i, parent_server in enumerate(parent_servers):
            # Assign children to the current parent
            assigned_children = children_per_parent + (1 if i < extra_children else 0)

            for _ in range(assigned_children):
                child_server = child_servers[child_index]
                child_server.parent_server_id = parent_server.server_id
                parent_server.child_server_ids.append(child_server.server_id)
                child_index += 1


def distribute_clients_to_servers(_leaf_servers: List[Server], num_clients: int) -> None:
    """
    Determines how the distribution of the clients to the servers at the leaves of the hierarchy should be done.
    :param _leaf_servers: List of servers at the leaves of the hierarchy
    :param num_clients: Number of clients
    :return: None
    """
    # Distribute the clients to the servers according to a given ratio (e.g., equally, etc.)
    num_servers = len(_leaf_servers)

    # Get the distribution based on the strategy
    client_distribution = equal_distribution(num_clients, num_servers)

    if sum(client_distribution) != num_clients:
        raise ValueError("The distribution strategy must allocate all clients.")

    # Distribute clients to servers in a sequence of ascending order of client IDs
    client_id = 0
    for i, server in enumerate(_leaf_servers):
        server.client_ids = list(range(client_id, client_id + client_distribution[i]))
        client_id += client_distribution[i]


def train_client_models(_all_clients, _sampled_client_ids, _server: Server, _drift: Drift) -> List:
    """
    Train the client models in the network while applying drift if necessary.
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
