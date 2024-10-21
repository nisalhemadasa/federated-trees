"""
Description: This module defines a federated network.

Author: Nisal Hemadasa
Date: 19-10-2024
Version: 1.0
"""
import random
from typing import List

from federated_network.client import client_fn, Client
from federated_network.server import server_fn


class FederatedNetwork:
    def __init__(self, num_client_instances, server_tree_layout, num_training_rounds, client_select_fraction=0.5):
        # Create client instances
        self.num_client_instances = num_client_instances
        self.clients = [client_fn(i) for i in range(num_client_instances)]

        # Fraction of clients to be selected for each round
        self.client_select_fraction = client_select_fraction

        # Number of training rounds
        self.num_training_rounds = num_training_rounds

        # Create server instances
        servers = []
        for i in range(len(server_tree_layout)):
            # Create servers for each level
            servers.append([server_fn(j) for j in range(server_tree_layout[i])])
        self.servers = servers

    def sample_clients(self) -> List[Client]:
        """
        Sample clients from the client pool.
        :return: list of client instances
        """
        return random.sample(self.clients, int(self.client_select_fraction * len(self.clients)))

    def run_simulation(self) -> None:
        """
        Run the simulation for the specified number of rounds.
        :return: None
        """
        for _round in range(self.num_training_rounds):
            # 1. The round from which the drift is to be included into the data
            # 2. Assign which clients are to be given drifted data
            # 3. Sample dataloader to the selected clients from the drift included data
            # 4. Sample clients for the round (the step implemented below)
            # Clients sampled for a single round
            sampled_clients = self.sample_clients()
            # Implement the clustering algorithm here. That is, which clients to be selected for each server

            # Implement local training for each client
            for client in sampled_clients:
                client.fit()
                l =1

            # As an example, only one server is considered
            clients_model_parameters = []
            for client in sampled_clients:
                clients_model_parameters.append(client.model.state_dict())

            # Aggregate the models of the clients to the server model
            self.servers[0][0].train(clients_model_parameters)
