"""
Description: This module defines a federated network.

Author: Nisal Hemadasa
Date: 19-10-2024
Version: 1.0
"""
import random
from typing import List

from data.dataset_loader import load_datasets
from federated_network.client import client_fn, Client
from federated_network.server import server_fn


class FederatedNetwork:
    def __init__(self, num_client_instances, server_tree_layout, num_training_rounds, dataset_name,
                 client_select_fraction=0.5, minibatch_size=32, num_local_epochs=10):
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

        # Create client instances
        self.num_client_instances = num_client_instances
        self.clients = [client_fn(i, self.dataset_name) for i in range(num_client_instances)]

        # Create instances for servers at each level of the server tree
        server_hierarchy = []
        for depth_level in range(len(server_tree_layout)):
            # For each level in the tree, create a list of server instances
            servers_at_level = [server_fn(server_id) for server_id in range(server_tree_layout[depth_level])]
            server_hierarchy .append(servers_at_level)
        self.server_hierarchy = server_hierarchy

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
        clients_loss_n_accuracy = []
        server_loss_n_accuracy = []

        # # All the clients are trained individually using local data initially
        # for client in self.clients:
        #     client.fit(None)
        #     clients_loss_n_accuracy.append(client.evaluate())

        # Load the test set for server evaluation
        _, server_test_set = load_datasets(self.minibatch_size, False, self.dataset_name)

        for _round in range(self.num_training_rounds):
            # 1. The round from which the drift is to be included into the data
            # 2. Assign which clients are to be given drifted data
            # 3. Sample dataloader to the selected clients from the drift included data
            # 4. Sample clients for the round (the step implemented below)
            # Clients sampled for a single round
            sampled_clients = self.sample_clients()
            # Implement the clustering algorithm here. That is, which clients to be selected for each server

            # As an example, only one server is considered
            clients_model_parameters = []
            for client in sampled_clients:
                clients_model_parameters.append(client.model.state_dict())

            # Aggregate the models of the clients to the server model
            for tree_depth_level in range(len(self.servers)):
                this_level_loss_n_accuracy = []
                for server in self.servers[tree_depth_level]:
                    server.train(clients_model_parameters)
                    # Evaluate server models
                    loss, accuracy = server.evaluate(server_test_set)
                    this_level_loss_n_accuracy.append((loss, accuracy))
                server_loss_n_accuracy.append(this_level_loss_n_accuracy)

            # Implement local training for each client with aggregated parameters
            for client in sampled_clients:
                client.fit(self.servers[0][0].server_model.state_dict())
                #### Evaluate the client model
