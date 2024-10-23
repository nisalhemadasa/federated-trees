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
        clients_loss_and_accuracy = []
        initial_client_loss_and_accuracy = []
        server_hierarchy_loss_and_accuracy = []
        sampled_clients_in_each_round = []  # To keep track of the client IDs sampled in each round

        # All the clients are trained individually using local data initially
        for client in self.clients:
            client.fit(None)
            initial_client_loss_and_accuracy.append(client.evaluate())

        clients_loss_and_accuracy.append(initial_client_loss_and_accuracy)

        # Load the test set for server evaluation
        _, server_test_set = load_datasets(self.minibatch_size, False, self.dataset_name)

        for _round in range(self.num_training_rounds):
            # 1. The round from which the drift is to be included into the data
            # 2. Assign which clients are to be given drifted data
            # 3. Sample dataloader to the selected clients from the drift included data
            # 4. Sample clients for the round (the step implemented below)

            # Clients sampled for a single round
            sampled_clients = self.sample_clients()

            # Extract the sampled client IDs and store them
            sampled_client_ids = [client.client_id for client in sampled_clients]
            sampled_clients_in_each_round.append(sampled_client_ids)

            # Implement the clustering algorithm here. That is, which clients to be selected for each server

            # As an example, only one server is considered
            sampled_clients_model_parameters = [sampled_client.model.state_dict() for sampled_client in sampled_clients]

            # Aggregate the models of the clients to the server model
            for depth_level in range(len(self.server_hierarchy)):
                loss_and_accuracy_at_level = []

                for server in self.server_hierarchy[depth_level]:
                    # Aggregate the models of the sampled clients to the server model
                    server.train(sampled_clients_model_parameters)

                    # Evaluate server models on the test set
                    loss, accuracy = server.evaluate(server_test_set)
                    loss_and_accuracy_at_level.append((loss, accuracy))

                server_hierarchy_loss_and_accuracy.append(loss_and_accuracy_at_level)

            # Implement local training for every client and evaluate the client models
            round_client_loss_and_accuracy = []

            for client in self.clients:
                if client.client_id in sampled_client_ids:
                    # if the model is sampled, then train using the server aggregated parameters
                    client.fit(self.server_hierarchy[0][0].server_model.state_dict())
                else:
                    # If the client is not sampled, perform local training without server parameters
                    client.fit(None)

                # Evaluate the client model after training
                round_client_loss_and_accuracy.append(client.evaluate())
                
            # Store the performance of all clients for this round
            clients_loss_and_accuracy.append(round_client_loss_and_accuracy)
