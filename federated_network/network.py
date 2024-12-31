"""
Description: This module defines a federated network.

Author: Nisal Hemadasa
Date: 19-10-2024
Version: 1.0
"""
import random
import time
from typing import List

import constants
from data.dataset_loader import load_datasets
from data.utils import split_dataset, convert_dataset_to_loader
from drift_concepts.drift import drift_fn
from federated_network.client import client_fn, Client, client_initial_training
from federated_network.server import server_fn, aggregate_client_models, downward_link_aggregate_server_models
from federated_network.utils import update_progress, link_server_hierarchy, train_client_models, link_clients_to_servers
from plots.plotting import plot_client_performance_vs_rounds, plot_server_performance_vs_rounds


class FederatedNetwork:
    def __init__(self, num_client_instances, server_tree_layout, num_training_rounds, dataset_name, drift_specs,
                 simulation_parameters, client_select_fraction=0.5, minibatch_size=32, num_local_epochs=4):
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
            client_fn(i, self.num_local_epochs, self.minibatch_size, self.dataset_name,
                      [partitioned_trainsets[i], partitioned_testsets[i]]) for i in range(num_client_instances)]

        # Concept drift properties
        self.drift = drift_fn(num_client_instances, num_training_rounds, drift_specs)

        # Simulation parameters
        self.simulation_parameters = simulation_parameters

        # Create instances for servers at each level of the server tree
        server_hierarchy = []
        for depth_level in range(len(server_tree_layout)):
            # For each level in the tree, create a list of server instances
            servers_at_level = [server_fn(server_id, self.dataset_name) for server_id in range(server_tree_layout[depth_level])]
            server_hierarchy.append(servers_at_level)
        self.server_hierarchy = server_hierarchy

        # Link servers in the hierarchical structure
        link_server_hierarchy(self.server_hierarchy)

        # Distribute the clients to the leaf servers
        link_clients_to_servers(self.server_hierarchy[0], self.clients, self.num_client_instances)

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
            # Add drift to the clients, if within the drift period
            if self.drift.drift_start_round < _round < self.drift.drift_end_round:
                self.drift.current_round = _round
                self.drift.is_drift = True
            else:
                self.drift.is_drift = False

            # Clients sampled for a single round
            sampled_clients = self.sample_clients()

            # Extract the sampled client IDs and store them
            sampled_client_ids = [client.client_id for client in sampled_clients]
            sampled_clients_in_each_round.append(sampled_client_ids)

            # Implement the clustering algorithm here. That is, which clients to be selected for each server

            # As an example, only one server is considered
            sampled_clients_model_parameters = [sampled_client.model.state_dict() for sampled_client in sampled_clients]

            # Aggregate the models of the clients to the server model
            round_server_loss_and_accuracy = aggregate_client_models(self.server_hierarchy,
                                                                     sampled_clients_model_parameters,
                                                                     server_test_set)
            # server_loss_and_accuracy.append(round_server_loss_and_accuracy)

            # Server downward-aggregator-link
            if self.simulation_parameters['is_server_downward_aggregation']:
                round_server_loss_and_accuracy = downward_link_aggregate_server_models(self.server_hierarchy,
                                                                                       server_test_set)
                server_loss_and_accuracy.append(round_server_loss_and_accuracy)

            # Implement local training for every client and evaluate the client models
            if self.simulation_parameters['is_download_from_root_server']:
                # If the clients download the model from the root server of the hierarchy
                server_depth = len(self.server_hierarchy) - 1
            else:
                # If the clients download the model from the leaf servers of the hierarchy
                server_depth = 0

            round_client_loss_and_accuracy = train_client_models(self.clients,
                                                                 sampled_client_ids,
                                                                 self.server_hierarchy[server_depth],
                                                                 self.drift,
                                                                 self.simulation_parameters)
            clients_loss_and_accuracy.append(round_client_loss_and_accuracy)

            # Update the progress of the simulation
            update_progress(_round=_round + 1, num_training_rounds=self.num_training_rounds)

        # Stop the timer
        end_time = time.time()

        print(f"Runtime: {end_time - start_time} seconds")

        # Plot the performance of the clients
        plot_client_performance_vs_rounds(clients_loss_and_accuracy)

        # Plot the performance of the server hierarchy
        plot_server_performance_vs_rounds(server_loss_and_accuracy)
