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
from drift_concepts.drift import drift_fn, modify_drifted_client_groups
from federated_network.client import client_fn, Client, client_initial_training
from federated_network.server import server_fn, aggregate_client_models, downward_link_aggregate_server_models
from federated_network.utils import update_progress, link_server_hierarchy, train_client_models, link_clients_to_servers
from logs.analysis_functions import compute_client_average_metrics, compute_server_average_metrics, \
    split_clients_loss_and_accuracy
from logs.logging import write_logs
from plots.plotting import plot_client_performance_vs_rounds, plot_server_performance_vs_rounds, \
    plot_client_avg_performance_vs_rounds, plot_server_lvl_avg_performance_vs_rounds, \
    plot_server_overall_avg_performance_vs_rounds


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
        absolute_index = 0

        for depth_level in range(len(server_tree_layout)):
            # For each level in the tree, create a list of server instances
            servers_at_level = [server_fn(server_id, self.dataset_name, absolute_index + i)  # Pass the absolute index
                                for i, server_id in enumerate(range(server_tree_layout[depth_level]))]

            server_hierarchy.append(servers_at_level)
            absolute_index += server_tree_layout[depth_level]

        self.server_hierarchy = server_hierarchy

        # Link servers in the hierarchical structure
        link_server_hierarchy(self.server_hierarchy)

        # Distribute the clients to the leaf servers
        link_clients_to_servers(self.server_hierarchy[-1], self.clients, self.num_client_instances)

    def sample_clients(self) -> List[Client]:
        """ Sample clients from the client pool and returns a list of client instances """
        return random.sample(self.clients, int(self.client_select_fraction * len(self.clients)))

    def run_simulation(self, file_save_path=None, log_save_path=None) -> None:
        """
        Run the simulation for the specified number of rounds
        :param file_save_path: Path to save the logs
        :param log_save_path: Path to save the logs
        :return: None
        """
        clients_loss_and_accuracy = []  # Store the loss and accuracy of the all the clients at each round
        sampled_clients_in_each_round = []  # To keep track of the client IDs sampled in each round
        server_loss_and_accuracy = []  # Store the loss and accuracy at each level of the server hierarchy

        # Start the timer
        start_time = time.time()

        # Train the clients initially using their local data
        initial_client_loss_and_accuracy = client_initial_training(self.clients)
        # clients_loss_and_accuracy.append(initial_client_loss_and_accuracy)

        # Load the test set for server evaluation
        server_test_set = convert_dataset_to_loader(_dataset=self.testset, _batch_size=self.minibatch_size)

        for _round in range(self.num_training_rounds):
            # Add drift to the clients, if within the drift period
            if self.drift.drift_start_round < _round < self.drift.drift_end_round:
                self.drift.current_round = _round
                self.drift.is_drift = True

                # Modify the client groups if the drift is asynchronous
                if not self.drift.is_synchronous:
                    modify_drifted_client_groups(self.drift, _round)
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

            # Server downward-aggregator-link
            if self.simulation_parameters['is_server_downward_aggregation']:
                # Aggregate the models of the clients to the server model
                _ = aggregate_client_models(self.server_hierarchy, sampled_clients_model_parameters, server_test_set)
                round_server_loss_and_accuracy = downward_link_aggregate_server_models(self.server_hierarchy,
                                                                                       server_test_set)
                server_loss_and_accuracy.append(round_server_loss_and_accuracy)
            else:
                # Aggregate the models of the clients to the server model
                round_server_loss_and_accuracy = aggregate_client_models(self.server_hierarchy,
                                                                         sampled_clients_model_parameters,
                                                                         server_test_set)
                server_loss_and_accuracy.append(round_server_loss_and_accuracy)

            # Implement local training for every client and evaluate the client models
            if self.simulation_parameters['is_download_from_root_server']:
                # If the clients download the model from the root server of the hierarchy
                server_depth = 0
            elif self.simulation_parameters['is_download_from_level1_server']:
                # If the clients download the model from the root server of the hierarchy
                server_depth = len(self.server_hierarchy) - 3
            elif self.simulation_parameters['is_download_from_level2_server']:
                # If the clients download the model from the root server of the hierarchy
                server_depth = len(self.server_hierarchy) - 2
            else:
                # If the clients download the model from the leaf servers of the hierarchy
                server_depth = len(self.server_hierarchy) - 1

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
        plot_client_performance_vs_rounds(clients_loss_and_accuracy, file_save_path=file_save_path)

        # Plot the performance of the server hierarchy
        plot_server_performance_vs_rounds(server_loss_and_accuracy, file_save_path=file_save_path)

        # Split the client performance to drifted and non-drifted clients
        if self.drift.is_synchronous:
            non_drifted_clients_loss_and_accuracy, drifted_clients_loss_and_accuracy = split_clients_loss_and_accuracy(
                clients_loss_and_accuracy, self.drift.drifted_client_indices, None)
        else:
            non_drifted_clients_loss_and_accuracy, drifted_clients_loss_and_accuracy = split_clients_loss_and_accuracy(
                clients_loss_and_accuracy, self.drift.drifted_client_indices,
                self.drift.async_drift_specs['drift_groups'])

        # Get average performance of the clients
        non_drifted_client_averages = compute_client_average_metrics(non_drifted_clients_loss_and_accuracy)
        if self.drift.is_synchronous:
            drifted_client_averages = compute_client_average_metrics(drifted_clients_loss_and_accuracy)
        else:
            drifted_client_averages = []
            for drited_groups in drifted_clients_loss_and_accuracy:
                drifted_client_averages.append(compute_client_average_metrics(drited_groups))

        if log_save_path is None:
            log_save_path = constants.Paths.LOG_SAVE_PATH

        # Log the performance of the clients
        write_logs(clients_loss_and_accuracy, file_name=log_save_path + constants.Logs.CLIENT_LOG)
        # Log the performance of the clients separated by drifted and non-drifted
        write_logs(non_drifted_clients_loss_and_accuracy,
                   file_name=log_save_path + constants.Logs.NON_DRIFTED_CLIENT_LOG)
        write_logs(drifted_clients_loss_and_accuracy,
                   file_name=log_save_path + constants.Logs.DRIFTED_CLIENT_LOG)
        # Average performance of the clients
        write_logs(non_drifted_client_averages,
                   file_name=log_save_path + constants.Logs.NON_DRIFTED_CLIENT_AVG_LOG)
        write_logs(drifted_client_averages,
                   file_name=log_save_path + constants.Logs.DRIFTED_CLIENT_AVG_LOG)

        # Get average performance of the servers
        server_level_averages, server_overall_averages = compute_server_average_metrics(server_loss_and_accuracy)

        # Log the performance of the server hierarchy
        write_logs(server_loss_and_accuracy, file_name=log_save_path + constants.Logs.SERVER_LOG)
        write_logs(server_level_averages, file_name=log_save_path + constants.Logs.SERVER_LVL_AVG_LOG)
        write_logs(server_overall_averages, file_name=log_save_path + constants.Logs.SERVER_OVERALL_AVG_LOG)

        # Plot average performances
        plot_client_avg_performance_vs_rounds([non_drifted_client_averages, drifted_client_averages],
                                              self.drift.is_synchronous,
                                              file_save_path=file_save_path)
        plot_server_lvl_avg_performance_vs_rounds(server_level_averages, file_save_path=file_save_path)
        plot_server_overall_avg_performance_vs_rounds(server_overall_averages, file_save_path=file_save_path)
