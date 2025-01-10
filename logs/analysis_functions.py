"""
Description: This module contains the functions to analyse the log data.

Author: Nisal Hemadasa
Date: 10-01-2025
Version: 1.0
"""

from typing import List, Tuple

import constants
from logs.logging import read_logs
from plots.plotting import plot_client_performance_vs_rounds, plot_server_performance_vs_rounds, \
    plot_client_avg_performance_vs_rounds, plot_server_lvl_avg_performance_vs_rounds, \
    plot_server_overall_avg_performance_vs_rounds


def compute_client_average_metrics(data: List[List[Tuple]]) -> List[Tuple[float, float]]:
    """
    Compute the average accuracy and loss for all clients in each round (epoch).

    :param data: List[List[Tuple[float, float]]], where:
                 - Outer List represents the number of epochs (rounds).
                 - Inner List represents the clients.
                 - Tuple contains (accuracy, loss) for each client.
    :return: List[Tuple[float, float]]: List of average (accuracy, loss) for each round.
    """
    averages = []

    for epoch_data in data:  # Iterate over each epoch
        total_loss = 0.0
        total_accuracy = 0.0
        num_clients = len(epoch_data)

        for client_metrics in epoch_data:  # Iterate over each client
            loss, accuracy = client_metrics
            total_loss += loss
            total_accuracy += accuracy

        # Calculate average accuracy and loss for the epoch
        avg_accuracy = total_accuracy / num_clients if num_clients > 0 else 0.0
        avg_loss = total_loss / num_clients if num_clients > 0 else 0.0

        averages.append((avg_accuracy, avg_loss))

    return averages


def compute_server_average_metrics(data: List[List[List[Tuple[float, float]]]]) -> Tuple[
    List[List[Tuple[float, float]]], List[Tuple[float, float]]]:
    """
    Compute the average accuracy and loss for all servers across depth levels in each round (epoch).
    Additionally, compute the overall average for all servers in each epoch.

    :param data: List[List[List[Tuple[float, float]]]], where:
                 - Outer List represents the number of epochs (rounds).
                 - Second List represents the depth levels of the hierarchy [root, ..., leaves].
                 - Inner List represents the servers inside a given depth level.
                 - Tuple contains (accuracy, loss) for each server.
    :return: Tuple containing:
             - List[List[Tuple[float, float]]]: Average (accuracy, loss) for each level in each epoch. [root, ..., leaves
             - List[Tuple[float, float]]: Overall average (accuracy, loss) for all servers in each epoch.
    """
    level_averages = []
    overall_averages = []

    for epoch_data in data:  # Iterate over each epoch
        epoch_level_averages = []
        total_accuracy = 0.0
        total_loss = 0.0
        num_servers_total = 0

        for depth_level_data in epoch_data:  # Iterate over each depth level
            level_accuracy = 0.0
            level_loss = 0.0
            num_servers_level = len(depth_level_data)

            for server_metrics in depth_level_data:  # Iterate over each server
                accuracy, loss = server_metrics
                level_accuracy += accuracy
                level_loss += loss

            # Calculate average for the current level
            avg_level_accuracy = level_accuracy / num_servers_level if num_servers_level > 0 else 0.0
            avg_level_loss = level_loss / num_servers_level if num_servers_level > 0 else 0.0
            epoch_level_averages.append((avg_level_accuracy, avg_level_loss))  # [root, ..., leaves]

            # Update overall totals
            total_accuracy += level_accuracy
            total_loss += level_loss
            num_servers_total += num_servers_level

        # Append level-wise averages for the current epoch
        level_averages.append(epoch_level_averages)

        # Calculate overall average for the epoch
        avg_epoch_accuracy = total_accuracy / num_servers_total if num_servers_total > 0 else 0.0
        avg_epoch_loss = total_loss / num_servers_total if num_servers_total > 0 else 0.0
        overall_averages.append((avg_epoch_accuracy, avg_epoch_loss))

    return level_averages, overall_averages


def plot_average_performance() -> None:
    """
    Read average performance logs and plot the average accuracy and loss for each round.
    :return: None
    """
    # Read the logs
    client_file_name = constants.Paths.LOG_SAVE_PATH + constants.Logs.CLIENT_AVG_LOG + constants.FileExtesions.PKL
    server_lvl_file_name = constants.Paths.LOG_SAVE_PATH + constants.Logs.SERVER_LVL_AVG_LOG + constants.FileExtesions.PKL
    server_overall_file_name = constants.Paths.LOG_SAVE_PATH + constants.Logs.SERVER_OVERALL_AVG_LOG + constants.FileExtesions.PKL

    clients_avg_loss_and_accuracy = read_logs(client_file_name)
    server_lvl_avg_loss_and_accuracy = read_logs(server_lvl_file_name)
    server_overall_avg_loss_and_accuracy = read_logs(server_overall_file_name)

    # Plot the logs
    plot_client_avg_performance_vs_rounds(clients_avg_loss_and_accuracy)
    plot_server_lvl_avg_performance_vs_rounds(server_lvl_avg_loss_and_accuracy)
    plot_server_overall_avg_performance_vs_rounds(server_overall_avg_loss_and_accuracy)
