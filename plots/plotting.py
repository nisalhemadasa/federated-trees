"""
Description: This module contains the functions to plot the performance metrics of the federated network.

Author: Nisal Hemadasa
Date: 23-10-2024
Version: 1.0
"""

from typing import List, Tuple

import matplotlib.pyplot as plt
import matplotlib

# Choose a backend based on what works best for your environment
import constants

matplotlib.use('TkAgg')  # Or 'TkAgg', 'Qt5Agg', etc.


def plot_performance_vs_rounds(loss_and_accuracy: List[List[Tuple]]) -> None:
    """
    Plot the loss of the models against the number of training rounds
    :param loss_and_accuracy: List of tuples containing the loss and accuracy of the models for each client
    :return: None
    """
    # List to store the loss and accuracy values of all clients across all rounds
    all_client_losses = []
    all_clients_accuracies = []

    # Indicate the indices of the loss value and accuracy value in the tuple
    LOSS_INDEX = 0
    ACCURACY_INDEX = 1

    # Get the total number of clients to iterate
    num_clients = len(loss_and_accuracy[0])

    # Collect losses for each client across all rounds. Note: Here client_id = index of the client in the list
    for client_id in range(num_clients):
        client_losses = []
        client_accuracies = []
        for round_index in range(len(loss_and_accuracy)):
            client_losses.append(loss_and_accuracy[round_index][client_id][LOSS_INDEX])
            client_accuracies.append(loss_and_accuracy[round_index][client_id][ACCURACY_INDEX])
        all_client_losses.append(client_losses)
        all_clients_accuracies.append(client_accuracies)

    # Plot the loss of each client against the number of rounds
    plt.figure()  # Create a new figure for loss
    for client_id, losses in enumerate(all_client_losses):
        plt.plot(losses, label=f'Client {client_id}')

    configure_and_save_plot(plt, constants.Plots.NUMBER_OF_ROUNDS, constants.Plots.LOSS,
                            constants.Plots.LOSS_VS_ROUNDS_TITLE,
                            constants.Paths.PLOT_SAVE_PATH + constants.Plots.LOSS_VS_ROUNDS_PNG)

    # Plot the accuracy of each client against the number of rounds
    plt.figure()  # Create a new figure for accuracy
    for client_id, accuracies in enumerate(all_clients_accuracies):
        plt.plot(accuracies, label=f'Client {client_id}')

    configure_and_save_plot(plt, constants.Plots.NUMBER_OF_ROUNDS, constants.Plots.ACCURACY,
                            constants.Plots.ACCURACY_VS_ROUNDS_TITLE,
                            constants.Paths.PLOT_SAVE_PATH + constants.Plots.ACCURACY_VS_ROUNDS_PNG)


def configure_and_save_plot(_plt, _x_label, _y_label, _title, _file_path):
    """
    Add labels, title, legend and save the plot and displays it.
    :param _plt: The matplotlib pyplot object.
    :param _x_label: The label for the x-axis.
    :param _y_label: The label for the y-axis.
    :param _title: The title of the plot.
    :param _file_path: The path and file name to save the plot.
    :return: None
    """
    _plt.xlabel(_x_label)
    _plt.ylabel(_y_label)
    _plt.legend()
    _plt.title(_title)
    _plt.savefig(_file_path)
    _plt.show()
