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


def plot_client_performance_vs_rounds(loss_and_accuracy: List[List[Tuple]]) -> None:
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
                            constants.Plots.CLIENT_LOSS_VS_ROUNDS_TITLE,
                            constants.Paths.PLOT_SAVE_PATH + constants.Plots.CLIENT_LOSS_VS_ROUNDS_PNG)

    # Plot the accuracy of each client against the number of rounds
    plt.figure()  # Create a new figure for accuracy
    for client_id, accuracies in enumerate(all_clients_accuracies):
        plt.plot(accuracies, label=f'Client {client_id}')

    configure_and_save_plot(plt, constants.Plots.NUMBER_OF_ROUNDS, constants.Plots.ACCURACY,
                            constants.Plots.CLIENT_ACCURACY_VS_ROUNDS_TITLE,
                            constants.Paths.PLOT_SAVE_PATH + constants.Plots.CLIENT_ACCURACY_VS_ROUNDS_PNG)


def plot_server_performance_vs_rounds(loss_and_accuracy: List[List[Tuple]]) -> None:
    """
    Plot the loss and accuracy of the server model against the number of training rounds
    :param loss_and_accuracy: List of tuples containing the loss and accuracy of the models for each client
    :return: None
    """
    # List to store the loss and accuracy values of all clients across all rounds
    all_server_losses = []
    all_server_accuracies = []

    # Indicate the indices of the loss value and accuracy value in the tuple
    LOSS_INDEX = 0
    ACCURACY_INDEX = 1

    # Get the number of server tree hierarchy levels
    num_levels = len(loss_and_accuracy[0])

    # Collect losses for each client across all rounds. Note: Here client_id = index of the client in the list
    for level in range(num_levels):
        level_server_losses = []
        level_server_accuracies = []
        for server_id in range(len(loss_and_accuracy[0][level])):
            server_losses = []
            server_accuracies = []
            for round_index in range(len(loss_and_accuracy)):
                server_losses.append(loss_and_accuracy[round_index][level][server_id][LOSS_INDEX])
                server_accuracies.append(loss_and_accuracy[round_index][level][server_id][ACCURACY_INDEX])
            level_server_losses.append(server_losses)
            level_server_accuracies.append(server_accuracies)
        all_server_losses.append(level_server_losses)
        all_server_accuracies.append(level_server_accuracies)

    # Plot the loss of each server against the number of rounds
    plt.figure()  # Create a new figure for loss
    for level in range(num_levels):
        for server_id, losses in enumerate(all_server_losses[level]):
            plt.plot(losses, label=f'Level {level} Server {server_id}')

    configure_and_save_plot(plt, constants.Plots.NUMBER_OF_ROUNDS, constants.Plots.LOSS,
                            constants.Plots.SERVER_LOSS_VS_ROUNDS_TITLE,
                            constants.Paths.PLOT_SAVE_PATH + constants.Plots.SERVER_LOSS_VS_ROUNDS_PNG)

    # Plot the accuracy of each client against the number of rounds
    plt.figure()  # Create a new figure for accuracy
    for level in range(num_levels):
        for server_id, accuracies in enumerate(all_server_accuracies[level]):
            plt.plot(accuracies, label=f'Level {level} Server {server_id}')

    configure_and_save_plot(plt, constants.Plots.NUMBER_OF_ROUNDS, constants.Plots.ACCURACY,
                            constants.Plots.SERVER_ACCURACY_VS_ROUNDS_TITLE,
                            constants.Paths.PLOT_SAVE_PATH + constants.Plots.SERVER_ACCURACY_VS_ROUNDS_PNG)


def plot_client_avg_performance_vs_rounds(loss_and_accuracy: List[List[Tuple]]) -> None:
    """
    Plot the average loss and accuracy of the clients against the number of training rounds
    :param loss_and_accuracy: List of tuples containing the average loss and accuracy of the all client models for each
    round. - Outer List: drifted and non-drifted clients
           - Inner List: List of performance for each epoch
    :return: None
    """
    # Extract the loss and accuracy values from the list of tuples
    non_drifted_client_avg_accuracies = [x[0] for x in loss_and_accuracy[0]]
    non_drifted_client_avg_losses = [x[1] for x in loss_and_accuracy[0]]
    drifted_client_avg_accuracies = [x[0] for x in loss_and_accuracy[1]]
    drifted_client_avg_losses = [x[1] for x in loss_and_accuracy[1]]

    # Plot the average loss of the clients against the number of rounds
    plt.figure()  # Create a new figure for loss
    plt.plot(non_drifted_client_avg_losses, label='Average Client Loss')
    plt.plot(drifted_client_avg_losses, label='Average Drifted Client Loss')

    configure_and_save_plot(plt, constants.Plots.NUMBER_OF_ROUNDS, constants.Plots.LOSS,
                            constants.Plots.CLIENT_AVG_LOSS_VS_ROUNDS_TITLE,
                            constants.Paths.PLOT_SAVE_PATH + constants.Plots.CLIENT_AVG_LOSS_VS_ROUNDS_PNG)

    # Plot the average accuracy of the clients against the number of rounds
    plt.figure()  # Create a new figure for accuracy
    plt.plot(client_avg_accuracies, label='Average Client Accuracy')

    configure_and_save_plot(plt, constants.Plots.NUMBER_OF_ROUNDS, constants.Plots.ACCURACY,
                            constants.Plots.CLIENT_AVG_ACCURACY_VS_ROUNDS_TITLE,
                            constants.Paths.PLOT_SAVE_PATH + constants.Plots.CLIENT_AVG_ACCURACY_VS_ROUNDS_PNG)


def plot_server_lvl_avg_performance_vs_rounds(loss_and_accuracy: List[Tuple]) -> None:
    """
    Plot the average loss and accuracy of the server models for each level against the number of training rounds
    :param loss_and_accuracy: List of tuples containing the average loss and accuracy of the models for each round of
    all server levels
    :return: None
    """
    if not loss_and_accuracy:
        print('No data to plot')
        return

    num_level = len(loss_and_accuracy[0])  # Number of server tree hierarchy levels
    server_avg_accuracies = []  # stores accuracies for each level [root, ..., leaf]
    server_avg_losses = []  # stores losses for each level [root, ..., leaf]

    for level in range(num_level):
        server_avg_accuracies.append([x[level][1] for x in loss_and_accuracy])
        server_avg_losses.append([x[level][0] for x in loss_and_accuracy])

    # Plot the average loss of the server against the number of rounds
    plt.figure()  # Create a new figure for loss
    for level in range(num_level):
        plt.plot(server_avg_losses[level], label=f'Level {level} Average Server Loss')

    configure_and_save_plot(plt, constants.Plots.NUMBER_OF_ROUNDS, constants.Plots.LOSS,
                            constants.Plots.SERVER_LEVEL_AVG_LOSS_VS_ROUNDS_TITLE,
                            constants.Paths.PLOT_SAVE_PATH + constants.Plots.SERVER_LEVEL_AVG_LOSS_VS_ROUNDS_PNG)

    # Plot the average accuracy of the server against the number of rounds
    plt.figure()  # Create a new figure for accuracy
    for level in range(num_level):
        plt.plot(server_avg_accuracies[level], label=f'Level {level} Average Server Accuracy')

    configure_and_save_plot(plt, constants.Plots.NUMBER_OF_ROUNDS, constants.Plots.ACCURACY,
                            constants.Plots.SERVER_LEVEL_AVG_ACCURACY_VS_ROUNDS_TITLE,
                            constants.Paths.PLOT_SAVE_PATH + constants.Plots.SERVER_LEVEL_AVG_ACCURACY_VS_ROUNDS_PNG)


def plot_server_overall_avg_performance_vs_rounds(loss_and_accuracy: List[Tuple]) -> None:
    """
    Plot the average loss and accuracy of the total server against the number of training rounds
    :param loss_and_accuracy: List of tuples containing the average loss and accuracy of the models for each round of
    all server
    :return: None
    """
    # Extract the loss and accuracy values from the list of tuples
    server_avg_losses = [x[0] for x in loss_and_accuracy]
    server_avg_accuracies = [x[1] for x in loss_and_accuracy]

    # Plot the average loss of the server against the number of rounds
    plt.figure()  # Create a new figure for loss
    plt.plot(server_avg_losses, label='Average Server Loss')

    configure_and_save_plot(plt, constants.Plots.NUMBER_OF_ROUNDS, constants.Plots.LOSS,
                            constants.Plots.SERVER_OVERALL_AVG_LOSS_VS_ROUNDS_TITLE,
                            constants.Paths.PLOT_SAVE_PATH + constants.Plots.SERVEr_OVERALL_AVG_LOSS_VS_ROUNDS_PNG)

    # Plot the average accuracy of the server against the number of rounds
    plt.figure()  # Create a new figure for accuracy
    plt.plot(server_avg_accuracies, label='Average Server Accuracy')

    configure_and_save_plot(plt, constants.Plots.NUMBER_OF_ROUNDS, constants.Plots.ACCURACY,
                            constants.Plots.SERVER_OVERALL_AVG_ACCURACY_VS_ROUNDS_TITLE,
                            constants.Paths.PLOT_SAVE_PATH + constants.Plots.SERVER_OVERALL_AVG_ACCURACY_VS_ROUNDS_PNG)


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
    # _plt.legend()
    # _plt.title(_title)

    # Save the plot as a high-quality PNG
    png_path = f"{_file_path}.png"
    _plt.savefig(png_path, dpi=300)  # Increase DPI for higher resolution

    # Save the plot as a PDF
    pdf_path = f"{_file_path}.pdf"
    _plt.savefig(pdf_path, format="pdf")

    # Display the plot
    _plt.show()
