"""
Description: This module defines the attributes of the concept drift.

Author: Nisal Hemadasa
Date: 29-10-2024
Version: 1.0
"""
import math
from typing import Dict, List

import torch
from scipy.ndimage import rotate
from torch.utils.data import DataLoader, TensorDataset

import constants
from federated_network.client import Client


class Drift:
    def __init__(self, num_drifted_clients, is_synchronous, drift_pattern, drift_method,
                 drift_start_round, drift_end_round, drifted_client_indices, max_rotation, class_pairs_to_swap):
        # Number of clients to be applied with drifted data
        self.num_drifted_clients = num_drifted_clients

        # If the drift is synchronous or asynchronous
        self.is_synchronous = is_synchronous

        # Drift pattern, i.e., abrupt, gradual, incremental, reoccurring, incr-abrupt-reoc, incr-reoc, out-of-control.
        self.drift_pattern = drift_pattern

        # Label-swapping, rotations
        self.drift_method = drift_method

        # Defines the period in training rounds which drift starts appearing in clients
        self.drift_start_round = drift_start_round
        self.drift_end_round = drift_end_round

        # List of clients that have drifted data
        self.drifted_client_indices = drifted_client_indices

        # Maximum rotation angle for the drift created by rotations
        self.max_rotation = max_rotation

        # Classes to be swapped in the label-swapping drift method
        self.class_pairs_to_swap = class_pairs_to_swap

        # Current round of the federated training
        self.current_round = 0

    def rotate_images(self, clients: List[Client]) -> List[Client]:
        """
        Apply rotation drift to the images of the client dataset. Both the rotation angle and the number of images to
        rotate increase linearly with the number of federated training rounds.
        :param clients: Client object List
        :return: Clients objects with the rotated images in their training set
        """
        # The magnitude of the rotation angle (drift) increases linearly with the number of rounds
        transition_progress = ((self.current_round + 1) - self.drift_start_round) / (
                self.drift_end_round - self.drift_start_round)
        rotation_angle = transition_progress * self.max_rotation

        # Calculate the number of images to rotate
        total_rounds = self.drift_end_round - self.drift_start_round + 1
        fraction_rotated = (self.current_round - self.drift_start_round + 1) / total_rounds

        for client in clients:
            # Collect the images and labels data into two separate lists
            all_images = []
            all_labels = []

            # Iterate through the batches in the client’s trainloader
            for batch in client.trainloader:
                images, labels = batch

                if client.client_id in self.drifted_client_indices:
                    # Only a fraction of the batch (of images) are rotated, which increases linearly with the number of
                    # rounds
                    num_images_to_rotate = int(fraction_rotated * len(images))

                    # Clone images for manipulation
                    drifted_images = images.clone()

                    # Rotate a random selection of images if applicable
                    if num_images_to_rotate > 0:
                        indices_to_rotate = torch.randperm(len(images))[:num_images_to_rotate]

                        for i in indices_to_rotate:
                            # Rotate and update the image in the drifted_images tensor
                            rotated_image = rotate(images[i].numpy(), rotation_angle, reshape=False)
                            drifted_images[i] = torch.tensor(rotated_image)

                    all_images.append(drifted_images)
                    all_labels.append(labels)
                else:
                    all_images.append(images)
                    all_labels.append(labels)

            # Concatenate all images and labels and convert to tensors
            all_images = torch.cat(all_images, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            client.trainloader.dataset.images = all_images
            client.trainloader.dataset.labels = all_labels

        return clients

    def swap_labels(self, clients: List[Client]) -> List[Client]:
        """
        Swap the labels of the specified classes in the training set.
        :param clients: Client object List
        :return: Clients objects with the labels swapped in their training set
        """
        # Process each client and update the trainloader in-place if they are in drifted_client_indices
        for client in clients:
            # Collect the images and labels data into two separate lists
            all_images = []
            all_labels = []

            # Iterate through the batches in the client’s trainloader
            for batch in client.trainloader:
                images, labels = batch

                if client.client_id in self.drifted_client_indices:
                    # Identify the indices for each label to be swapped
                    for class_a, class_b in self.class_pairs_to_swap:
                        indices_a = (labels == class_a).nonzero(as_tuple=True)[0]
                        indices_b = (labels == class_b).nonzero(as_tuple=True)[0]

                        # Swap the labels at these indices
                        labels[indices_a] = class_b
                        labels[indices_b] = class_a

                all_images.append(images)
                all_labels.append(labels)

            # Concatenate all images and labels and convert to tensors
            all_images = torch.cat(all_images, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            client.trainloader.dataset.images = all_images
            client.trainloader.dataset.labels = all_labels

        return clients


def get_clients_with_drift(_num_client_instances: int, _clients_fraction_with_drift: float) -> list:
    """
    Get the list of clients that have drifted data.
    :param _num_client_instances: Total number of client instances in the federated network
    :param _clients_fraction_with_drift: Fraction of clients with drifted data
    :return: Indices of clients with drifted data
    """
    num_clients_with_drift = int(_clients_fraction_with_drift * _num_client_instances)
    client_indices = torch.randperm(_num_client_instances).tolist()
    return client_indices[:num_clients_with_drift]


def drift_fn(num_client_instances: int, num_training_rounds: int, drift_specs: Dict) -> Drift:
    """
    Create a drift object using the specifications given as inputs.
    :param num_client_instances: Total number of client instances in the federated network
    :param num_training_rounds: Total number of training rounds
    :param drift_specs: Dictionary containing the drift specifications
    :return: Drift object
    """
    return Drift(num_drifted_clients=drift_specs['clients_fraction'] * num_client_instances,
                 is_synchronous=drift_specs['is_synchronous'],
                 drift_pattern=drift_specs['drift_pattern'],
                 drift_method=drift_specs['drift_method'],
                 drift_start_round=math.ceil(drift_specs['drift_start_round'] * num_training_rounds),
                 drift_end_round=math.ceil(drift_specs['drift_end_round'] * num_training_rounds),
                 drifted_client_indices=get_clients_with_drift(num_client_instances, drift_specs['clients_fraction']),
                 max_rotation=drift_specs['max_rotation'],
                 class_pairs_to_swap=drift_specs['class_pairs_to_swap'])


def apply_drift(clients: List[Client], drift: Drift) -> List[Client]:
    """
    Apply drift to the training data of the clients.
    :param clients: List of Client objects
    :param drift: Drift object
    :return: List of Client objects with drifted data (dataloaders)
    """
    if drift.drift_method == constants.DriftCreationMethods.LABEL_SWAPPING:
        return drift.swap_labels(clients)
    elif drift.drift_method == constants.DriftCreationMethods.ROTATION:
        return drift.rotate_images(clients)
    else:
        print("Drift method not recognized. No drift applied.")
        return clients
