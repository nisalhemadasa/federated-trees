"""
Description: This module defines the attributes of the concept drift.

Author: Nisal Hemadasa
Date: 29-10-2024
Version: 1.0
"""
from typing import Dict, List

import torch
from scipy.ndimage import rotate
from torch.utils.data import DataLoader

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

    def rotate_images(self, _current_round, _drift_start_round, _drift_end_round, _max_rotation, _images):
        """
        Apply rotation drift to the images.
        :param _current_round: Current federated training round of the simulation
        :param _max_rotation: Maximum rotation angle
        :param _drift_start_round: Federated training round with which the drift starts
        :param _drift_end_round: Federated training round with which the drift ends
        :param _images: Images to be rotated
        :return:
        """

        # Determine the rotation angle based on the current epoch
        if _current_round < _drift_start_round or _current_round > _drift_end_round:
            rotation_angle = 0
        else:
            transition_progress = (_current_round - _drift_start_round) / (_drift_end_round - _drift_start_round)
            rotation_angle = transition_progress * _max_rotation

        # Calculate the number of images to rotate
        total_epochs = _drift_end_round - _drift_start_round + 1
        fraction_rotated = (_current_round - _drift_start_round + 1) / total_epochs
        num_images_to_rotate = int(fraction_rotated * len(_images))

        # Clone images for manipulation
        drifted_images = _images.clone()

        # Rotate a random selection of images if applicable
        if num_images_to_rotate > 0 and fraction_rotated <= 1:
            indices_to_rotate = torch.randperm(len(_images))[:num_images_to_rotate]
            for idx in indices_to_rotate:
                # Rotate and update the image in the drifted_images tensor
                rotated_image = rotate(_images[idx].numpy(), rotation_angle, reshape=False)
                drifted_images[idx] = torch.tensor(rotated_image)

        return drifted_images

    def swap_labels(self, clients: List[Client]) -> List[Client]:
        """
        Swap the labels of the specified classes in the training set.
        :param clients: Client object List
        :return: Clients objects with the labels swapped in their training set
        """
        # drifted_trainloaders = [client.trainloader for client in clients if
        #                         client.client_id in self.drifted_client_indices]
        #
        # # Create a mapping of old labels to new labels
        # label_map = {}
        # for class_a, class_b in self.class_pairs_to_swap:
        #     label_map[class_a] = class_b
        #     label_map[class_b] = class_a
        #
        # # Iterate through the DataLoader
        # for batch in drifted_trainloaders:
        #     images, labels = batch["img"], batch["label"]
        #
        #     # Swap the class labels
        #     for old_label, new_label in label_map.items():
        #         labels[labels == old_label] = new_label
        #
        # return drifted_trainloaders  # Optionally return the modified trainloader

        # Create a mapping of old labels to new labels
        label_map = {}
        for class_a, class_b in self.class_pairs_to_swap:
            label_map[class_a] = class_b
            label_map[class_b] = class_a

        # Process each client and update the trainloader in-place if they are in drifted_client_indices
        for client in clients:
            if client.client_id in self.drifted_client_indices:
                updated_trainloader = []

                # Iterate through the batches in the client’s trainloader
                for batch in client.trainloader:
                    images, labels = batch["img"], batch["label"]

                    # Swap the class labels
                    for old_label, new_label in label_map.items():
                        labels[labels == old_label] = new_label

                    # Append the modified batch to the updated trainloader
                    updated_trainloader.append({"img": images, "label": labels})

                # Update the client's trainloader with the label-swapped version
                client.trainloader = updated_trainloader

        # Return the modified list of clients
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
                 drift_start_round=drift_specs['drift_start_round'] * num_training_rounds,
                 drift_end_round=drift_specs['drift_end_round'] * num_training_rounds,
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
    if drift.drift_method == 'label-swapping':
        return drift.swap_labels(clients)
    elif drift.drift_method == 'rotation':
        for client in clients:
            client.trainloader = drift.rotate_images(drift.current_round, drift.drift_start_round,
                                                     drift.drift_end_round, drift.max_rotation, client.trainloader)
        return clients
