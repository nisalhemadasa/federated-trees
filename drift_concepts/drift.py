"""
Description: This module defines the attributes of the concept drift.

Author: Nisal Hemadasa
Date: 29-10-2024
Version: 1.0
"""
import copy
import math
from typing import Dict, List

import torch
from scipy.ndimage import rotate

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
        print("Drifted clients: ", self.drifted_client_indices)

        # Maximum rotation angle for the drift created by rotations
        self.max_rotation = max_rotation

        # Classes to be swapped in the label-swapping drift method
        self.class_pairs_to_swap = class_pairs_to_swap

        # Current round of the federated training
        self.current_round = 0

        # Flag ot indicate when the drift is being applied to the client data
        self.is_drift = False

        # Flag to indicate if the drift is applied on the client data at least once before
        self.is_already_applied = False

    def rotate_images(self, clients: List[Client]) -> List[Client]:
        """
        Apply rotation drift to the images of the client dataset. Both the rotation angle and the number of images to
        rotate increase linearly with the number of federated training rounds.
        :param clients: List of Client objects
        :return: List of Client objects with the rotated images in their datasets
        """

        def apply_rotation(dataset, _rotation_angle):
            """
            Apply rotation drift to a fraction of the images.
            :param dataset: Dataset to process
            :param _fraction_rotated: Fraction of images to rotate
            :param _rotation_angle: Angle of rotation
            :return: Drifted images and original labels
            """
            _images = dataset.data  # Access dataset images
            _labels = dataset.targets  # Access dataset labels
            # num_images_to_rotate = int(_fraction_rotated * len(_images))
            _drifted_images = _images.clone()

            for i in range(len(_images)):
                rotated_image = rotate(_images[i].numpy(), _rotation_angle, reshape=False)
                _drifted_images[i] = torch.tensor(rotated_image)

            return _drifted_images, _labels

        # Calculate rotation parameters
        transition_progress = ((self.current_round + 1) - self.drift_start_round) / (
                self.drift_end_round - self.drift_start_round)
        # rotation_angle = transition_progress * self.max_rotation
        rotation_angle = self.max_rotation
        total_rounds = self.drift_end_round - self.drift_start_round + 1
        fraction_rotated = (self.current_round - self.drift_start_round + 1) / total_rounds

        # Check if there are drifted clients
        if self.drifted_client_indices:
            # Identify the first drifted client to process the dataset and duplicate a copy (not the reference)
            first_drifted_client = copy.deepcopy(clients[self.drifted_client_indices[0]])

            # Process training dataset
            train_images, train_labels = apply_rotation(first_drifted_client.local_trainset.dataset, rotation_angle)
            first_drifted_client.local_trainset.dataset.data = train_images
            first_drifted_client.local_trainset.dataset.targets = train_labels

            # Process testing dataset
            test_images, test_labels = apply_rotation(first_drifted_client.testset.dataset, rotation_angle)
            first_drifted_client.testset.dataset.data = test_images
            first_drifted_client.testset.dataset.targets = test_labels

            # Assign the updated datasets to all drifted clients, since they share the same data
            for idx in self.drifted_client_indices:
                clients[idx].local_trainset.dataset = first_drifted_client.local_trainset.dataset
                clients[idx].testset.dataset = first_drifted_client.testset.dataset

        return clients


def swap_labels(self, clients: List[Client]) -> List[Client]:
    """
    Swap the labels of the specified classes in the training and testing sets for drifted clients.
    :param clients: List of Client objects
    :return: Updated list of Client objects with swapped labels in their datasets
    """

    def swap_labels_in_dataset(dataset):
        """
        Swap labels in a dataset based on the class pairs to swap.
        :param dataset: Dataset to process
        :return: Updated images and labels tensors
        """
        images = dataset.data  # Access dataset images
        labels = dataset.targets  # Access dataset labels

        for class_a, class_b in self.class_pairs_to_swap:
            indices_a = (labels == class_a).nonzero(as_tuple=True)[0]
            indices_b = (labels == class_b).nonzero(as_tuple=True)[0]

            # Swap the labels
            labels[indices_a] = class_b
            labels[indices_b] = class_a

        return images, labels

    # Check if there are drifted clients
    if self.drifted_client_indices:
        # Identify the first drifted client to process the dataset and duplicate a copy (not the reference)
        first_drifted_client = copy.deepcopy(clients[self.drifted_client_indices[0]])

        # Process training dataset
        train_images, train_labels = swap_labels_in_dataset(first_drifted_client.local_trainset.dataset)
        first_drifted_client.local_trainset.dataset.data = train_images
        first_drifted_client.local_trainset.dataset.targets = train_labels

        # Process testing dataset
        test_images, test_labels = swap_labels_in_dataset(first_drifted_client.testset.dataset)
        first_drifted_client.testset.dataset.data = test_images
        first_drifted_client.testset.dataset.targets = test_labels

        # Assign the updated datasets to all drifted clients, since they share the same data
        for idx in self.drifted_client_indices:
            clients[idx].local_trainset.dataset = first_drifted_client.local_trainset.dataset
            clients[idx].testset.dataset = first_drifted_client.testset.dataset

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
    print("Drift start round: ", math.ceil(drift_specs['drift_start_round'] * num_training_rounds))
    print("Drift end round: ", math.ceil(drift_specs['drift_end_round'] * num_training_rounds))

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
        # For label swapping, application of drift once in the simulation is sufficient & speeds up the simulation
        if not drift.is_already_applied:
            drift.is_already_applied = True
            return drift.swap_labels(clients)
        else:
            return clients
    elif drift.drift_method == constants.DriftCreationMethods.ROTATION:
        # Since rotation is continuously applied, it is speed-wise optimum to apply drift for sampled data in each round
        for client in clients:
            # Each client samples data for local training from their mutually own (exclusively partitioned) datasets
            client.sample_data()

        return drift.rotate_images(clients)
    else:
        print("Drift method not recognized. No drift applied.")
        return clients
