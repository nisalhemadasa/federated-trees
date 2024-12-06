"""
Description: This script contains utility functions and classes regarding the dataset loading and processing.

Author: Nisal Hemadasa
Date: 19-09-2024
Version: 1.0
"""
from __future__ import annotations

import os

import torch
from torch import Tensor
from torch.utils import data
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision.datasets.mnist import read_image_file, read_label_file
from torchvision import transforms

from typing import List, Tuple, Any

import constants


class CustomDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[Tensor | Any, Tensor]:
        image = self.images[idx]
        label = self.labels[idx]

        if isinstance(image, torch.Tensor):
            # Skip applying transformations that expect PIL or ndarray, if image is already a tensor
            pass
        else:
            image = self.transform(image)

        return image, label


def convert_dataset_to_loader(_dataset: data, _batch_size: int, _is_shuffle: bool = True) -> DataLoader:
    """
    Converts the Dataset object to DataLoader object.
    :param _dataset: Dataset (torch.utils.data objects) object that needs to be converted to DataLoader object.
    :param _batch_size: batch size of loading data. i.e., not all the data available in the dataset is returned.
    :param _is_shuffle: whether to shuffle the data or not.
    :return: DataLoader object.
    """
    return DataLoader(_dataset, batch_size=_batch_size, shuffle=_is_shuffle)


def convert_custom_dataset_to_loader(_dataset: List[Tensor, Tensor], _batch_size: int,
                                     _is_shuffle: bool) -> DataLoader:
    """
    Converts the CustomDataset object to DataLoader object.
    :param _dataset: Input data and labels that needs to be converted to DataLoader object.
    :param _batch_size: batch size of loading data. i.e., not all the data available in the dataset is returned.
    :param _is_shuffle: whether to shuffle the data or not.
    :return: DataLoader object.
    """
    # Assuming grayscale, update for RGB in 'transforms.Normalize', if necessary
    transform_mnist = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.5,), (0.5,))])

    # Wrap the image and label data in a Dataset and create a dataloader
    dataset = CustomDataset(images=_dataset[0], labels=_dataset[1], transform=transform_mnist)

    return DataLoader(dataset, batch_size=_batch_size, shuffle=_is_shuffle)


def read_datasets(_dataset_dir: str, _dataset_filename: constants.DatasetFileNames,
                  _dataset_name: str, _batch_size: int) -> List[Tuple[Tensor, Tensor]]:
    """
    Reads and returns datasets.
    :param _dataset_dir: directory where the dataset is stored.
    :param _dataset_filename: constants.MNISTFilesNames object that contains the filename of the dataset.
    :param _dataset_name: name of the dataset.
    :param _batch_size: batch size of loading data. i.e., not all the data available in the dataset is returned.
    :return: Dataset as a list of Tuples of Tensors.
    """

    trainset = (
        read_image_file(os.path.join(_dataset_dir, _dataset_filename.get_train_images(_dataset_name)[0])),
        read_label_file(os.path.join(_dataset_dir, _dataset_filename.get_train_labels(_dataset_name)[0])),
    )
    testset = (
        read_image_file(os.path.join(_dataset_dir, _dataset_filename.get_test_images(_dataset_name)[0])),
        read_label_file(os.path.join(_dataset_dir, _dataset_filename.get_test_labels(_dataset_name)[0]))
    )

    return [trainset, testset]


def split_dataset(_dataset: Dataset, _num_partitions: int) -> List[Subset]:
    """
    Splits the dataset into mutually exclusive partitions (Dataset -> Subset).
    :param _dataset: Dataset that needs to be split.
    :param _num_partitions: Number of partitions to split the dataset.
    :return: None
    """
    partition_size = len(_dataset) // _num_partitions  # Compute size of each partition
    partition_lengths = [partition_size] * _num_partitions  # Create a list; value=partition_size, length=num_partitions

    # If the dataset cannot be evenly split into partitions, add the remaining data to the last partition. This is to
    # avoid the possible runtime exception when calling random_split() function.
    if len(_dataset) % _num_partitions != 0:
        partition_lengths[-1] += len(_dataset) % _num_partitions

    # Randomly split the training dataset into partitions
    split_datasets = random_split(_dataset, partition_lengths)
    client_indices = [subset.indices for subset in split_datasets]

    # Create subsets based on the indices of the split datasets
    return [Subset(_dataset, indices) for indices in client_indices]


def equal_distribution(num_clients: int, num_servers: int) -> List[int]:
    """
    Distribute clients as evenly as possible across servers.
    :param num_clients: Number of clients.
    :param num_servers: Number of servers.
    :return: List of integers representing the number of clients assigned to each server.
    """
    base_clients = num_clients // num_servers
    extra_clients = num_clients % num_servers

    # Distribute extra clients to the first few servers
    return [base_clients + (1 if i < extra_clients else 0) for i in range(num_servers)]
