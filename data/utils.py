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
from torch.utils.data import Dataset, DataLoader
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

            # if self.transform:
        #     image = self.transform(image)

        return image, label


def convert_dataset_to_loader(_dataset: Tuple[Tensor, Tensor], _batch_size: int) -> DataLoader:
    """
    Converts the Dataset object to DataLoader object.
    :param _dataset: Dataset object that needs to be converted to DataLoader object.
    :param _batch_size: batch size of loading data. i.e., not all the data available in the dataset is returned.
    :return: DataLoader object.
    """
    # Assuming grayscale, update for RGB in 'transforms.Normalize', if necessary
    transform_mnist = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.5,), (0.5,))])

    # Wrap the image and label data in a Dataset and create a dataloader
    dataset = CustomDataset(images=_dataset[0], labels=_dataset[1], transform=transform_mnist)
    dataloader = DataLoader(dataset, batch_size=_batch_size, shuffle=True)

    return dataloader


def get_loaders_from_datasets(_dataset_dir: str, _dataset_filename: constants.DatasetFileNames,
                              _dataset_name: str, _batch_size: int) -> \
        List[DataLoader]:
    """
    Reads and converts the saved datasets to DataLoader objects.
    :param _dataset_dir: directory where the dataset is stored.
    :param _dataset_filename: constants.MNISTFilesNames object that contains the filename of the dataset.
    :param _dataset_name: name of the dataset.
    :param _batch_size: batch size of loading data. i.e., not all the data available in the dataset is returned.
    :return: List of DataLoader objects.
    """

    training_set = (
        read_image_file(os.path.join(_dataset_dir, _dataset_filename.get_train_images(_dataset_name)[0])),
        read_label_file(os.path.join(_dataset_dir, _dataset_filename.get_train_labels(_dataset_name)[0])),
    )
    test_set = (
        read_image_file(os.path.join(_dataset_dir, _dataset_filename.get_test_images(_dataset_name)[0])),
        read_label_file(os.path.join(_dataset_dir, _dataset_filename.get_test_labels(_dataset_name)[0]))
    )

    return [convert_dataset_to_loader(training_set, _batch_size), convert_dataset_to_loader(test_set, _batch_size)]
