"""
Description: This script loads the downloaded the datasets for further processing. It downloads the dataset if needed.

Author: Nisal Hemadasa
Date: 06-08-2024
Version: 1.0
"""
import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from typing import List

from torchvision.datasets.mnist import read_image_file, read_label_file

import constants
from data.utils import get_loaders_from_datasets


def load_datasets(_batch_size: int, _if_dataset_download: bool, _dataset_name: str) -> List[DataLoader]:
    """
    Downloads and loads the datasets MNIST, Fashion MNIST and CIFAR-10.
    :param _batch_size: batch size of loading data. i.e., not all the data avaialable in the dataset is returned.
    :param _if_dataset_download: if the dataset seeds to be downloaded before loading.
    :param _dataset_name: name of the dataset that needs to be returned. i.e., not the data from all datasets are
    returned.
    :return: List of datasets of type torchvision.datasets
    """
    _if_dataset_download = False

    if _dataset_name == constants.DatasetNames.MNIST:
        data_files = [constants.DatasetFileNames.MNIST_TRAIN_IMAGES,
                      constants.DatasetFileNames.MNIST_TRAIN_LABELS,
                      constants.DatasetFileNames.MNIST_TEST_IMAGES,
                      constants.DatasetFileNames.MNIST_TEST_LABELS]

        files_exist = all(os.path.exists(constants.Paths.MNIST_DATA_READ) for file in data_files)

        if files_exist:
            return get_loaders_from_datasets(constants.Paths.MNIST_DATA_READ, constants.DatasetFileNames(),
                                             _dataset_name,
                                             _batch_size)
        else:
            return download_dataset(_dataset_name, _batch_size)

    elif _dataset_name == constants.DatasetNames.F_MNIST:
        data_files = [constants.DatasetFileNames.F_MNIST_TRAIN_IMAGES,
                      constants.DatasetFileNames.F_MNIST_TRAIN_LABELS,
                      constants.DatasetFileNames.F_MNIST_TEST_IMAGES,
                      constants.DatasetFileNames.F_MNIST_TEST_LABELS]

        files_exist = all(os.path.exists(file) for file in data_files)

        if files_exist:
            return get_loaders_from_datasets(constants.Paths.F_MNIST_DATA_READ, constants.DatasetFileNames(),
                                             _dataset_name,
                                             _batch_size)
        else:
            return download_dataset(_dataset_name, _batch_size)

    else:
        # CIFAR-10
        training_set = (
            read_image_file(os.path.join(constants.Paths.F_MNIST_DATA_READ, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(constants.Paths.F_MNIST_DATA_READ, 'train-labels-idx1-ubyte')),
        )
        test_set = (
            read_image_file(os.path.join(constants.Paths.F_MNIST_DATA_READ, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(constants.Paths.F_MNIST_DATA_READ, 't10k-labels-idx1-ubyte'))
        )


def download_dataset(_dataset_name: str, _batch_size: int) -> list[DataLoader]:
    """
    Downloads and loads the dataset.
    :param _dataset_name: Name of the dataset that needs to be downloaded.
    :param _batch_size: Batch of the data that needs to be downloaded.
    :return:
    """
    # Define transforms for the datasets
    transform_mnist = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.5,), (0.5,))])

    transform_cifar10 = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if _dataset_name == constants.DatasetNames.MNIST:
        trainset_mnist = datasets.MNIST(constants.Paths.MNIST_DATA_DOWNLOAD, download=True, train=True,
                                        transform=transform_mnist)
        trainloader = torch.utils.data.DataLoader(trainset_mnist, batch_size=_batch_size, shuffle=True)

        testset_mnist = datasets.MNIST(constants.Paths.MNIST_DATA_DOWNLOAD, download=True, train=False,
                                       transform=transform_mnist)
        testloader = torch.utils.data.DataLoader(testset_mnist, batch_size=_batch_size, shuffle=False)

    elif _dataset_name == constants.DatasetNames.F_MNIST:
        trainset_fmnist = datasets.FashionMNIST(constants.Paths.F_MNIST_DATA_DOWNLOAD, download=True,
                                                train=True,
                                                transform=transform_mnist)
        trainloader = torch.utils.data.DataLoader(trainset_fmnist, batch_size=_batch_size,
                                                  shuffle=True)

        testset_fmnist = datasets.FashionMNIST(constants.Paths.F_MNIST_DATA_DOWNLOAD, download=True,
                                               train=False,
                                               transform=transform_mnist)
        testloader = torch.utils.data.DataLoader(testset_fmnist, batch_size=_batch_size,
                                                 shuffle=False)

    else:
        # CIFAR-10
        trainset_cifar10 = datasets.CIFAR10(constants.Paths.CIFAR_10_DATA_DOWNLOAD, download=True,
                                            train=True,
                                            transform=transform_cifar10)
        trainloader = torch.utils.data.DataLoader(trainset_cifar10,
                                                  batch_size=_batch_size,
                                                  shuffle=True)

        testset_cifar10 = datasets.CIFAR10(constants.Paths.CIFAR_10_DATA_DOWNLOAD, download=True,
                                           train=False,
                                           transform=transform_cifar10)
        testloader = torch.utils.data.DataLoader(testset_cifar10, batch_size=_batch_size,
                                                 shuffle=False)

    return [trainloader, testloader]

# already downloaded datasets are read from the directory


# # Check the shape of the MNIST data
# dataiter_mnist = iter(trainloader_mnist)
# images_mnist, labels_mnist = next(dataiter_mnist)
# print(constants.Messages.MNIST_TRAINING_DATA_SHAPE, images_mnist.shape)
#
# # Check the shape of the Fashion MNIST data
# dataiter_fmnist = iter(trainloader_fmnist)
# images_fmnist, labels_fmnist = next(dataiter_fmnist)
# print(constants.Messages.F_MNIST_TRAINING_DATA_SHAPE, images_fmnist.shape)
#
# # Check the shape of the CIFAR-10 data
# dataiter_cifar10 = iter(trainloader_cifar10)
# images_cifar10, labels_cifar10 = next(dataiter_cifar10)
# print(constants.Messages.CIFAR_10_TRAINING_DATA_SHAPE, images_cifar10.shape)
