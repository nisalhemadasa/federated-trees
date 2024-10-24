"""
Description: This constants used in this project

Author: Nisal Hemadasa
Date: 18-10-2024
Version: 1.0
"""


# Directory paths related to datasets
class Paths:
    # path to download the datasets
    MNIST_DATA_DOWNLOAD = './MNIST_data/'
    F_MNIST_DATA_DOWNLOAD = './F_MNIST_data/'
    CIFAR_10_DATA_DOWNLOAD = './CIFAR10_data/'

    # path to read the already downloaded datasets
    MNIST_DATA_READ = 'data/MNIST_data/MNIST/raw/'
    F_MNIST_DATA_READ = 'data/F_MNIST_data/FashionMNIST/raw/'
    CIFAR_10_DATA_READ = 'data/CIFAR10_data/'

    # path related to saved plots
    PLOT_SAVE_PATH = './plots/saved_plots/'


# String related to miscellaneous messages
class MiscMessages:
    ACCURACY = "accuracy"


# Names of the dataset
class DatasetNames:
    MNIST = 'MNIST'
    F_MNIST = 'Fashion_MNIST'
    CIFAR_10 = 'CIFAR_10'


class DatasetFileNames:
    # MNIST dataset file names
    MNIST_TRAIN_IMAGES = 'train-images-idx3-ubyte'
    MNIST_TRAIN_LABELS = 'train-labels-idx1-ubyte'
    MNIST_TEST_IMAGES = 't10k-images-idx3-ubyte'
    MNIST_TEST_LABELS = 't10k-labels-idx1-ubyte'

    # F_MNIST dataset file names
    F_MNIST_TRAIN_IMAGES = 'train-images-idx3-ubyte'
    F_MNIST_TRAIN_LABELS = 'train-labels-idx1-ubyte'
    F_MNIST_TEST_IMAGES = 't10k-images-idx3-ubyte'
    F_MNIST_TEST_LABELS = 't10k-labels-idx1-ubyte'

    def get_train_images(self, dataset_name: str):
        if dataset_name == DatasetNames.MNIST:
            return [self.MNIST_TRAIN_IMAGES]
        elif dataset_name == DatasetNames.F_MNIST:
            return [self.F_MNIST_TRAIN_IMAGES]
        else:
            pass

    def get_train_labels(self, dataset_name: str):
        if dataset_name == DatasetNames.MNIST:
            return [self.MNIST_TRAIN_LABELS]
        elif dataset_name == DatasetNames.F_MNIST:
            return [self.F_MNIST_TRAIN_LABELS]
        else:
            pass

    def get_test_images(self, dataset_name: str):
        if dataset_name == DatasetNames.MNIST:
            return [self.MNIST_TEST_IMAGES]
        elif dataset_name == DatasetNames.F_MNIST:
            return [self.F_MNIST_TEST_IMAGES]
        else:
            pass

    def get_test_labels(self, dataset_name: str):
        if dataset_name == DatasetNames.MNIST:
            return [self.MNIST_TEST_LABELS]
        elif dataset_name == DatasetNames.F_MNIST:
            return [self.F_MNIST_TEST_LABELS]
        else:
            pass


# Strings related to plots
class Plots:
    NUMBER_OF_ROUNDS = 'Number of Rounds'
    LOSS = 'Loss'
    ACCURACY = 'Accuracy'
    LOSS_VS_ROUNDS_TITLE = 'Loss per Client Across Rounds'
    ACCURACY_VS_ROUNDS_TITLE = 'Accuracy per Client Across Rounds'
    LOSS_VS_ROUNDS_PNG = 'loss_vs_rounds.png'
    ACCURACY_VS_ROUNDS_PNG = 'accuracy_vs_rounds.png'
