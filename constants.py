"""
Description: This constants used in this project

Author: Nisal Hemadasa
Date: 18-10-2024
Version: 1.0
"""


# Directory paths related to datasets
class Paths:
    # path to download and read the datasets
    DATASET = 'data/'

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
    # for client loss vs rounds plot
    CLIENT_LOSS_VS_ROUNDS_TITLE = 'Loss per Client Across Rounds'
    CLIENT_ACCURACY_VS_ROUNDS_TITLE = 'Accuracy per Client Across Rounds'
    CLIENT_LOSS_VS_ROUNDS_PNG = 'client_loss_vs_rounds.png'
    CLIENT_ACCURACY_VS_ROUNDS_PNG = 'client_accuracy_vs_rounds.png'
    # for server loss vs rounds plot
    SERVER_LOSS_VS_ROUNDS_TITLE = 'Loss per servers Across Rounds'
    SERVER_ACCURACY_VS_ROUNDS_TITLE = 'Accuracy per server Across Rounds'
    SERVER_LOSS_VS_ROUNDS_PNG = 'server_loss_vs_rounds.png'
    SERVER_ACCURACY_VS_ROUNDS_PNG = 'server_accuracy_vs_rounds.png'


# Drift patterns
class DriftPatterns:
    ABRUPT = 'abrupt'
    GRADUAL = 'gradual'
    INCREMENTAL = 'incremental'
    REOCCURRING = 'reoccurring'
    INCREMENTAL_ABRUPT = 'incre-abrupt'
    ABRUPT_REOCURRING = 'abrupt-reoc'
    INCREMENTAL_REOCCURRING = 'incr-reoc'
    OUT_OF_CONTROL = 'out-of-control'


# Drift creation methods
class DriftCreationMethods:
    LABEL_SWAPPING = 'label_swapping'
    ROTATION = 'rotation'


# Server hierarchical structures
class HierarchicalStructure:
    STRICT_BINARY_TREE = 'strict_binary_tree'
    RELAXED_BINARY_TREE = 'relaxed_binary_tree'
