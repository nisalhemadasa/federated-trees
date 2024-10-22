"""
This is the main entry point for the simulation. You can run this script

Author: Nisal Hemadasa
Date: 19-10-2024
Version: 1.0
"""
import constants
from federated_network.network import FederatedNetwork


def main():
    # Create a federated network
    fed_net = FederatedNetwork(
        num_client_instances=10,     # Number of clients in the federated network
        server_tree_layout=[1],      # Number of servers at each level of the server tree
        num_training_rounds=10,        # Number of training rounds
        dataset_name=constants.DatasetNames.MNIST       # Name of the dataset
    )

    # Running the simulation
    fed_net.run_simulation()


if __name__ == "__main__":
    main()
