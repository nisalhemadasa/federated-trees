"""
This is the main entry point for the simulation. You can run this script

Author: Nisal Hemadasa
Date: 19-10-2024
Version: 1.0
"""
from federated_network.network import FederatedNetwork


def main():
    # Number of clients in the federated network
    num_client_instances = 10
    # Number of servers at each level of the server tree
    server_tree_layout = [1]
    # Creating the federated network instance
    fed_net = FederatedNetwork(num_client_instances, server_tree_layout)

    # Running the simulation
    fed_net.run_simulation()


if __name__ == "__main__":
    main()
