"""
This is the main entry point for the simulation. You can run this script

Author: Nisal Hemadasa
Date: 19-10-2024
Version: 1.0
"""
from federated_network.network import FederatedNetwork


def main():
    # Cretaing the federated network instance
    fed_net = FederatedNetwork()

    # Running the simulation
    fed_net.run_simulation()


if __name__ == "__main__":
    main()
