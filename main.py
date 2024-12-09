"""
This is the main entry point for the simulation. You can run this script

Author: Nisal Hemadasa
Date: 19-10-2024
Version: 1.0
"""
import constants
from federated_network.network import FederatedNetwork


def main():
    # # Create a drift object
    # drift = Drift()
    # Define the drift specifications
    drift_specifications = dict(
        clients_fraction=0.25,
        # Fraction of clients that are affected by the drift (literature also uses a list of fractions)
        is_synchronous=True,  # If the drift is synchronous or asynchronous
        drift_pattern=constants.DriftPatterns.GRADUAL,  # Drift pattern, i.e., abrupt, gradual, etc.
        drift_method=constants.DriftCreationMethods.ROTATION,
        # Drift creation method, i.e., label-swapping, rotations
        drift_start_round=0.25,  # Round at which the drift starts as a fraction of the total number of rounds
        drift_end_round=0.45,  # Round at which the drift ends as a fraction of the total number of rounds
        max_rotation=45,  # Maximum rotation angle for the drift created by rotations
        class_pairs_to_swap=[(1, 2), (5, 6)],  # Classes to be swapped in the label-swapping drift method
    )

    # Create a federated network
    fed_net = FederatedNetwork(
        num_client_instances=13,  # Number of clients in the federated network
        server_tree_layout=[8, 3, 2, 1],  # Number of servers at each level of the server tree of depth n = [n, n-1,..., 1]
        num_training_rounds=5,  # Number of training rounds (in literature, over 50 rounds are trained.
        # This is where the training accuracy begins to plateau without clustering)
        dataset_name=constants.DatasetNames.MNIST,  # Name of the dataset
        # drift_specs=None # Drift specifications
        drift_specs=drift_specifications,  # Drift specifications
        # drift=drift  # Drift object
        client_select_fraction=1  # Fraction of clients to be selected for each round
    )

    # Running the simulation
    fed_net.run_simulation()


if __name__ == "__main__":
    main()
