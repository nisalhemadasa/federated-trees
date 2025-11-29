"""
This is the main entry point for the simulation. You can run this script

Author: Nisal Hemadasa
Date: 19-10-2024
Version: 1.0
"""
import constants
from federated_network.network import FederatedNetwork
from logs.analysis_functions import plot_average_performance

def main():
    async_drift_specs = dict(
        num_drift_groups=2,  # Number of groups of clients that are affected by the drift asynchronously
        drift_groups=None,  # Groups of clients that are affected by the drift asynchronously
        drift_split_round=0.8,  # Times at which the drift is split into multiple asynchronous drifts,
    )
    # Define the drift specifications
    drift_specifications = dict(
        clients_fraction=0.375,
        # clients_fraction=0.7,
        # Fraction of clients that are affected by the drift (literature also uses a list of fractions)
        drift_localization_factor=1,  # Factor to localize the drift to a certain concentrated group of clients
        is_synchronous=False,  # If the drift is synchronous or asynchronous
        async_drift_specs=async_drift_specs,  # Specifications for the asynchronous case
        drift_pattern=constants.DriftPatterns.INCREMENTAL_REOCCURRING,  # Drift pattern, i.e., abrupt, gradual, etc.
        drift_method=constants.DriftCreationMethods.ROTATION,
        # Drift creation method, i.e., label-swapping, rotations
        drift_start_round=0.5,  # Round at which the drift starts as a fraction of the total number of rounds
        drift_end_round=0.75,  # Round at which the drift ends as a fraction of the total number of rounds
        max_rotation=45,  # Maximum rotation angle for the drift created by rotations
        class_pairs_to_swap=[(1, 2), (5, 6)],  # Classes to be swapped in the label-swapping drift method
        # class_pairs_to_swap=[('Sandal', 'Shirt'), ('Trouser', 'Bag')],  # Classes to be swapped in F_MNIST
    )

    # Define simulation parameters
    simulation_parameters = dict(
        is_server_adaptability=False,  # Evaluate the adaptability of servers/clients to the data/drift distribution
        is_download_from_root_server=False,  # Downloads the model from the root server of the server hierarchy
        is_download_from_level1_server=False,  # Downloads the model from the depth-level 1 server
        is_download_from_level2_server=False,  # Downloads the model from the  depth-level 2 server
        is_server_downward_aggregation=False,  # Aggregates the server models along the downward links
    )

    # Create a federated network
    fed_net = FederatedNetwork(
        num_client_instances=16,  # Number of clients in the federated network
        server_tree_layout=[1, 2],
        # Number of servers at each level of the server tree of depth n = [n, n-1,..., 1]
        num_training_rounds=50,  # Number of training rounds (in literature, over 50 rounds are trained.
        dataset_name=constants.DatasetNames.MNIST,  # Name of the dataset
        drift_specs=drift_specifications,  # Drift specifications
        simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
        client_select_fraction=1,  # Fraction of clients to be selected for each round
    )

    # #################################
    # Async - UTA - D=0.375, L=1
    # #################################

    # Running the simulation
    fed_net.run_simulation(
        file_save_path='./plots/saved_plots/paper/1_2_4_8/76_incr_reocc_D0.375_L1_async_UTA_3/',
        log_save_path='./logs/saved_logs/paper/1_2_4_8/76_incr_reocc_D0.375_L1_async_UTA_3/')


if __name__ == "__main__":
    main()
