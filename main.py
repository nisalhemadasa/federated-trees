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
        drift_pattern=constants.DriftPatterns.GRADUAL,  # Drift pattern, i.e., abrupt, gradual, etc.
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
        num_training_rounds=20,  # Number of training rounds (in literature, over 50 rounds are trained.
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
        file_save_path='./plots/saved_plots/paper/1_2_4_8/76_gradual_D0.375_L1_async_UTA_3/',
        log_save_path='./logs/saved_logs/paper/1_2_4_8/76_gradual_D0.375_L1_async_UTA_3/')

    # #################################
    # Async - DTA - D=0.375, L=1
    # ################################
    async_drift_specs = dict(
        num_drift_groups=2,  # Number of groups of clients that are affected by the drift asynchronously
        drift_groups=None,  # Groups of clients that are affected by the drift asynchronously
        drift_split_round=0.8,  # Times at which the drift is split into multiple asynchronous drifts,
    )
    # Define the drift specifications
    drift_specifications = dict(
        clients_fraction=0.375,
        # Fraction of clients that are affected by the drift (literature also uses a list of fractions)
        drift_localization_factor=1,  # Factor to localize the drift to a certain concentrated group of clients
        is_synchronous=False,  # If the drift is synchronous or asynchronous
        async_drift_specs=async_drift_specs,  # Specifications for the asynchronous case
        drift_pattern=constants.DriftPatterns.GRADUAL,  # Drift pattern, i.e., abrupt, gradual, etc.
        drift_method=constants.DriftCreationMethods.ROTATION,
        # Drift creation method, i.e., label-swapping, rotations
        drift_start_round=0.25,  # Round at which the drift starts as a fraction of the total number of rounds
        drift_end_round=0.5,  # Round at which the drift ends as a fraction of the total number of rounds
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
        is_server_downward_aggregation=True,  # Aggregates the server models along the downward links
    )

    # Create a federated network
    fed_net = FederatedNetwork(
        num_client_instances=32,  # Number of clients in the federated network
        server_tree_layout=[1, 2, 4, 8],
        # Number of servers at each level of the server tree of depth n = [n, n-1,..., 1]
        num_training_rounds=40,  # Number of training rounds (in literature, over 50 rounds are trained.
        dataset_name=constants.DatasetNames.MNIST,  # Name of the dataset
        drift_specs=drift_specifications,  # Drift specifications
        simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
        client_select_fraction=1,  # Fraction of clients to be selected for each round
    )

    # Running the simulation
    fed_net.run_simulation(
        file_save_path='./plots/saved_plots/paper/1_2_4_8/77_gradual_D0.375_L1_async_DTA_3/',
        log_save_path='./logs/saved_logs/paper/1_2_4_8/77_gradual_D0.375_L1_async_DTA_3/')
    l = 0

    # #################################
    # # Drift localization factor = 0.75
    # #################################
    print('------------')
    print('[1, 2, 4, 8] - 80_abrupt_D0.5_L0.75_UTA')
    print('------------')

    # Define the drift specifications
    drift_specifications = dict(
        clients_fraction=0.5,
        # Fraction of clients that are affected by the drift (literature also uses a list of fractions)
        drift_localization_factor=0.75,  # Factor to localize the drift to a certain concentrated group of clients
        is_synchronous=True,  # If the drift is synchronous or asynchronous
        async_drift_specs=async_drift_specs,  # Specifications for the asynchronous case
        drift_pattern=constants.DriftPatterns.ABRUPT,  # Drift pattern, i.e., abrupt, gradual, etc.
        drift_method=constants.DriftCreationMethods.LABEL_SWAPPING,
        # Drift creation method, i.e., label-swapping, rotations
        drift_start_round=0.25,  # Round at which the drift starts as a fraction of the total number of rounds
        drift_end_round=0.5,  # Round at which the drift ends as a fraction of the total number of rounds
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
        num_client_instances=32,  # Number of clients in the federated network
        server_tree_layout=[1, 2, 4, 8],
        # Number of servers at each level of the server tree of depth n = [n, n-1,..., 1]
        num_training_rounds=40,  # Number of training rounds (in literature, over 50 rounds are trained.
        dataset_name=constants.DatasetNames.MNIST,  # Name of the dataset
        drift_specs=drift_specifications,  # Drift specifications
        simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
        client_select_fraction=1,  # Fraction of clients to be selected for each round
    )

    # # Running the simulation
    # fed_net.run_simulation(file_save_path='./plots/saved_plots/paper/1_2_4_8/80_abrupt_D0.5_L0.75_UTA/',
    #                        log_save_path='./logs/saved_logs/paper/1_2_4_8/80_abrupt_D0.5_L0.75_UTA/')

    # #################################################################################################
    print('------------')
    print('[1, 2, 4, 8] - 81_abrupt_D0.5_L0.75_DTA')
    print('------------')

    # Define the drift specifications
    drift_specifications = dict(
        clients_fraction=0.5,
        # Fraction of clients that are affected by the drift (literature also uses a list of fractions)
        drift_localization_factor=0.75,  # Factor to localize the drift to a certain concentrated group of clients
        is_synchronous=True,  # If the drift is synchronous or asynchronous
        async_drift_specs=async_drift_specs,  # Specifications for the asynchronous case
        drift_pattern=constants.DriftPatterns.ABRUPT,  # Drift pattern, i.e., abrupt, gradual, etc.
        drift_method=constants.DriftCreationMethods.LABEL_SWAPPING,
        # Drift creation method, i.e., label-swapping, rotations
        drift_start_round=0.25,  # Round at which the drift starts as a fraction of the total number of rounds
        drift_end_round=0.5,  # Round at which the drift ends as a fraction of the total number of rounds
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
        is_server_downward_aggregation=True,  # Aggregates the server models along the downward links
    )

    # Create a federated network
    fed_net = FederatedNetwork(
        num_client_instances=32,  # Number of clients in the federated network
        server_tree_layout=[1, 2, 4, 8],
        # Number of servers at each level of the server tree of depth n = [n, n-1,..., 1]
        num_training_rounds=40,  # Number of training rounds (in literature, over 50 rounds are trained.
        dataset_name=constants.DatasetNames.MNIST,  # Name of the dataset
        drift_specs=drift_specifications,  # Drift specifications
        simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
        client_select_fraction=1,  # Fraction of clients to be selected for each round
    )

    # # Running the simulation
    # fed_net.run_simulation(file_save_path='./plots/saved_plots/paper/1_2_4_8/81_abrupt_D0.5_L0.75_DTA/',
    #                        log_save_path='./logs/saved_logs/paper/1_2_4_8/81_abrupt_D0.5_L0.75_DTA/')

    # #################################
    # # Drift localization factor = 1
    # #################################
    print('------------')
    print('[1, 2, 4, 8] - 82_abrupt_D0.375_L1_UTA')
    print('------------')

    # Define the drift specifications
    drift_specifications = dict(
        clients_fraction=0.375,
        # Fraction of clients that are affected by the drift (literature also uses a list of fractions)
        drift_localization_factor=1,  # Factor to localize the drift to a certain concentrated group of clients
        is_synchronous=True,  # If the drift is synchronous or asynchronous
        async_drift_specs=async_drift_specs,  # Specifications for the asynchronous case
        drift_pattern=constants.DriftPatterns.ABRUPT,  # Drift pattern, i.e., abrupt, gradual, etc.
        drift_method=constants.DriftCreationMethods.LABEL_SWAPPING,
        # Drift creation method, i.e., label-swapping, rotations
        drift_start_round=0.25,  # Round at which the drift starts as a fraction of the total number of rounds
        drift_end_round=0.5,  # Round at which the drift ends as a fraction of the total number of rounds
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
        num_client_instances=32,  # Number of clients in the federated network
        server_tree_layout=[1, 2, 4, 8],
        # Number of servers at each level of the server tree of depth n = [n, n-1,..., 1]
        num_training_rounds=40,  # Number of training rounds (in literature, over 50 rounds are trained.
        dataset_name=constants.DatasetNames.MNIST,  # Name of the dataset
        drift_specs=drift_specifications,  # Drift specifications
        simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
        client_select_fraction=1,  # Fraction of clients to be selected for each round
    )

    # # Running the simulation
    # fed_net.run_simulation(file_save_path='./plots/saved_plots/paper/1_2_4_8/82_abrupt_D0.375_L1_UTA/',
    #                        log_save_path='./logs/saved_logs/paper/1_2_4_8/82_abrupt_D0.375_L1_UTA/')
    # #################################################################################################
    print('------------')
    print('[1, 2, 4, 8] - 83_abrupt_D0.375_L1_DTA')
    print('------------')

    # Define the drift specifications
    drift_specifications = dict(
        clients_fraction=0.375,
        # Fraction of clients that are affected by the drift (literature also uses a list of fractions)
        drift_localization_factor=1,  # Factor to localize the drift to a certain concentrated group of clients
        is_synchronous=True,  # If the drift is synchronous or asynchronous
        async_drift_specs=async_drift_specs,  # Specifications for the asynchronous case
        drift_pattern=constants.DriftPatterns.ABRUPT,  # Drift pattern, i.e., abrupt, gradual, etc.
        drift_method=constants.DriftCreationMethods.LABEL_SWAPPING,
        # Drift creation method, i.e., label-swapping, rotations
        drift_start_round=0.25,  # Round at which the drift starts as a fraction of the total number of rounds
        drift_end_round=0.5,  # Round at which the drift ends as a fraction of the total number of rounds
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
        is_server_downward_aggregation=True,  # Aggregates the server models along the downward links
    )

    # Create a federated network
    fed_net = FederatedNetwork(
        num_client_instances=32,  # Number of clients in the federated network
        server_tree_layout=[1, 2, 4, 8],
        # Number of servers at each level of the server tree of depth n = [n, n-1,..., 1]
        num_training_rounds=40,  # Number of training rounds (in literature, over 50 rounds are trained.
        dataset_name=constants.DatasetNames.MNIST,  # Name of the dataset
        drift_specs=drift_specifications,  # Drift specifications
        simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
        client_select_fraction=1,  # Fraction of clients to be selected for each round
    )

    # Running the simulation
    fed_net.run_simulation(file_save_path='./plots/saved_plots/paper/1_2_4_8/83_abrupt_D0.375_L1_DTA/',
                           log_save_path='./logs/saved_logs/paper/1_2_4_8/83_abrupt_D0.375_L1_DTA/')

    l=0
    # #################################
    # # Drift localization factor = 0.75
    # #################################
    # print('------------')
    # print('[1, 2, 4, 8] - sub-scenario 7')
    # print('------------')
    # # Create a federated network
    # fed_net = FederatedNetwork(
    #     num_client_instances=32,  # Number of clients in the federated network
    #     server_tree_layout=[1, 2, 4, 8],
    #     # Number of servers at each level of the server tree of depth n = [n, n-1,..., 1]
    #     num_training_rounds=40,  # Number of training rounds (in literature, over 50 rounds are trained.
    #     dataset_name=constants.DatasetNames.MNIST,  # Name of the dataset
    #     drift_specs=drift_specifications,  # Drift specifications
    #     simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
    #     client_select_fraction=1,  # Fraction of clients to be selected for each round
    # )
    #
    # # Running the simulation
    # fed_net.run_simulation(file_save_path='./plots/saved_plots/paper/1_2_4_8/74_abrupt_driftfraction0.75_localizefactor0.75_UTA/',
    #                        log_save_path='./logs/saved_logs/paper/1_2_4_8/74_abrupt_driftfraction0.75_localizefactor0.75_UTA/')
    #
    # #################################################################################################
    # print('------------')
    # print('[1, 2, 4, 8] - sub-scenario 12')
    # print('------------')
    #
    # # Define simulation parameters
    # simulation_parameters = dict(
    #     is_server_adaptability=False,  # Evaluate the adaptability of servers/clients to the data/drift distribution
    #     is_download_from_root_server=False,  # Downloads the model from the root server of the server hierarchy
    #     is_download_from_level1_server=False,  # Downloads the model from the depth-level 1 server
    #     is_download_from_level2_server=False,  # Downloads the model from the  depth-level 2 server
    #     is_server_downward_aggregation=True,  # Aggregates the server models along the downward links
    # )
    #
    # # Create a federated network
    # fed_net = FederatedNetwork(
    #     num_client_instances=32,  # Number of clients in the federated network
    #     server_tree_layout=[1, 2, 4, 8],
    #     # Number of servers at each level of the server tree of depth n = [n, n-1,..., 1]
    #     num_training_rounds=40,  # Number of training rounds (in literature, over 50 rounds are trained.
    #     dataset_name=constants.DatasetNames.MNIST,  # Name of the dataset
    #     drift_specs=drift_specifications,  # Drift specifications
    #     simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
    #     client_select_fraction=1,  # Fraction of clients to be selected for each round
    # )
    #
    # # Running the simulation
    # fed_net.run_simulation(file_save_path='./plots/saved_plots/paper/1_2_4_8/75_abrupt_driftfraction0.75_localizefactor0.75_DTA/',
    #                        log_save_path='./logs/saved_logs/paper/1_2_4_8/75_abrupt_driftfraction0.75_localizefactor0.75_DTA/')

    #################################################################################################

    # print('------------')
    # print('[1] - 4 clients - 15')
    # print('------------')

    # # Create a federated network
    # fed_net = FederatedNetwork(
    #     num_client_instances=4,  # Number of clients in the federated network
    #     server_tree_layout=[1],
    #     # Number of servers at each level of the server tree of depth n = [n, n-1,..., 1]
    #     num_training_rounds=40,  # Number of training rounds (in literature, over 50 rounds are trained.
    #     dataset_name=constants.DatasetNames.MNIST,  # Name of the dataset
    #     drift_specs=drift_specifications,  # Drift specifications
    #     simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
    #     client_select_fraction=1,  # Fraction of clients to be selected for each round
    # )
    #
    # # Running the simulation
    # fed_net.run_simulation(file_save_path='./plots/saved_plots/paper/1/68_abrupt_4clients/',
    #                        log_save_path='./logs/saved_logs/paper/1/68_abrupt_4clients/')
    #
    # ###################################################################
    # print('------------')
    # print('[1] - 8 clients - 16')
    # print('------------')
    #
    # # Create a federated network
    # fed_net = FederatedNetwork(
    #     num_client_instances=8,  # Number of clients in the federated network
    #     server_tree_layout=[1],
    #     # Number of servers at each level of the server tree of depth n = [n, n-1,..., 1]
    #     num_training_rounds=40,  # Number of training rounds (in literature, over 50 rounds are trained.
    #     dataset_name=constants.DatasetNames.MNIST,  # Name of the dataset
    #     drift_specs=drift_specifications,  # Drift specifications
    #     simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
    #     client_select_fraction=1,  # Fraction of clients to be selected for each round
    # )
    #
    # # Running the simulation
    # fed_net.run_simulation(file_save_path='./plots/saved_plots/paper/1/69_abrupt_8clients/',
    #                        log_save_path='./logs/saved_logs/paper/1/69_abrupt_8clients/')
    #
    # ###################################################################
    #
    # print('------------')
    # print('[1] - 16 clients - 17')
    # print('------------')
    #
    # # Create a federated network
    # fed_net = FederatedNetwork(
    #     num_client_instances=16,  # Number of clients in the federated network
    #     server_tree_layout=[1],
    #     # Number of servers at each level of the server tree of depth n = [n, n-1,..., 1]
    #     num_training_rounds=40,  # Number of training rounds (in literature, over 50 rounds are trained.
    #     dataset_name=constants.DatasetNames.MNIST,  # Name of the dataset
    #     drift_specs=drift_specifications,  # Drift specifications
    #     simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
    #     client_select_fraction=1,  # Fraction of clients to be selected for each round
    # )
    #
    # # Running the simulation
    # fed_net.run_simulation(file_save_path='./plots/saved_plots/paper/1/70_abrupt_16clients/',
    #                        log_save_path='./logs/saved_logs/paper/1/70_abrupt_16clients/')

    # print('------------')
    # print('[1] - 1')
    # print('------------')
    #
    # # Create a federated network
    # fed_net = FederatedNetwork(
    #     num_client_instances=32,  # Number of clients in the federated network
    #     server_tree_layout=[1],
    #     # Number of servers at each level of the server tree of depth n = [n, n-1,..., 1]
    #     num_training_rounds=40,  # Number of training rounds (in literature, over 50 rounds are trained.
    #     dataset_name=constants.DatasetNames.MNIST,  # Name of the dataset
    #     drift_specs=drift_specifications,  # Drift specifications
    #     simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
    #     client_select_fraction=1,  # Fraction of clients to be selected for each round
    # )
    #
    # # Running the simulation
    # fed_net.run_simulation(file_save_path='./plots/saved_plots/paper/1/2_with_drift_abrupt/',
    #                        log_save_path='./logs/saved_logs/paper/1/2_with_drift_abrupt/')
    #
    # # Read and plot the log files
    # # plot_average_performance()
    #
    # #####################################################################
    # print('------------')
    # print('[1, 2] - 2')
    # print('------------')
    #
    # # Create a federated network
    # fed_net = FederatedNetwork(
    #     num_client_instances=32,  # Number of clients in the federated network
    #     server_tree_layout=[1, 2],
    #     # Number of servers at each level of the server tree of depth n = [n, n-1,..., 1]
    #     num_training_rounds=40,  # Number of training rounds (in literature, over 50 rounds are trained.
    #     dataset_name=constants.DatasetNames.MNIST,  # Name of the dataset
    #     drift_specs=drift_specifications,  # Drift specifications
    #     simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
    #     client_select_fraction=1,  # Fraction of clients to be selected for each round
    # )
    #
    # # Running the simulation
    # fed_net.run_simulation(file_save_path='./plots/saved_plots/paper/1_2/10_abrupt/',
    #                        log_save_path='./logs/saved_logs/paper/1_2/10_abrupt/')
    # #####################################################################
    # print('------------')
    # print('[1, 2, 4] - 3')
    # print('------------')
    #
    # # Create a federated network
    # fed_net = FederatedNetwork(
    #     num_client_instances=32,  # Number of clients in the federated network
    #     server_tree_layout=[1, 2, 4],
    #     # Number of servers at each level of the server tree of depth n = [n, n-1,..., 1]
    #     num_training_rounds=40,  # Number of training rounds (in literature, over 50 rounds are trained.
    #     dataset_name=constants.DatasetNames.MNIST,  # Name of the dataset
    #     drift_specs=drift_specifications,  # Drift specifications
    #     simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
    #     client_select_fraction=1,  # Fraction of clients to be selected for each round
    # )
    #
    # # Running the simulation
    # fed_net.run_simulation(file_save_path='./plots/saved_plots/paper/1_2_4/18_abrupt/',
    #                        log_save_path='./logs/saved_logs/paper/1_2_4/18_abrupt/')
    #
    # #####################################################################
    # print('------------')
    # print('[1, 2, 4, 8] - 4')
    # print('------------')
    #
    # # Create a federated network
    # fed_net = FederatedNetwork(
    #     num_client_instances=32,  # Number of clients in the federated network
    #     server_tree_layout=[1, 2, 4, 8],
    #     # Number of servers at each level of the server tree of depth n = [n, n-1,..., 1]
    #     num_training_rounds=40,  # Number of training rounds (in literature, over 50 rounds are trained.
    #     dataset_name=constants.DatasetNames.MNIST,  # Name of the dataset
    #     drift_specs=drift_specifications,  # Drift specifications
    #     simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
    #     client_select_fraction=1,  # Fraction of clients to be selected for each round
    # )
    #
    # # Running the simulation
    # fed_net.run_simulation(file_save_path='./plots/saved_plots/paper/1_2_4_8/26_abrupt/',
    #                        log_save_path='./logs/saved_logs/paper/1_2_4_8/26_abrupt/')
    #
    # #####################################################################
    # # Define simulation parameters
    # simulation_parameters = dict(
    #     is_server_adaptability=False,  # Evaluate the adaptability of servers/clients to the data/drift distribution
    #     is_download_from_root_server=True,  # Downloads the model from the root server of the server hierarchy
    #     is_download_from_level1_server=False,  # Downloads the model from the depth-level 1 server
    #     is_download_from_level2_server=False,  # Downloads the model from the  depth-level 2 server
    #     is_server_downward_aggregation=False,  # Aggregates the server models along the downward links
    # )
    #
    # print('------------')
    # print('[1] - 5')
    # print('------------')
    #
    # # Create a federated network
    # fed_net = FederatedNetwork(
    #     num_client_instances=32,  # Number of clients in the federated network
    #     server_tree_layout=[1],
    #     # Number of servers at each level of the server tree of depth n = [n, n-1,..., 1]
    #     num_training_rounds=40,  # Number of training rounds (in literature, over 50 rounds are trained.
    #     dataset_name=constants.DatasetNames.MNIST,  # Name of the dataset
    #     drift_specs=drift_specifications,  # Drift specifications
    #     simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
    #     client_select_fraction=1,  # Fraction of clients to be selected for each round
    # )
    #
    # # Running the simulation
    # fed_net.run_simulation(file_save_path='./plots/saved_plots/paper/1/6_abrupt_downloadfromroot/',
    #                        log_save_path='./logs/saved_logs/paper/1/6_abrupt_downloadfromroot/')
    #
    # # Read and plot the log files
    # # plot_average_performance()
    #
    # #####################################################################
    # print('------------')
    # print('[1, 2] - 6')
    # print('------------')
    #
    # # Create a federated network
    # fed_net = FederatedNetwork(
    #     num_client_instances=32,  # Number of clients in the federated network
    #     server_tree_layout=[1, 2],
    #     # Number of servers at each level of the server tree of depth n = [n, n-1,..., 1]
    #     num_training_rounds=40,  # Number of training rounds (in literature, over 50 rounds are trained.
    #     dataset_name=constants.DatasetNames.MNIST,  # Name of the dataset
    #     drift_specs=drift_specifications,  # Drift specifications
    #     simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
    #     client_select_fraction=1,  # Fraction of clients to be selected for each round
    # )
    #
    # # Running the simulation
    # fed_net.run_simulation(file_save_path='./plots/saved_plots/paper/1_2/43_abrupt_downloadfromroot/',
    #                        log_save_path='./logs/saved_logs/paper/1_2/43_abrupt_downloadfromroot/')
    # #####################################################################
    # print('------------')
    # print('[1, 2, 4] - 7')
    # print('------------')
    #
    # # Create a federated network
    # fed_net = FederatedNetwork(
    #     num_client_instances=32,  # Number of clients in the federated network
    #     server_tree_layout=[1, 2, 4],
    #     # Number of servers at each level of the server tree of depth n = [n, n-1,..., 1]
    #     num_training_rounds=40,  # Number of training rounds (in literature, over 50 rounds are trained.
    #     dataset_name=constants.DatasetNames.MNIST,  # Name of the dataset
    #     drift_specs=drift_specifications,  # Drift specifications
    #     simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
    #     client_select_fraction=1,  # Fraction of clients to be selected for each round
    # )
    #
    # # Running the simulation
    # fed_net.run_simulation(file_save_path='./plots/saved_plots/paper/1_2_4/45_abrupt_downloadfromroot/',
    #                        log_save_path='./logs/saved_logs/paper/1_2_4/45_abrupt_downloadfromroot/')
    #
    # #####################################################################
    # print('------------')
    # print('[1, 2, 4, 8] - 8')
    # print('------------')
    #
    # # Create a federated network
    # fed_net = FederatedNetwork(
    #     num_client_instances=32,  # Number of clients in the federated network
    #     server_tree_layout=[1, 2, 4, 8],
    #     # Number of servers at each level of the server tree of depth n = [n, n-1,..., 1]
    #     num_training_rounds=40,  # Number of training rounds (in literature, over 50 rounds are trained.
    #     dataset_name=constants.DatasetNames.MNIST,  # Name of the dataset
    #     drift_specs=drift_specifications,  # Drift specifications
    #     simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
    #     client_select_fraction=1,  # Fraction of clients to be selected for each round
    # )
    #
    # # Running the simulation
    # fed_net.run_simulation(file_save_path='./plots/saved_plots/paper/1_2_4_8/47_abrupt_downloadfromroot/',
    #                        log_save_path='./logs/saved_logs/paper/1_2_4_8/47_abrupt_downloadfromroot/')
    #
    #
    #
    # #####################################################################
    #
    # print('#####################################################################')
    # print('Download from levels')
    # print('#####################################################################')
    # print('------------')
    # print('level1 UTA - 9')
    # print('------------')
    #
    # # Define simulation parameters
    # simulation_parameters = dict(
    #     is_server_adaptability=False,  # Evaluate the adaptability of servers/clients to the data/drift distribution
    #     is_download_from_root_server=False,  # Downloads the model from the root server of the server hierarchy
    #     is_download_from_level1_server=True,  # Downloads the model from the depth-level 1 server
    #     is_download_from_level2_server=False,  # Downloads the model from the  depth-level 2 server
    #     is_server_downward_aggregation=False,  # Aggregates the server models along the downward links
    # )
    #
    # # Create a federated network
    # fed_net = FederatedNetwork(
    #     num_client_instances=32,  # Number of clients in the federated network
    #     server_tree_layout=[1, 2, 4, 8],
    #     # Number of servers at each level of the server tree of depth n = [n, n-1,..., 1]
    #     num_training_rounds=40,  # Number of training rounds (in literature, over 50 rounds are trained.
    #     dataset_name=constants.DatasetNames.MNIST,  # Name of the dataset
    #     drift_specs=drift_specifications,  # Drift specifications
    #     simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
    #     client_select_fraction=1,  # Fraction of clients to be selected for each round
    # )
    #
    # # Running the simulation
    # fed_net.run_simulation(file_save_path='./plots/saved_plots/paper/1_2_4_8/49_abrupt_downloadfromlevel1/',
    #                        log_save_path='./logs/saved_logs/paper/1_2_4_8/49_abrupt_downloadfromlevel1/')
    #
    # print('------------')
    # print('level2 UTA - 10')
    # print('------------')
    #
    # # Define simulation parameters
    # simulation_parameters = dict(
    #     is_server_adaptability=False,  # Evaluate the adaptability of servers/clients to the data/drift distribution
    #     is_download_from_root_server=False,  # Downloads the model from the root server of the server hierarchy
    #     is_download_from_level1_server=False,  # Downloads the model from the depth-level 1 server
    #     is_download_from_level2_server=True,  # Downloads the model from the  depth-level 2 server
    #     is_server_downward_aggregation=False,  # Aggregates the server models along the downward links
    # )
    #
    # # Create a federated network
    # fed_net = FederatedNetwork(
    #     num_client_instances=32,  # Number of clients in the federated network
    #     server_tree_layout=[1, 2, 4, 8],
    #     # Number of servers at each level of the server tree of depth n = [n, n-1,..., 1]
    #     num_training_rounds=40,  # Number of training rounds (in literature, over 50 rounds are trained.
    #     dataset_name=constants.DatasetNames.MNIST,  # Name of the dataset
    #     drift_specs=drift_specifications,  # Drift specifications
    #     simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
    #     client_select_fraction=1,  # Fraction of clients to be selected for each round
    # )
    #
    # # Running the simulation
    # fed_net.run_simulation(file_save_path='./plots/saved_plots/paper/1_2_4_8/50_abrupt_downloadfromlevel2/',
    #                        log_save_path='./logs/saved_logs/paper/1_2_4_8/50_abrupt_downloadfromlevel2/')
    #
    # print('------------')
    # print('level1 DTA - 11')
    # print('------------')
    #
    # # Define simulation parameters
    # simulation_parameters = dict(
    #     is_server_adaptability=False,  # Evaluate the adaptability of servers/clients to the data/drift distribution
    #     is_download_from_root_server=False,  # Downloads the model from the root server of the server hierarchy
    #     is_download_from_level1_server=True,  # Downloads the model from the depth-level 1 server
    #     is_download_from_level2_server=False,  # Downloads the model from the  depth-level 2 server
    #     is_server_downward_aggregation=True,  # Aggregates the server models along the downward links
    # )
    #
    # # Create a federated network
    # fed_net = FederatedNetwork(
    #     num_client_instances=32,  # Number of clients in the federated network
    #     server_tree_layout=[1, 2, 4, 8],
    #     # Number of servers at each level of the server tree of depth n = [n, n-1,..., 1]
    #     num_training_rounds=40,  # Number of training rounds (in literature, over 50 rounds are trained.
    #     dataset_name=constants.DatasetNames.MNIST,  # Name of the dataset
    #     drift_specs=drift_specifications,  # Drift specifications
    #     simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
    #     client_select_fraction=1,  # Fraction of clients to be selected for each round
    # )
    #
    # # Running the simulation
    # fed_net.run_simulation(file_save_path='./plots/saved_plots/paper/1_2_4_8/66_abrupt_downloadfromlevel1_DTA/',
    #                        log_save_path='./logs/saved_logs/paper/1_2_4_8/66_abrupt_downloadfromlevel1_DTA/')
    #
    # print('------------')
    # print('level2 DTA - 12')
    # print('------------')
    #
    # # Define simulation parameters
    # simulation_parameters = dict(
    #     is_server_adaptability=False,  # Evaluate the adaptability of servers/clients to the data/drift distribution
    #     is_download_from_root_server=False,  # Downloads the model from the root server of the server hierarchy
    #     is_download_from_level1_server=False,  # Downloads the model from the depth-level 1 server
    #     is_download_from_level2_server=True,  # Downloads the model from the  depth-level 2 server
    #     is_server_downward_aggregation=True,  # Aggregates the server models along the downward links
    # )
    #
    # # Create a federated network
    # fed_net = FederatedNetwork(
    #     num_client_instances=32,  # Number of clients in the federated network
    #     server_tree_layout=[1, 2, 4, 8],
    #     # Number of servers at each level of the server tree of depth n = [n, n-1,..., 1]
    #     num_training_rounds=40,  # Number of training rounds (in literature, over 50 rounds are trained.
    #     dataset_name=constants.DatasetNames.MNIST,  # Name of the dataset
    #     drift_specs=drift_specifications,  # Drift specifications
    #     simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
    #     client_select_fraction=1,  # Fraction of clients to be selected for each round
    # )
    #
    # # Running the simulation
    # fed_net.run_simulation(file_save_path='./plots/saved_plots/paper/1_2_4_8/67_abrupt_downloadfromlevel2_DTA/',
    #                        log_save_path='./logs/saved_logs/paper/1_2_4_8/67_abrupt_downloadfromlevel2_DTA/')
    #
    # print('#####################################################################')
    # print('DTA with tree growth')
    # print('#####################################################################')
    # # Define simulation parameters
    # simulation_parameters = dict(
    #     is_server_adaptability=False,  # Evaluate the adaptability of servers/clients to the data/drift distribution
    #     is_download_from_root_server=False,  # Downloads the model from the root server of the server hierarchy
    #     is_download_from_level1_server=False,  # Downloads the model from the depth-level 1 server
    #     is_download_from_level2_server=False,  # Downloads the model from the  depth-level 2 server
    #     is_server_downward_aggregation=True,  # Aggregates the server models along the downward links
    # )
    #
    # print('------------')
    # print('[1, 2, 4, 8] - 16')
    # print('------------')
    #
    # # Create a federated network
    # fed_net = FederatedNetwork(
    #     num_client_instances=32,  # Number of clients in the federated network
    #     server_tree_layout=[1, 2, 4, 8],
    #     # Number of servers at each level of the server tree of depth n = [n, n-1,..., 1]
    #     num_training_rounds=40,  # Number of training rounds (in literature, over 50 rounds are trained.
    #     dataset_name=constants.DatasetNames.MNIST,  # Name of the dataset
    #     drift_specs=drift_specifications,  # Drift specifications
    #     simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
    #     client_select_fraction=1,  # Fraction of clients to be selected for each round
    # )
    #
    # # Running the simulation
    # fed_net.run_simulation(file_save_path='./plots/saved_plots/paper/1_2_4_8/30_abrupt_DTA/',
    #                        log_save_path='./logs/saved_logs/paper/1_2_4_8/30_abrupt_DTA/')
    #
    # print('------------')
    # print('[1, 2] - 14')
    # print('------------')
    #
    # # Create a federated network
    # fed_net = FederatedNetwork(
    #     num_client_instances=32,  # Number of clients in the federated network
    #     server_tree_layout=[1, 2],
    #     # Number of servers at each level of the server tree of depth n = [n, n-1,..., 1]
    #     num_training_rounds=40,  # Number of training rounds (in literature, over 50 rounds are trained.
    #     dataset_name=constants.DatasetNames.MNIST,  # Name of the dataset
    #     drift_specs=drift_specifications,  # Drift specifications
    #     simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
    #     client_select_fraction=1,  # Fraction of clients to be selected for each round
    # )
    #
    # # Running the simulation
    # fed_net.run_simulation(file_save_path='./plots/saved_plots/paper/1_2/14_abrupt_DTA/',
    #                        log_save_path='./logs/saved_logs/paper/1_2/14_abrupt_DTA/')
    #
    # #####################################################################
    # print('------------')
    # print('[1, 2, 4] - 15')
    # print('------------')
    #
    # # Create a federated network
    # fed_net = FederatedNetwork(
    #     num_client_instances=32,  # Number of clients in the federated network
    #     server_tree_layout=[1, 2, 4],
    #     # Number of servers at each level of the server tree of depth n = [n, n-1,..., 1]
    #     num_training_rounds=40,  # Number of training rounds (in literature, over 50 rounds are trained.
    #     dataset_name=constants.DatasetNames.MNIST,  # Name of the dataset
    #     drift_specs=drift_specifications,  # Drift specifications
    #     simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
    #     client_select_fraction=1,  # Fraction of clients to be selected for each round
    # )
    #
    # # Running the simulation
    # fed_net.run_simulation(file_save_path='./plots/saved_plots/paper/1_2_4/22_abrupt_DTA/',
    #                        log_save_path='./logs/saved_logs/paper/1_2_4/22_abrupt_DTA/')

    ################################################################
    ################################################################
    # Fashion MNIST
    ################################################################
    ################################################################

    print('------------')
    print('[1] - 4 clients - 15')
    print('------------')

    # Create a federated network
    fed_net = FederatedNetwork(
        num_client_instances=4,  # Number of clients in the federated network
        server_tree_layout=[1],
        # Number of servers at each level of the server tree of depth n = [n, n-1,..., 1]
        num_training_rounds=40,  # Number of training rounds (in literature, over 50 rounds are trained.
        dataset_name=constants.DatasetNames.F_MNIST,  # Name of the dataset
        drift_specs=drift_specifications,  # Drift specifications
        simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
        client_select_fraction=1,  # Fraction of clients to be selected for each round
    )

    # Running the simulation
    fed_net.run_simulation(file_save_path='./plots/saved_plots/paper/fashionMNIST/1/68_abrupt_4clients/',
                           log_save_path='./logs/saved_logs/paper/fashionMNIST/1/68_abrupt_4clients/')

    ###################################################################
    print('------------')
    print('[1] - 8 clients - 16')
    print('------------')

    # Create a federated network
    fed_net = FederatedNetwork(
        num_client_instances=8,  # Number of clients in the federated network
        server_tree_layout=[1],
        # Number of servers at each level of the server tree of depth n = [n, n-1,..., 1]
        num_training_rounds=40,  # Number of training rounds (in literature, over 50 rounds are trained.
        dataset_name=constants.DatasetNames.F_MNIST,  # Name of the dataset
        drift_specs=drift_specifications,  # Drift specifications
        simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
        client_select_fraction=1,  # Fraction of clients to be selected for each round
    )

    # Running the simulation
    fed_net.run_simulation(file_save_path='./plots/saved_plots/paper/fashionMNIST/1/69_abrupt_8clients/',
                           log_save_path='./logs/saved_logs/paper/fashionMNIST/1/69_abrupt_8clients/')

    ###################################################################

    print('------------')
    print('[1] - 16 clients - 17')
    print('------------')

    # Create a federated network
    fed_net = FederatedNetwork(
        num_client_instances=16,  # Number of clients in the federated network
        server_tree_layout=[1],
        # Number of servers at each level of the server tree of depth n = [n, n-1,..., 1]
        num_training_rounds=40,  # Number of training rounds (in literature, over 50 rounds are trained.
        dataset_name=constants.DatasetNames.F_MNIST,  # Name of the dataset
        drift_specs=drift_specifications,  # Drift specifications
        simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
        client_select_fraction=1,  # Fraction of clients to be selected for each round
    )

    # Running the simulation
    fed_net.run_simulation(file_save_path='./plots/saved_plots/paper/fashionMNIST/1/70_abrupt_16clients/',
                           log_save_path='./logs/saved_logs/paper/fashionMNIST/1/70_abrupt_16clients/')

    print('------------')
    print('[1] - 1')
    print('------------')

    # Create a federated network
    fed_net = FederatedNetwork(
        num_client_instances=32,  # Number of clients in the federated network
        server_tree_layout=[1],
        # Number of servers at each level of the server tree of depth n = [n, n-1,..., 1]
        num_training_rounds=40,  # Number of training rounds (in literature, over 50 rounds are trained.
        dataset_name=constants.DatasetNames.F_MNIST,  # Name of the dataset
        drift_specs=drift_specifications,  # Drift specifications
        simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
        client_select_fraction=1,  # Fraction of clients to be selected for each round
    )

    # Running the simulation
    fed_net.run_simulation(file_save_path='./plots/saved_plots/paper/fashionMNIST/1/2_with_drift_abrupt/',
                           log_save_path='./logs/saved_logs/paper/fashionMNIST/1/2_with_drift_abrupt/')

    # Read and plot the log files
    # plot_average_performance()

    #####################################################################
    print('------------')
    print('[1, 2] - 2')
    print('------------')

    # Create a federated network
    fed_net = FederatedNetwork(
        num_client_instances=32,  # Number of clients in the federated network
        server_tree_layout=[1, 2],
        # Number of servers at each level of the server tree of depth n = [n, n-1,..., 1]
        num_training_rounds=40,  # Number of training rounds (in literature, over 50 rounds are trained.
        dataset_name=constants.DatasetNames.F_MNIST,  # Name of the dataset
        drift_specs=drift_specifications,  # Drift specifications
        simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
        client_select_fraction=1,  # Fraction of clients to be selected for each round
    )

    # Running the simulation
    fed_net.run_simulation(file_save_path='./plots/saved_plots/paper/fashionMNIST/1_2/10_abrupt/',
                           log_save_path='./logs/saved_logs/paper/fashionMNIST/1_2/10_abrupt/')
    #####################################################################
    print('------------')
    print('[1, 2, 4] - 3')
    print('------------')

    # Create a federated network
    fed_net = FederatedNetwork(
        num_client_instances=32,  # Number of clients in the federated network
        server_tree_layout=[1, 2, 4],
        # Number of servers at each level of the server tree of depth n = [n, n-1,..., 1]
        num_training_rounds=40,  # Number of training rounds (in literature, over 50 rounds are trained.
        dataset_name=constants.DatasetNames.F_MNIST,  # Name of the dataset
        drift_specs=drift_specifications,  # Drift specifications
        simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
        client_select_fraction=1,  # Fraction of clients to be selected for each round
    )

    # Running the simulation
    fed_net.run_simulation(file_save_path='./plots/saved_plots/paper/fashionMNIST/1_2_4/18_abrupt/',
                           log_save_path='./logs/saved_logs/paper/fashionMNIST/1_2_4/18_abrupt/')

    #####################################################################
    print('------------')
    print('[1, 2, 4, 8] - 4')
    print('------------')

    # Create a federated network
    fed_net = FederatedNetwork(
        num_client_instances=32,  # Number of clients in the federated network
        server_tree_layout=[1, 2, 4, 8],
        # Number of servers at each level of the server tree of depth n = [n, n-1,..., 1]
        num_training_rounds=40,  # Number of training rounds (in literature, over 50 rounds are trained.
        dataset_name=constants.DatasetNames.F_MNIST,  # Name of the dataset
        drift_specs=drift_specifications,  # Drift specifications
        simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
        client_select_fraction=1,  # Fraction of clients to be selected for each round
    )

    # Running the simulation
    fed_net.run_simulation(file_save_path='./plots/saved_plots/paper/fashionMNIST/1_2_4_8/26_abrupt/',
                           log_save_path='./logs/saved_logs/paper/fashionMNIST/1_2_4_8/26_abrupt/')

    #####################################################################
    # Define simulation parameters
    simulation_parameters = dict(
        is_server_adaptability=False,  # Evaluate the adaptability of servers/clients to the data/drift distribution
        is_download_from_root_server=True,  # Downloads the model from the root server of the server hierarchy
        is_download_from_level1_server=False,  # Downloads the model from the depth-level 1 server
        is_download_from_level2_server=False,  # Downloads the model from the  depth-level 2 server
        is_server_downward_aggregation=False,  # Aggregates the server models along the downward links
    )

    print('------------')
    print('[1] - 5')
    print('------------')

    # Create a federated network
    fed_net = FederatedNetwork(
        num_client_instances=32,  # Number of clients in the federated network
        server_tree_layout=[1],
        # Number of servers at each level of the server tree of depth n = [n, n-1,..., 1]
        num_training_rounds=40,  # Number of training rounds (in literature, over 50 rounds are trained.
        dataset_name=constants.DatasetNames.F_MNIST,  # Name of the dataset
        drift_specs=drift_specifications,  # Drift specifications
        simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
        client_select_fraction=1,  # Fraction of clients to be selected for each round
    )

    # Running the simulation
    fed_net.run_simulation(file_save_path='./plots/saved_plots/paper/fashionMNIST/1/6_abrupt_downloadfromroot/',
                           log_save_path='./logs/saved_logs/paper/fashionMNIST/1/6_abrupt_downloadfromroot/')

    # Read and plot the log files
    # plot_average_performance()

    #####################################################################
    print('------------')
    print('[1, 2] - 6')
    print('------------')

    # Create a federated network
    fed_net = FederatedNetwork(
        num_client_instances=32,  # Number of clients in the federated network
        server_tree_layout=[1, 2],
        # Number of servers at each level of the server tree of depth n = [n, n-1,..., 1]
        num_training_rounds=40,  # Number of training rounds (in literature, over 50 rounds are trained.
        dataset_name=constants.DatasetNames.F_MNIST,  # Name of the dataset
        drift_specs=drift_specifications,  # Drift specifications
        simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
        client_select_fraction=1,  # Fraction of clients to be selected for each round
    )

    # Running the simulation
    fed_net.run_simulation(file_save_path='./plots/saved_plots/paper/fashionMNIST/1_2/43_abrupt_downloadfromroot/',
                           log_save_path='./logs/saved_logs/paper/fashionMNIST/1_2/43_abrupt_downloadfromroot/')
    #####################################################################
    print('------------')
    print('[1, 2, 4] - 7')
    print('------------')

    # Create a federated network
    fed_net = FederatedNetwork(
        num_client_instances=32,  # Number of clients in the federated network
        server_tree_layout=[1, 2, 4],
        # Number of servers at each level of the server tree of depth n = [n, n-1,..., 1]
        num_training_rounds=40,  # Number of training rounds (in literature, over 50 rounds are trained.
        dataset_name=constants.DatasetNames.F_MNIST,  # Name of the dataset
        drift_specs=drift_specifications,  # Drift specifications
        simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
        client_select_fraction=1,  # Fraction of clients to be selected for each round
    )

    # Running the simulation
    fed_net.run_simulation(file_save_path='./plots/saved_plots/paper/fashionMNIST/1_2_4/45_abrupt_downloadfromroot/',
                           log_save_path='./logs/saved_logs/paper/fashionMNIST/1_2_4/45_abrupt_downloadfromroot/')

    #####################################################################
    print('------------')
    print('[1, 2, 4, 8] - 8')
    print('------------')

    # Create a federated network
    fed_net = FederatedNetwork(
        num_client_instances=32,  # Number of clients in the federated network
        server_tree_layout=[1, 2, 4, 8],
        # Number of servers at each level of the server tree of depth n = [n, n-1,..., 1]
        num_training_rounds=40,  # Number of training rounds (in literature, over 50 rounds are trained.
        dataset_name=constants.DatasetNames.F_MNIST,  # Name of the dataset
        drift_specs=drift_specifications,  # Drift specifications
        simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
        client_select_fraction=1,  # Fraction of clients to be selected for each round
    )

    # Running the simulation
    fed_net.run_simulation(file_save_path='./plots/saved_plots/paper/fashionMNIST/1_2_4_8/47_abrupt_downloadfromroot/',
                           log_save_path='./logs/saved_logs/paper/fashionMNIST/1_2_4_8/47_abrupt_downloadfromroot/')

    #####################################################################

    print('#####################################################################')
    print('Download from levels')
    print('#####################################################################')
    print('------------')
    print('level1 UTA - 9')
    print('------------')

    # Define simulation parameters
    simulation_parameters = dict(
        is_server_adaptability=False,  # Evaluate the adaptability of servers/clients to the data/drift distribution
        is_download_from_root_server=False,  # Downloads the model from the root server of the server hierarchy
        is_download_from_level1_server=True,  # Downloads the model from the depth-level 1 server
        is_download_from_level2_server=False,  # Downloads the model from the  depth-level 2 server
        is_server_downward_aggregation=False,  # Aggregates the server models along the downward links
    )

    # Create a federated network
    fed_net = FederatedNetwork(
        num_client_instances=32,  # Number of clients in the federated network
        server_tree_layout=[1, 2, 4, 8],
        # Number of servers at each level of the server tree of depth n = [n, n-1,..., 1]
        num_training_rounds=40,  # Number of training rounds (in literature, over 50 rounds are trained.
        dataset_name=constants.DatasetNames.F_MNIST,  # Name of the dataset
        drift_specs=drift_specifications,  # Drift specifications
        simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
        client_select_fraction=1,  # Fraction of clients to be selected for each round
    )

    # Running the simulation
    fed_net.run_simulation(
        file_save_path='./plots/saved_plots/paper/fashionMNIST/1_2_4_8/49_abrupt_downloadfromlevel1/',
        log_save_path='./logs/saved_logs/paper/fashionMNIST/1_2_4_8/49_abrupt_downloadfromlevel1/')

    print('------------')
    print('level2 UTA - 10')
    print('------------')

    # Define simulation parameters
    simulation_parameters = dict(
        is_server_adaptability=False,  # Evaluate the adaptability of servers/clients to the data/drift distribution
        is_download_from_root_server=False,  # Downloads the model from the root server of the server hierarchy
        is_download_from_level1_server=False,  # Downloads the model from the depth-level 1 server
        is_download_from_level2_server=True,  # Downloads the model from the  depth-level 2 server
        is_server_downward_aggregation=False,  # Aggregates the server models along the downward links
    )

    # Create a federated network
    fed_net = FederatedNetwork(
        num_client_instances=32,  # Number of clients in the federated network
        server_tree_layout=[1, 2, 4, 8],
        # Number of servers at each level of the server tree of depth n = [n, n-1,..., 1]
        num_training_rounds=40,  # Number of training rounds (in literature, over 50 rounds are trained.
        dataset_name=constants.DatasetNames.F_MNIST,  # Name of the dataset
        drift_specs=drift_specifications,  # Drift specifications
        simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
        client_select_fraction=1,  # Fraction of clients to be selected for each round
    )

    # Running the simulation
    fed_net.run_simulation(
        file_save_path='./plots/saved_plots/paper/fashionMNIST/1_2_4_8/50_abrupt_downloadfromlevel2/',
        log_save_path='./logs/saved_logs/paper/fashionMNIST/1_2_4_8/50_abrupt_downloadfromlevel2/')

    print('------------')
    print('level1 DTA - 11')
    print('------------')

    # Define simulation parameters
    simulation_parameters = dict(
        is_server_adaptability=False,  # Evaluate the adaptability of servers/clients to the data/drift distribution
        is_download_from_root_server=False,  # Downloads the model from the root server of the server hierarchy
        is_download_from_level1_server=True,  # Downloads the model from the depth-level 1 server
        is_download_from_level2_server=False,  # Downloads the model from the  depth-level 2 server
        is_server_downward_aggregation=True,  # Aggregates the server models along the downward links
    )

    # Create a federated network
    fed_net = FederatedNetwork(
        num_client_instances=32,  # Number of clients in the federated network
        server_tree_layout=[1, 2, 4, 8],
        # Number of servers at each level of the server tree of depth n = [n, n-1,..., 1]
        num_training_rounds=40,  # Number of training rounds (in literature, over 50 rounds are trained.
        dataset_name=constants.DatasetNames.F_MNIST,  # Name of the dataset
        drift_specs=drift_specifications,  # Drift specifications
        simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
        client_select_fraction=1,  # Fraction of clients to be selected for each round
    )

    # Running the simulation
    fed_net.run_simulation(
        file_save_path='./plots/saved_plots/paper/fashionMNIST/1_2_4_8/66_abrupt_downloadfromlevel1_DTA/',
        log_save_path='./logs/saved_logs/paper/fashionMNIST/1_2_4_8/66_abrupt_downloadfromlevel1_DTA/')

    print('------------')
    print('level2 DTA - 12')
    print('------------')

    # Define simulation parameters
    simulation_parameters = dict(
        is_server_adaptability=False,  # Evaluate the adaptability of servers/clients to the data/drift distribution
        is_download_from_root_server=False,  # Downloads the model from the root server of the server hierarchy
        is_download_from_level1_server=False,  # Downloads the model from the depth-level 1 server
        is_download_from_level2_server=True,  # Downloads the model from the  depth-level 2 server
        is_server_downward_aggregation=True,  # Aggregates the server models along the downward links
    )

    # Create a federated network
    fed_net = FederatedNetwork(
        num_client_instances=32,  # Number of clients in the federated network
        server_tree_layout=[1, 2, 4, 8],
        # Number of servers at each level of the server tree of depth n = [n, n-1,..., 1]
        num_training_rounds=40,  # Number of training rounds (in literature, over 50 rounds are trained.
        dataset_name=constants.DatasetNames.F_MNIST,  # Name of the dataset
        drift_specs=drift_specifications,  # Drift specifications
        simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
        client_select_fraction=1,  # Fraction of clients to be selected for each round
    )

    # Running the simulation
    fed_net.run_simulation(
        file_save_path='./plots/saved_plots/paper/fashionMNIST/1_2_4_8/67_abrupt_downloadfromlevel2_DTA/',
        log_save_path='./logs/saved_logs/paper/fashionMNIST/1_2_4_8/67_abrupt_downloadfromlevel2_DTA/')

    print('#####################################################################')
    print('DTA with tree growth')
    print('#####################################################################')
    # Define simulation parameters
    simulation_parameters = dict(
        is_server_adaptability=False,  # Evaluate the adaptability of servers/clients to the data/drift distribution
        is_download_from_root_server=False,  # Downloads the model from the root server of the server hierarchy
        is_download_from_level1_server=False,  # Downloads the model from the depth-level 1 server
        is_download_from_level2_server=False,  # Downloads the model from the  depth-level 2 server
        is_server_downward_aggregation=True,  # Aggregates the server models along the downward links
    )

    print('------------')
    print('[1, 2, 4, 8] - 16')
    print('------------')

    # Create a federated network
    fed_net = FederatedNetwork(
        num_client_instances=32,  # Number of clients in the federated network
        server_tree_layout=[1, 2, 4, 8],
        # Number of servers at each level of the server tree of depth n = [n, n-1,..., 1]
        num_training_rounds=40,  # Number of training rounds (in literature, over 50 rounds are trained.
        dataset_name=constants.DatasetNames.F_MNIST,  # Name of the dataset
        drift_specs=drift_specifications,  # Drift specifications
        simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
        client_select_fraction=1,  # Fraction of clients to be selected for each round
    )

    # Running the simulation
    fed_net.run_simulation(file_save_path='./plots/saved_plots/paper/fashionMNIST/1_2_4_8/30_abrupt_DTA/',
                           log_save_path='./logs/saved_logs/paper/fashionMNIST/1_2_4_8/30_abrupt_DTA/')

    print('------------')
    print('[1, 2] - 14')
    print('------------')

    # Create a federated network
    fed_net = FederatedNetwork(
        num_client_instances=32,  # Number of clients in the federated network
        server_tree_layout=[1, 2],
        # Number of servers at each level of the server tree of depth n = [n, n-1,..., 1]
        num_training_rounds=40,  # Number of training rounds (in literature, over 50 rounds are trained.
        dataset_name=constants.DatasetNames.F_MNIST,  # Name of the dataset
        drift_specs=drift_specifications,  # Drift specifications
        simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
        client_select_fraction=1,  # Fraction of clients to be selected for each round
    )

    # Running the simulation
    fed_net.run_simulation(file_save_path='./plots/saved_plots/paper/fashionMNIST/1_2/14_abrupt_DTA/',
                           log_save_path='./logs/saved_logs/paper/fashionMNIST/1_2/14_abrupt_DTA/')

    #####################################################################
    print('------------')
    print('[1, 2, 4] - 15')
    print('------------')

    # Create a federated network
    fed_net = FederatedNetwork(
        num_client_instances=32,  # Number of clients in the federated network
        server_tree_layout=[1, 2, 4],
        # Number of servers at each level of the server tree of depth n = [n, n-1,..., 1]
        num_training_rounds=40,  # Number of training rounds (in literature, over 50 rounds are trained.
        dataset_name=constants.DatasetNames.F_MNIST,  # Name of the dataset
        drift_specs=drift_specifications,  # Drift specifications
        simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
        client_select_fraction=1,  # Fraction of clients to be selected for each round
    )

    # Running the simulation
    fed_net.run_simulation(file_save_path='./plots/saved_plots/paper/fashionMNIST/1_2_4/22_abrupt_DTA/',
                           log_save_path='./logs/saved_logs/paper/fashionMNIST/1_2_4/22_abrupt_DTA/')


if __name__ == "__main__":
    main()
