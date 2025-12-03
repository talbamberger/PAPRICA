"""
Example Multi-Omic Model Training Pipeline

The script supports five different model architectures:
- 'd': Discriminative model (predicts phenotype from combined omics)
- 'y': Y-model (predicts phenotype from first omic, second omic from first omic)
- 'x': X-model (cross-modal prediction between omics)
- 'pd': Parallel Discriminative model (two separate networks for each omic)
- 'pdp': Parallel Discriminative with Phenotype model (extends pd with phenotype alginment)

Each model performs hyperparameter grid search with cross-validation using synthetic
but realistic multi-omic data.

Usage:
    python train_all_models.py --model <model_name>

Example:
    python train_all_models.py --model d
    python train_all_models.py --model pdp


"""

import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse
import src.d as d
import src.y as y
import src.x as x
import src.pd as pd
import src.pdp as pdp
import example_data_utils


def parse_args():
    """
    Parse command line arguments to select the model architecture to train.
    
    This function sets up the argument parser for the command-line interface,
    allowing users to specify which model architecture to train on synthetic data.
    
    Returns:
        str: The selected model name, one of ['d', 'y', 'x', 'pd', 'pdp'].
        
    Model Options: (see models descriptions in models directory)
        - 'd':
        - 'y': 
        - 'x': 
        - 'pd': 
        - 'pdp': 
        
    Raises:
        SystemExit: If invalid model name is provided or required argument is missing.
    """
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="A script that runs a specified model.")

    # Add arguments
    parser.add_argument('--model', type=str, required=True, 
                        choices=['d', 'y', 'x', 'pd', 'pdp'],
                        help="The model to use. Choose from: 'd', 'y', 'x', 'pd', 'pdp'.")

    # Parse the arguments
    args = parser.parse_args()
    return args.model
    


def main():
    """
    Main function that orchestrates the model training process.
    
    This function:
    1. Parses command line arguments to get the selected model
    2. Sets up hyperparameter grids for different model types
    3. Configures training parameters for synthetic data
    4. Calls the appropriate model's grid search function
    
    The function uses reduced parameter ranges suitable for quick testing
    with synthetic data, including fewer epochs and repetitions than would
    be used for real research.
    """
    
    model = parse_args()  # Get model from command line arguments

    # Parameters for the single networks (d, y, x)
    # Reduced parameters for quick testing with synthetic data
    params_for_single_network = {
        'n_folds': [5], # Not hyperparameter, always 5
        'LR': [0.001], # Learning rate for the single network
        'batch_size': [32], # Batch size for training
        'N': [50, 100], # Number of epochs (reduced for testing)
        'hidden_layers': [0], # Number of hidden layers in the encoder (and decoder)
        'latent_layer_size': [16], # Size of the latent layer
        'rep': [i for i in range(3)], # Number of repetitions (reduced for testing)
        'weight_decay': [1e-5], # Keep this value low to avoid overfitting
        'eps': [1e-8], # Epsilon for numerical stability, keep 1e-8
    }

    # All the parameters for the parallel models (pd and pdp)
    params_for_parallel_models = {
        'n_folds': [5],
        'LR1': [0.001],
        'LR2': [0.001], # Learning rates for the second model in case of parallel models
        'batch_size': [32],
        'N': [50, 100], # Number of epochs (reduced for testing)
        'hidden_layers1': [0],
        'hidden_layers2': [0], # Number of hidden layers in the second model
        'latent_layer_size1': [16],
        'latent_layer_size2': [16], # Size of the latent layer in the second model
        'rep': [i for i in range(3)],
        'weight_decay': [1e-5],
        'eps': [1e-8],
    }

    omic1 = "microbiome"
    omic2 = "metabolome"
    phenotype_col = "disease_progression"  # Example phenotype column, can be changed based on the dataset
    run_identifier = "example"  # Results will be saved under results/<run_identifier>/
    output_path = "."  # Output directory root
    num_processes = 10
    quiet_mode = False
    same_hyperparameters = True

    # Results will be saved in:
    # results/<run_identifier>/<model>_<omic1>_<omic2>_<phenotype_col>/
    # For example, with run_identifier="example", model="d", omic1="microbiome", omic2="metabolome", phenotype_col="disease_progression":
    # results/example/d_microbiome_metabolome_disease_progression/

    if model == 'd':
        d.grid_search(param_grid=params_for_single_network,
                    get_data_func=example_data_utils.get_data,
                    omic1=omic1,
                    omic2=omic2,
                    phenotype_col=phenotype_col,
                    output_path=output_path,
                    run_identifier=run_identifier,
                    num_processes=num_processes,
                    quiet_mode = quiet_mode)
        
    elif model == 'y':
        y.grid_search(param_grid=params_for_single_network,
                    get_data_func=example_data_utils.get_data,
                    omic1=omic1,
                    omic2=omic2,
                    phenotype_col=phenotype_col,
                    output_path=output_path,
                    run_identifier=run_identifier,
                    num_processes=num_processes,
                    quiet_mode = quiet_mode)

        
    elif model == 'x':
        x.grid_search(param_grid=params_for_single_network,
                      get_data_func=example_data_utils.get_data,
                      omic1=omic1,
                      omic2=omic2,
                      phenotype_col=phenotype_col,
                      output_path=output_path,
                      run_identifier=run_identifier,
                      num_processes=num_processes,
                      quiet_mode=quiet_mode)


    elif model == 'pd':
        pd.grid_search(param_grid=params_for_parallel_models,
                    get_data_func=example_data_utils.get_data,
                    omic1=omic1,
                    omic2=omic2,
                    phenotype_col=phenotype_col,
                    output_path=output_path,
                    run_identifier=run_identifier,
                    num_processes=num_processes,
                    quiet_mode = quiet_mode,
                    same_hyperparameters=same_hyperparameters)

    elif model == 'pdp':
        pdp.grid_search(param_grid=params_for_parallel_models,
                    get_data_func=example_data_utils.get_data,
                    omic1=omic1,
                    omic2=omic2,
                    phenotype_col=phenotype_col,
                    output_path=output_path,
                    run_identifier=run_identifier,
                    num_processes=num_processes,
                    quiet_mode = quiet_mode,
                    same_hyperparameters=same_hyperparameters)

if __name__ == "__main__":
    main()
