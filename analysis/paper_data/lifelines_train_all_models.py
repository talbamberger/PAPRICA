"""
Lifelines Data Training Pipeline

NOTE: This code was run on the Lifelines server. Paths and data access are specific to the Lifelines cluster environment.

This script provides a command-line interface for training various neural network models
on Lifelines cohort multi-omic data, specifically microbiome and metabolome
data with fecal calprotectin as the target phenotype.

The script supports five different model architectures: 'd', 'y', 'x', 'pd' and 'pdp'.
For models description refer to the respective modules in the `src` directory.

Each model performs hyperparameter grid search with cross-validation and evaluates the performance across multiple repetitions.

Usage:
    python lifelines_train_all_models.py --model <model_name>

Example:
    python lifelines_train_all_models.py --model d
    python lifelines_train_all_models.py --model pdp
"""

import argparse
import src.d as d
import src.y as y
import src.x as x
import src.pd as pd
import src.pdp as pdp
import lifelines_data_utils

def parse_args():
    """
    Parse command line arguments to select the model to run.
    :return: The selected model as a string.
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

    model =  parse_args()

    # Parameters for the single networks (d, y, x)
    params_for_single_network = {
        'n_folds': [5], # Not hyperparameter, always 5
        'LR': [0.001, 0.0001], # Learning rate for the single network
        'batch_size': [32], # Batch size for training
        'N': [500, 1000], # Number of epochs
        'hidden_layers': [0], # Number of hidden layers in the encoder (and decoder)
        'latent_layer_size': [16, 32], # Size of the latent layer
        'rep': [i for i in range(10)], # Number of repetitions for the model stability
        'weight_decay': [1e-5], # Keep this value low to avoid overfitting
        'eps': [1e-8], # Epsilon for numerical stability, keep 1e-8
    }

    # All the parameters for the parallel models (pd and pdp)
    params_for_parallel_models = {
        'n_folds': [5],
        'LR1': [0.001, 0.0001],
        'LR2': [0.001], # Learning rates for the second model in case of parallel models
        'batch_size': [32],
        'N': [500, 1000],
        'hidden_layers1': [0],
        'hidden_layers2': [0], # Number of hidden layers in the second model
        'latent_layer_size1': [16, 32],
        'latent_layer_size2': [16], # Size of the latent layer in the second model
        'rep': [i for i in range(10)],
        'weight_decay': [1e-5],
        'eps': [1e-8],
    }

    omic1 = "microbiome"
    omic2 = "metabolome"
    phenotype_col = "age"
    run_identifier = "lifelines"  # Identifier for the run, can be used to save results in a specific folder
    output_path = "."
    num_processes = 10
    quiet_mode = False
    same_hyperparameters = True

    if model == 'd':

        d.grid_search(param_grid=params_for_single_network,
                    get_data_func=lifelines_data_utils.get_data,
                    omic1=omic1,
                    omic2=omic2,
                    phenotype_col=phenotype_col,
                    output_path=output_path,
                    run_identifier=run_identifier,
                    num_processes=num_processes,
                    quiet_mode = quiet_mode)
        
    elif model == 'y':
        y.grid_search(param_grid=params_for_single_network,
                    get_data_func=lifelines_data_utils.get_data,
                    omic1=omic1,
                    omic2=omic2,
                    phenotype_col=phenotype_col,
                    output_path=output_path,
                    run_identifier=run_identifier,
                    num_processes=num_processes,
                    quiet_mode = quiet_mode)

        
    elif model == 'x':
        x.grid_search(param_grid=params_for_single_network,
                      get_data_func=lifelines_data_utils.get_data,
                      omic1=omic1,
                      omic2=omic2,
                      phenotype_col=phenotype_col,
                      output_path=output_path,
                      run_identifier=run_identifier,
                      num_processes=num_processes,
                      quiet_mode=quiet_mode)


    elif model == 'pd':
        pd.grid_search(param_grid=params_for_parallel_models,
                    get_data_func=lifelines_data_utils.get_data,
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
                    get_data_func=lifelines_data_utils.get_data,
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
