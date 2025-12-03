
#######################################################################
#                             Pdp Model                               #
#######################################################################

# This file is part of the Pdp model.

# Pdp Model (Phenotype-Aware Parallel Representation for Integrative omiC Analysis - PAPRICA)
#
# The Pdp model is a parallel autoencoder-based architecture designed for multi-omic integration.
# It consists of separate autoencoders for each omic dataset, each trained to reconstruct its input
# while learning a distinct latent representation that preserves omic-specific structure.
#
# In addition to individual reconstruction losses, Pdp incorporates two coupling loss components:
# 1. A distance-based alignment loss that encourages samples with similar latent representations
#    in one omic to remain close in the other omics' latent spaces.
# 2. A phenotype-aware loss that aligns latent representations with pairwise distances in the
#    phenotype space, encouraging the learned representations to reflect phenotypic variation.
#
# This architecture enables biologically meaningful integration while preserving modality-specific patterns.
#
# This code can run from the train_all_models.py script, which is the main entry point for training
# the model.
# It performs a grid search over the hyperparameters and trains the model on the specified omics.
# This code outputs the results of the grid search to a CSV file in the specified output path.


#######################################################################
#                              Imports                                #
#######################################################################

import os
import random
from functools import partial
from itertools import product
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
import torch
from torch import device
from torch import nn
from torch.cuda import is_available
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from joblib import Parallel, delayed

# Local utilities
import src.models_utils as models_utils


#######################################################################
#                             TRAINING                                #
#######################################################################

def grid_search(param_grid, get_data_func, omic1, omic2, phenotype_col,
                output_path=".", run_identifier="", num_processes=8, quiet_mode=True,
                same_hyperparameters=False):
    """
     This function performs a grid search over the specified hyperparameters for the Pdp model.
     :param param_grid: The grid of hyperparameters to search over. It should be a dictionary where
     keys are hyperparameter names and values are lists of possible values for that hyperparameter.
     :param get_data_func: A function that retrieves the data for the specified omics and phenotype
     column. This function should return a tuple containing list of two dataframes (one for each
     omic) and a metadata dataframe.
     The omics dataframes should contain a "sample" column and the omic features, while the metadata
     dataframe should contain the "sample", "subject", and the phenotype_col columns.
     :param omic1: The first omic type (e.g., "microbiome").
     :param omic2: The second omic type (e.g., "metabolome").
     :param phenotype_col: The name of the column in the metadata that contains the phenotype to
     predict. Can be continuous or categorical.
     :param output_path: The path where the results will be saved.
     :param run_identifier: A string to identify the run, which will be appended to the results' path.
     :param num_processes: The number of parallel processes to use for the grid search. Default is 8.
     :param quiet_mode: If True, suppresses the output messages during the grid search.
     Default is True.
     :param same_hyperparameters: In case of True, the hyperparameters for the second omic will be
     set to be the same as the first omic. This is useful for comparing the performance of the
     models to single autoencoder model.
     :return: None. The results of the grid search will be saved to a CSV file in the specified
     output path.
     """

    output_path = f"{output_path}/results/{run_identifier}/pdp_{omic1}_{omic2}_{phenotype_col}"
    if not quiet_mode:
        print("Running Pdp (PAPRICA) model...")
        print(f"The results will be saved to: {output_path}")

    if not os.path.exists(f"{output_path}/"):
        os.makedirs(f"{output_path}")
        os.makedirs(f"{output_path}/loss")
        os.makedirs(f"{output_path}/pca")
        os.makedirs(f"{output_path}/cross_omic_predictions")
        os.makedirs(f"{output_path}/processed")

    param_combinations = list(enumerate(product(*param_grid.values())))
    random.shuffle(param_combinations)
    omics_df, metadata = get_data_func(omic1, omic2, phenotype_col)
    partial_func = partial(train_predict_cv_parallel, omic1=omic1, omic2=omic2, omics_df=omics_df,
                           metadata=metadata, phenotype_col=phenotype_col,
                           output_path=output_path, quiet_mode=quiet_mode,
                           same_hyperparameters=same_hyperparameters)
    results = Parallel(n_jobs=num_processes)(
        delayed(partial_func)(params) for params in param_combinations)

    # Save the results to a CSV file
    hyper_df = pd.DataFrame(results)
    if output_path:
        hyper_df.to_csv(
            f"{output_path}/final.csv")

    if not quiet_mode:
        print("Grid search completed. Results saved to "
              f"{output_path}/final.csv")


def train_predict_cv_parallel(params, omic1, omic2, omics_df, metadata, phenotype_col,
                              output_path, quiet_mode, same_hyperparameters):
    """
    This function trains the Pdp (PAPRICS) model on the given omics data and metadata.
    :param params: The hyperparameters to use for the model.
    It should be a tuple where the first element is the index of the hyperparameters and the second
    element is a list of hyperparameter values.
    :param omics_df: List of two dataframes containing the omics data for the two omics types.
    Each df must contain a "sample" column and the omic features.
    :param metadata: Metadata dataframe containing the phenotype_col,  "sample" and "subject"
    columns.
    :param phenotype_col: The name of the column in the metadata that contains the phenotype to
    predict. Can be continuous or categorical.
    :param output_path: The path where the results will be saved.
    :param quiet_mode: If True, suppresses the output messages during the training and prediction.
    :param same_hyperparameters: A boolean indicating whether to use the same hyperparameters
    for both omics. If True, the hyperparameters for the second omic will be set to be the same as
    the first omic.
    :return: A dictionary containing the results of the training and predictions for this set of
    hyperparameters.    """

    n_folds, LR1, LR2, batch_size, N, hidden_layers1, hidden_layers2, latent_layer_size1, \
        latent_layer_size2, rep, weight_decay, eps = params[1]

    # If 'same_hyperparameters' is True, set LR2, hidden_layers2, and latent_layer_size2
    # to be similar to LR1, hidden_layers1, and latent_layer_size1 respectively.
    if same_hyperparameters:
        LR2 = LR1
        hidden_layers2 = hidden_layers1
        latent_layer_size2 = latent_layer_size1

    df1 = omics_df[0]
    df2 = omics_df[1]

    n_input_features1 = df1.shape[1] - 1
    n_input_features2 = df2.shape[1] - 1


    ENCODER_1 = models_utils.build_encoder(hidden_layers1, latent_layer_size1, n_input_features1)
    ENCODER_2 = models_utils.build_encoder(hidden_layers2, latent_layer_size2, n_input_features2)
    subject_df = metadata.groupby('subject', as_index=False)[phenotype_col].mean()

    # Initialize the stratifies KFold object
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    kf_split = kf.split(range(subject_df.shape[0]))

    # Placeholders for results
    rf_orig_1 = []
    rf_orig_2 = []
    rf_latent_1 = []
    rf_latent_2 = []
    train_loss1, train_rec_loss1, train_omic_loss1, train_phenotype_loss1, train_weight_loss1 = [], [], [], [], []
    train_loss2, train_rec_loss2, train_omic_loss2, train_phenotype_loss2, train_weight_loss2 = [], [], [], [], []
    test_loss1, test_rec_loss1, test_omic_loss1, test_phenotype_loss1, test_weight_loss1 = [], [], [], [], []
    test_loss2, test_rec_loss2, test_omic_loss2, test_phenotype_loss2, test_weight_loss2 = [], [], [], [], []
    significant_correlations1, significant_correlations2 = [], []

    for fold, (train_index, test_index) in enumerate(kf_split):

        if not quiet_mode:
            print(f"Fold {fold + 1}/{n_folds} with parameters: "
                  f"LR1={LR1}, LR2={LR2}, batch_size={batch_size}, N={N}, "
                  f"hidden_layers1={hidden_layers1}, latent_layer_size1={latent_layer_size1}, "
                  f"hidden_layers2={hidden_layers2}, latent_layer_size2={latent_layer_size2}, "
                  f"rep={rep}")

        # Get the train and test subjects based on the indices
        subject_df['tmp_index'] = range(0, len(subject_df))

        # Map indices to subject IDs
        train_subjects = subject_df.loc[subject_df['tmp_index'].isin(train_index), 'subject'].values
        test_subjects = subject_df.loc[subject_df['tmp_index'].isin(test_index), 'subject'].values

        # Filter metadata and get sample IDs
        train_metadata = metadata[metadata["subject"].isin(train_subjects)]
        test_metadata = metadata[metadata["subject"].isin(test_subjects)]
        train_samples, test_samples = train_metadata["sample"].values, test_metadata[
            "sample"].values

        # Filter and clean df1 and df2
        train_df1, test_df1 = [df1[df1["sample"].isin(s)].drop(columns=["sample"]) for s in
                               [train_samples, test_samples]]
        train_df2, test_df2 = [df2[df2["sample"].isin(s)].drop(columns=["sample"]) for s in
                               [train_samples, test_samples]]

        # Make a string with the hyperparameters to use as a parameter for the model in output files
        params_string = f"f-{fold}_N-{N}_LR1-{LR2}_LR2-{LR2}_batchsize-{batch_size}_hiddenlayers1-{hidden_layers1}_hiddenlayers2-{hidden_layers2}_latentlayersize1-{latent_layer_size1}_latentlayersize2-{latent_layer_size2}_rep-{rep}"

        model1, model2, loss_dict =  init_and_train(omic1=omic1,
                                                    omic2=omic2,
                                                    train_df1=train_df1,
                                                    test_df1=test_df1,
                                                    train_df2=train_df2,
                                                    test_df2=test_df2,
                                                    train_metadata=train_metadata,
                                                    test_metadata=test_metadata,
                                                    phenotype_col=phenotype_col,
                                                    batch_size=batch_size,
                                                    ENCODER_1=ENCODER_1,
                                                    ENCODER_2=ENCODER_2,
                                                    criterion=Pdp_loss(),
                                                    N=N,
                                                    LR1=LR1,
                                                    LR2=LR2,
                                                    weight_decay=weight_decay,
                                                    eps=eps,
                                                    quiet_mode=quiet_mode)

        dev = device("cuda" if is_available() else "cpu")

        # Get the embeddings and decoded representations for the train and test dataframes
        data_dict = {
            name: apply_pdp_model(name, model1, model2, train_df1, test_df1, train_df2, test_df2)
            for name in ["Original", "Latent", "Decoded"]
        }

        # Evaluate cross omic prediction
        pred_output_path = f"{output_path}/cross_omic_predictions/first_omic_prediction_{params_string}.csv"
        n_predicted1 = cross_omic_prediction(data_dict, model1, 2, 1, pred_output_path)
        pred_output_path = f"{output_path}/cross_omic_predictions/second_omic_prediction_{params_string}.csv"
        n_predicted2 = cross_omic_prediction(data_dict, model2, 1, 2, pred_output_path)

        if not quiet_mode:

            # Plot the train and test loss
            fig, axs = plt.subplots(4, 5, figsize=(18, 12), gridspec_kw={'hspace': 0.3, 'wspace': 0.3})
            loss_types = ["loss", "rec_loss", "omic_loss", "phenotype_loss", "weight_loss"]
            splits = ["train", "train", "test", "test"]
            omics = ["1", "2", "1", "2"]
            row_labels = ["Training", "Training", "Test", "Test"]
            for row in range(4):
                for col, loss_type in enumerate(loss_types):
                    key = f"{splits[row]}_{loss_type}{omics[row]}"
                    title = f"{loss_type.replace('_', ' ').capitalize()} ({omics[row]})"
                    models_utils.plot_losses(axs[row, col], loss_dict[key], title=title)
            plt.savefig(
                f"{output_path}/loss/{params_string}.png")

        # Write loss values to txt files
        for loss_name in loss_dict:
            loss_values = loss_dict[loss_name]
            with open(f"{output_path}/loss/{loss_name}_{params_string}.txt", "w") as f:
                f.write("\n".join(map(str, loss_values)))

        # Predict the phenotype using Random Forest on the original and latent representations
        rf_dict1 = models_utils.get_rf_prediction(data_dict, train_metadata, test_metadata,
                                                 phenotype_col, 1)
        rf_dict2 = models_utils.get_rf_prediction(data_dict, train_metadata, test_metadata,
                                                 phenotype_col, 2)

        # Save the train and test data  for each representation
        processed_path = f"{output_path}/processed"
        for omic_i in [1, 2]:
            for split in ["train", "test"]:
                for space in ["Original", "Latent"]:
                    df = data_dict[space][f"{split}_df{omic_i}"]
                    filename = f"{split}_{space.lower()}{omic_i}_{params_string}.csv"
                    df.to_csv(f"{processed_path}/{filename}")

        # Plot PCA of the latent and original representations
        pca_out_path = f"{output_path}/pca"
        spaces = ["Original", "Latent", "Decoded"]
        omics = ["1", "2"]
        splits = ["train", "test"]
        # Individual plots per train and test
        for space in spaces:
            for omic in omics:
                for split in splits:
                    df = data_dict[space][f"{split}_df{omic}"]
                    meta = train_metadata if split == "train" else test_metadata
                    suffix = f"{split}{omic}"
                    plot_train_or_test_pca(pca_out_path, f"pca_{space.lower()}_{suffix}", df, meta,
                                           phenotype_col)
        # Joint plots for train and test
        for space in spaces:
            for omic in omics:
                plot_joint_pca(
                    pca_out_path,
                    f"pca_{space.lower()}_{omic}",
                    data_dict[space][f"train_df{omic}"], train_metadata,
                    data_dict[space][f"test_df{omic}"], test_metadata,
                    phenotype_col
                )

        # Save the results on this fold (Random Forest scores, loss values, and number of
        # significant correlations of the cross omic predictions)
        rf_orig_1.append(rf_dict1["Original1"])
        rf_orig_2.append(rf_dict2["Original2"])
        rf_latent_1.append(rf_dict1["Latent1"])
        rf_latent_2.append(rf_dict2["Latent2"])
        for k, v in loss_dict.items():
            locals()[k].append(v)
        significant_correlations1.append(n_predicted1)
        significant_correlations2.append(n_predicted2)

    # Generate a new row with the results to add to the final hyperparameter dataframe
    new_row = {"n_folds": n_folds,
               "LR1": LR1,
               "LR2": LR2,
               "batch_size": batch_size,
               "N": N,
               "hidden_layers_1": hidden_layers1,
               "hidden_layers_2": hidden_layers2,
               "latent_layer_size_1": latent_layer_size1,
               "latent_layer_size_2": latent_layer_size2,
               "weight_decay": weight_decay,
               "rep": rep,
               "rf_orig_cor1": np.mean(rf_orig_1),
               "rf_orig_cor2": np.mean(rf_orig_2),
               "rf_latent_cor1": np.mean(rf_latent_1),
               "rf_latent_cor2": np.mean(rf_latent_2),
               "train_loss1": np.mean(train_loss1),
               "train_rec_loss1": np.mean(train_rec_loss1),
               "train_omic_loss1": np.mean(train_omic_loss1),
               "train_phenotype_loss1": np.mean(train_phenotype_loss1),
               "train_weight_loss1": np.mean(train_weight_loss1),
               "train_rec_loss2": np.mean(train_rec_loss2),
               "train_omic_loss2": np.mean(train_omic_loss2),
               "train_phenotype_loss2": np.mean(train_phenotype_loss2),
               "train_weight_loss2": np.mean(train_weight_loss2),
               "test_loss1": np.mean(test_loss1),
               "test_rec_loss1": np.mean(test_rec_loss1),
               "test_omic_loss1": np.mean(test_omic_loss1),
               "test_phenotype_loss1": np.mean(test_phenotype_loss1),
               "test_weight_loss1": np.mean(test_weight_loss1),
               "test_loss2": np.mean(test_loss2),
               "test_rec_loss2": np.mean(test_rec_loss2),
               "test_omic_loss2": np.mean(test_omic_loss2),
               "test_phenotype_loss2": np.mean(test_phenotype_loss2),
               "test_weight_loss2": np.mean(test_weight_loss2),
               "significant_correlations1": np.mean(significant_correlations1),
               "significant_correlations2": np.mean(significant_correlations2),
               "number_of_features1": n_input_features1,
               "number_of_features2": n_input_features2
               }

    return new_row


def init_and_train(omic1,
                   omic2,
                   train_df1,
                   test_df1,
                   train_df2,
                   test_df2,
                   train_metadata,
                   test_metadata,
                   phenotype_col,
                   batch_size,
                   ENCODER_1,
                   ENCODER_2,
                   criterion,
                   N,
                   LR1,
                   LR2,
                   weight_decay,
                   eps,
                   quiet_mode):
    """
    This function initializes the Pdp models and trains it on the provided dataframes.
    :param omic1: The first omic type (e.g., "microbiome").
    :param omic2: The second omic type (e.g., "metabolome").
    :param train_df1: Train dataframe for the first omic.
    :param test_df1: Test dataframe for the first omic.
    :param train_df2: Train dataframe for the second omic.
    :param test_df2: Test dataframe for the second omic.
    :param train_metadata: The metadata for the training set, which includes the phenotype_col.
    :param test_metadata: The metadata for the test set, which includes the phenotype_col.
    :param phenotype_col: The name of the column in the metadata that contains the phenotype to
    predict. Can be continuous or categorical.
    :param batch_size: The batch size to use for training.
    :param ENCODER_1: The sizes of the encoder layers (list of integers), of the first omic model.
    :param ENCODER_2: The sizes of the encoder layers (list of integers), of the second omic model.
    :param criterion: Loss function to use for training the model.
    :param N: Number of epochs to train the model for.
    :param LR1: The learning rate for the optimizer, of the first omic model.
    :param LR2: The learning rate for the optimizer, of the second omic model.
    :param weight_decay: The weight decay (L2 regularization) for the optimizer.
    :param eps: The epsilon value for the optimizer to avoid division by zero.
    :param quiet_mode: Boolean flag to suppress output messages during training.
    :return: The model, training loss, and test loss.
    """

    # Set the seed for NumPy (used by PyTorch's DataLoader for shuffling)
    np.random.seed(42)

    # Init the DataLoader
    loader = DataLoader(dataset=Pdp_Dataset(train_df1, train_df2,
                     torch.tensor(train_metadata[phenotype_col].tolist()).unsqueeze(1)), batch_size=int(batch_size),
                        shuffle=True)

    # Init all the model parameters
    dev = device("cuda" if is_available() else "cpu")
    if not quiet_mode:
        print(f"Device: {dev}")
    model1 = Pdp(encoder_sizes=ENCODER_1, omic=omic1).to(dev)
    model2 = Pdp(encoder_sizes=ENCODER_2, omic=omic2).to(dev)

    # Apply the weight initialization to the model
    model1.apply(models_utils.initialize_weights)
    model2.apply(models_utils.initialize_weights)
    optimizer1 = Adam(model1.parameters(), lr=LR1, weight_decay=weight_decay, eps=eps)
    optimizer2 = Adam(model2.parameters(), lr=LR2, weight_decay=weight_decay, eps=eps)

    transform = Compose([ToTensor()])

    # Create dictionary to hold loss values
    loss_dict = {
        "train_loss1": [], "train_rec_loss1": [], "train_omic_loss1": [],
        "train_phenotype_loss1": [], "train_weight_loss1": [],
        "train_loss2": [], "train_rec_loss2": [], "train_omic_loss2": [],
        "train_phenotype_loss2": [], "train_weight_loss2": [],
        "test_loss1": [], "test_rec_loss1": [], "test_omic_loss1": [], "test_phenotype_loss1": [],
        "test_weight_loss1": [],
        "test_loss2": [], "test_rec_loss2": [], "test_omic_loss2": [], "test_phenotype_loss2": [],
        "test_weight_loss2": []
    }

    # Train loop
    for epoch in range(N):

        sum_train_loss1, sum_train_rec_loss1, sum_train_omic_loss1, sum_train_phenotype_loss1, sum_train_weight_loss1 = 0, 0, 0, 0, 0
        sum_train_loss2, sum_train_rec_loss2, sum_train_omic_loss2, sum_train_phenotype_loss2, sum_train_weight_loss2 = 0, 0, 0, 0, 0

        for batch_features1, batch_features2, batch_metadata in loader:

            batch_features1 = batch_features1.view(-1, batch_features1.shape[1]).to(dev)
            batch_features2 = batch_features2.view(-1, batch_features2.shape[1]).to(dev)

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            latent1, outputs1 = model1(batch_features1)
            latent2, outputs2 = model2(batch_features2)

            curr_train_loss1, curr_train_rec_loss1, curr_train_omic_loss1, \
                curr_train_phenotype_loss1, curr_train_weight_loss1 = criterion(batch_features1,
                                                                     latent1, latent2, outputs1,
                                                                     epoch, N, batch_metadata)
            curr_train_loss2, curr_train_rec_loss2, curr_train_omic_loss2, \
                curr_train_phenotype_loss2, curr_train_weight_loss2 = criterion(batch_features2,
                                                                     latent2, latent1, outputs2,
                                                                     epoch, N, batch_metadata)

            curr_train_loss1.backward(retain_graph=True)
            curr_train_loss2.backward()

            optimizer1.step()
            optimizer2.step()

            sum_train_loss1 += curr_train_loss1.item()
            sum_train_rec_loss1 += curr_train_rec_loss1.item()
            sum_train_omic_loss1 += curr_train_omic_loss1.item()
            sum_train_phenotype_loss1 += curr_train_phenotype_loss1.item()
            sum_train_weight_loss1 += curr_train_weight_loss1.item()

            sum_train_loss2 += curr_train_loss2.item()
            sum_train_rec_loss2 += curr_train_rec_loss2.item()
            sum_train_omic_loss2 += curr_train_omic_loss2.item()
            sum_train_phenotype_loss2 += curr_train_phenotype_loss2.item()
            sum_train_weight_loss2 += curr_train_weight_loss2.item()

        loss_dict["train_loss1"].append(sum_train_loss1 / len(loader))
        loss_dict["train_rec_loss1"].append(sum_train_rec_loss1 / len(loader))
        loss_dict["train_omic_loss1"].append(sum_train_omic_loss1 / len(loader))
        loss_dict["train_phenotype_loss1"].append(sum_train_phenotype_loss1 / len(loader))
        loss_dict["train_weight_loss1"].append(sum_train_weight_loss1 / len(loader))

        loss_dict["train_loss2"].append(sum_train_loss2 / len(loader))
        loss_dict["train_rec_loss2"].append(sum_train_rec_loss2 / len(loader))
        loss_dict["train_omic_loss2"].append(sum_train_omic_loss2 / len(loader))
        loss_dict["train_phenotype_loss2"].append(sum_train_phenotype_loss2 / len(loader))
        loss_dict["train_weight_loss2"].append(sum_train_weight_loss2 / len(loader))

        if not quiet_mode:
            print(f'epoch : {epoch + 1}/{N}, '
                  f'epoch loss1 = {loss_dict["train_loss1"][-1]:.6f}, '
                  f'epoch loss2 = {loss_dict["train_loss2"][-1]:.6f}')

        # Evaluate on test set
        test_tensor1 = torch.tensor(test_df1.values.astype(np.float32)).float()
        test_latent1, test_decoded1 = model1(test_tensor1.to(dev))

        test_tensor2 = torch.tensor(test_df2.values.astype(np.float32)).float()
        test_latent2, test_decoded2 = model2(test_tensor2.to(dev))

        curr_test_loss1, curr_test_rec_loss1, curr_test_omic_loss1, \
            curr_test_phenotype_loss1, curr_test_weight_loss1 = criterion(
                                                        test_tensor1.to(dev),
                                                        test_latent1, test_latent2,
                                                        test_decoded1,                                                         epoch, N,
                                                        torch.tensor(test_metadata[phenotype_col].tolist()).unsqueeze(1))

        curr_test_loss2, curr_test_rec_loss2, curr_test_omic_loss2, \
            curr_test_phenotype_loss2, curr_test_weight_loss2 = criterion(
                                                        test_tensor2.to(dev),
                                                        test_latent2, test_latent1,
                                                        test_decoded2,
                                                        epoch, N,
                                                        torch.tensor(test_metadata[phenotype_col].tolist()).unsqueeze(1))


        loss_dict["test_loss1"].append(curr_test_loss1.item())
        loss_dict["test_rec_loss1"].append(curr_test_rec_loss1.item())
        loss_dict["test_omic_loss1"].append(curr_test_omic_loss1.item())
        loss_dict["test_phenotype_loss1"].append(curr_test_phenotype_loss1.item())
        loss_dict["test_weight_loss1"].append(curr_test_weight_loss1.item())

        loss_dict["test_loss2"].append(curr_test_loss2.item())
        loss_dict["test_rec_loss2"].append(curr_test_rec_loss2.item())
        loss_dict["test_omic_loss2"].append(curr_test_omic_loss2.item())
        loss_dict["test_phenotype_loss2"].append(curr_test_phenotype_loss2.item())
        loss_dict["test_weight_loss2"].append(curr_test_weight_loss2.item())

    return model1, model2, loss_dict


#######################################################################
#                               MODEL                                 #
#######################################################################

class Pdp(torch.nn.Module):
    """
    Pdp (PAPRICA) model - a parallel autoencoder-based architecture for multi-omic integration.
    """
    def __init__(self, encoder_sizes, omic):
        super().__init__()
        self.encoder_sizes = encoder_sizes

        # Encoder
        enc_layers = []
        for i in range(len(encoder_sizes) - 1):
            enc_layers.append(
                nn.Linear(encoder_sizes[i], encoder_sizes[i + 1]))  # , bias=use_bias))
            enc_layers.append(nn.ReLU(inplace=True))  # inplace=True
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder
        dec_layers = []
        for i in reversed(range(1, len(encoder_sizes))):
            dec_layers.append(
                nn.Linear(encoder_sizes[i], encoder_sizes[i - 1]))  # , bias=use_bias))
            dec_layers.append(nn.ReLU(inplace=True))
        if omic != "microbiome": # TODO: think!
            dec_layers[-1] = nn.Tanh()
        self.decoder = nn.Sequential(*dec_layers[:-1])

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed


#######################################################################
#                           LOSS FUNCTIONS                            #
#######################################################################


class Pdp_loss(nn.Module):
    """
    Loss function for the Pdp (PAPRICA) model.
    """
    def __init__(self):
        super(Pdp_loss, self).__init__()

    def forward(self, batch_features, latent, latent_other,
                outputs, p, N, metadata):

        ############################# Reconstruction loss ###############################

        reconstruction_loss = torch.nn.functional.mse_loss(outputs, batch_features)

        ############################### Distance loss ################################

        # Calculate the sum of each column in the latent space
        column_sums = torch.sum(latent, dim=0)
        # Filter out columns with zero-sum
        non_zero_columns_mask = column_sums != 0
        # Filter the latent space to keep only non-zero columns
        latent_filtered = latent[:, non_zero_columns_mask]

        # Calculate the sum of each column in the other latent space
        latent_distances = torch.cdist(latent_filtered, latent_filtered, p=2)
        latent_n = latent_distances.size(0)
        # Get the upper triangle of the distance matrix
        latent_upper_triangle_mask = torch.triu(torch.ones(latent_n, latent_n,
                                                           dtype=torch.bool), diagonal=1)
        # Extract the upper triangle distances
        latent_upper_triangle = latent_distances[latent_upper_triangle_mask]

        # Repeat for the other omic
        other_distances = torch.cdist(latent_other, latent_other, p=2)
        other_n = other_distances.size(0)
        other_upper_triangle_mask = torch.triu(
            torch.ones(other_n, other_n, dtype=torch.bool), diagonal=1)
        other_upper_triangle = other_distances[other_upper_triangle_mask]

        # Repeat for the metadata
        metadata_distances = torch.cdist(metadata.float(), metadata.float(), p=2)
        metadata_n = metadata_distances.size(0)
        metadata_upper_triangle_mask = torch.triu(
            torch.ones(metadata_n, metadata_n, dtype=torch.bool),
            diagonal=1)
        metadata_upper_triangle = metadata_distances[metadata_upper_triangle_mask]

        # Calculate the Pearson correlation for the latent space, other omic, and metadata
        pearson_correlation_omics = pearson_for_loss(latent_upper_triangle.view(1, -1),
                                                       other_upper_triangle.view(1, -1))

        # Calculate the Pearson correlation between the latent space and the metadata
        pearson_correlation_this_omic_metadata = pearson_for_loss(
            latent_upper_triangle.view(1, -1),
            metadata_upper_triangle.view(1, -1))

        # Calculate the distance loss
        distance_loss = 1 - ((pearson_correlation_omics + pearson_correlation_this_omic_metadata) / 2)

        ############################## Final loss ##############################

        # Apply a sigmoid function to the regularization term
        sig_function = lambda p, N: 1 / (1 + np.exp((-10 / N) * (p - N / 2)))
        regularization = sig_function(p, N)
        regularization *= 0.5
        # Calculate the final loss as a weighted sum of the reconstruction and distance losses
        final_loss = ((1 - regularization) * reconstruction_loss) + (regularization * distance_loss)

        # Return the final loss, reconstruction loss, and the Pearson correlations
        return (final_loss, reconstruction_loss, 1 - pearson_correlation_omics,
                1 - pearson_correlation_this_omic_metadata, torch.tensor(regularization))

def pearson_for_loss(target, pred):
    return corrcoef(target, pred / pred.shape[-1])

def corrcoef(target, pred):
    # https://forum.numer.ai/t/custom-loss-functions-for-xgboost-using-pytorch/960
    pred_n = pred - pred.mean()
    target_n = target - target.mean()
    pred_n = pred_n / pred_n.norm()
    target_n = target_n / target_n.norm()
    # # Move both tensors to the same device
    dev = device("cuda" if is_available() else "cpu")
    pred_n, target_n = pred_n.to(dev), target_n.to(dev)
    return (pred_n * target_n).sum()



#######################################################################
#                             Data loader                             #
#######################################################################


class Pdp_Dataset(Dataset):
    """
    Custom dataset for the Pdp (PAPRICA) model.
    """

    def __init__(self, df1, df2, metadata):
        """
        Initializes the dataset with two dataframes.
        :param df1: Input dataframe (1st omic).
        :param df2: Output dataframe (2nd omic).
        :param metadata: Metadata dataframe containing the phenotype to predict.
        """
        self.train1 = torch.tensor(df1.values.astype(float), dtype=torch.float32)
        self.train2 = torch.tensor(df2.values.astype(float), dtype=torch.float32)
        self.train_metadata = metadata

    def __len__(self):
        """
        Returns the length of the dataset.
        :return: The number of samples in the dataset.
        """
        return self.train1.shape[0]

    def __getitem__(self, idx):
        """
        Returns the item at the specified index.
        :param idx: Given index.
        :return: Tuple containing the input dataframe, output dataframe, and metadata for the
        given index.
        """
        return self.train1[idx], self.train2[idx], self.train_metadata[idx]


#######################################################################
#                              TRAINING                               #
#######################################################################


def cross_omic_prediction(data_dict, model, input_omic_i, output_omic_i,pred_output_path):
    """
    This function performs cross-omic prediction by training a regression model on the latent space
    and then using the decoder of the second model to predict the output omic.
    :param data_dict: The dictionary containing the original, latent, and decoded dataframes for
    both omics.
    :param model: The models to use for the prediction (as decoder).
    :param input_omic_i: The index of the input omic (1 or 2).
    :param output_omic_i: The index of the output omic (1 or 2).
    :param pred_output_path: The path where the predictions will be saved.
    :return: Number of significant features with FDR < 0.1 and r > 0.3.
    """

    # Extract the original and latent dataframes for the input and output omics
    original_test_df_output_omic = data_dict["Original"][f"test_df{output_omic_i}"]
    latent_train_df_input_omic = data_dict["Latent"][f"train_df{input_omic_i}"]
    latent_train_df_output_omic = data_dict["Latent"][f"train_df{output_omic_i}"]
    latent_test_df_input_omic = data_dict["Latent"][f"test_df{input_omic_i}"]
    dev = device("cuda" if is_available() else "cpu")

    # Train a regression model to map latent_train_df_input_omic to latent_train_df_output_omic
    lr_model = LinearRegression().fit(latent_train_df_input_omic,  latent_train_df_output_omic)
    # Apply the transformation to latent_test_df_input_omic
    projected_latent_test_df_input_omic = lr_model.predict(latent_test_df_input_omic)
    # Convert to tensor
    projected_latent_test_tensor_input_omic = torch.tensor(projected_latent_test_df_input_omic, dtype=torch.float32)
    # Use the decoder of the output model to get predictions
    predicted_decoded = model.decoder(projected_latent_test_tensor_input_omic.to(dev)).cpu().detach().numpy()
    # Get the number of well predicted features
    return models_utils.get_omic_prediction(original_test_df_output_omic, pd.DataFrame(predicted_decoded), pred_output_path)




def apply_pdp_model(space, model1, model2, train_df1, test_df1, train_df2, test_df2):
    """
    This function applies the Pdp model to the given dataframes and returns the
    transformed dataframes.
    :param space: The space to apply the model to. Can be "Original", "Latent", or "Decoded".
    :param model1: The trained model (of the first omic) to apply.
    :param model2: The trained model (of the second omic)to apply.
    :param train_df1: The training dataframe of this first omic to apply the model to.
    :param test_df1: The test dataframe of this first omic to apply the model to.
    :param train_df2: The training dataframe of the second omic to apply the model to.
    :param test_df2: The test dataframe of the second omic to apply the model to.
    :return: The transformed train and test dataframes as a dictionary.
    """
    
    dev = device("cuda" if is_available() else "cpu")

    if space in ["Latent", "Decoded"]:
        train_latent1, train_decoded1 = model1(torch.Tensor(train_df1.values.astype(np.float32)).to(dev))
        train_df1 = pd.DataFrame(train_latent1.cpu().detach().numpy())
        train_latent2, train_decoded2 = model2(torch.Tensor(train_df2.values.astype(np.float32)).to(dev))
        train_df2 = pd.DataFrame(train_latent2.cpu().detach().numpy())

        test_latent1, test_decoded1 = model1(torch.Tensor(test_df1.values.astype(np.float32)).to(dev))
        test_df1 = pd.DataFrame(test_latent1.cpu().detach().numpy())
        test_latent2, test_decoded2 = model2(torch.Tensor(test_df2.values.astype(np.float32)).to(dev))
        test_df2 = pd.DataFrame(test_latent2.cpu().detach().numpy())

        if space == "Decoded":
            train_df1 = pd.DataFrame(train_decoded1.cpu().detach().numpy())
            train_df2 = pd.DataFrame(train_decoded2.cpu().detach().numpy())

            test_df1 = pd.DataFrame(test_decoded1.cpu().detach().numpy())
            test_df2 = pd.DataFrame(test_decoded2.cpu().detach().numpy())

    return {"train_df1": train_df1, "test_df1": test_df1,
            "train_df2": train_df2, "test_df2": test_df2}


def plot_train_or_test_pca(out_path, output_file, df, metadata, phenotype_col):
    """
    This function plots the PCA of the given dataframe and adds the regression direction.
    :param out_path: The output path where the plot will be saved.
    :param output_file: The name of the output file (without extension).
    :param df: The dataframe to plot (train or test).
    :param metadata: The metadata dataframe containing the target variable.
    :param phenotype_col: The metadata column name that contains the phenotype to predict.
    :return: Creates a PCA plot with regression direction and saves it to the specified path.
    """

    df = df.reset_index(drop=True)
    metadata = metadata.reset_index(drop=True)
    y = metadata[phenotype_col].values  # target variable

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(df)

    # Regression in original space
    reg = LinearRegression().fit(df, y)
    reg_dir = reg.coef_

    # Project regression direction into PCA space
    reg_pca_dir = pca.components_ @ reg_dir
    reg_pca_dir = reg_pca_dir / np.linalg.norm(reg_pca_dir)

    # Center and projections
    center = X_pca.mean(axis=0)
    projections = (X_pca - center) @ reg_pca_dir
    projected_points = np.outer(projections, reg_pca_dir) + center

    # Define regression line in PCA limits
    min_proj, max_proj = projections.min(), projections.max()
    line_range = np.linspace(min_proj, max_proj, 300)
    line_points = np.outer(line_range, reg_pca_dir) + center

    # Plot with GridSpec to add colorbar on the right (no upper plot)
    fig = plt.figure(figsize=(9, 10))
    gs = GridSpec(1, 2, width_ratios=[20, 1], wspace=0.1)

    ax_main = fig.add_subplot(gs[0, 0])
    ax_cbar = fig.add_subplot(gs[0, 1])  # colorbar on the right side

    # PCA with regression direction
    sc_main = ax_main.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6,
                              edgecolor='black', linewidth=0.5, s=50)

    # Colored regression line
    for i in range(len(line_points) - 1):
        color_val = (line_range[i] - min_proj) / (max_proj - min_proj)
        ax_main.plot(
            line_points[i:i+2, 0],
            line_points[i:i+2, 1],
            color=cm.viridis(color_val),
            linewidth=2
        )

    # Lines from data points to projections
    for i in range(len(X_pca)):
        ax_main.plot(
            [X_pca[i, 0], projected_points[i, 0]],
            [X_pca[i, 1], projected_points[i, 1]],
            color='gray', linestyle='--', alpha=0.3
        )

    ax_main.set_xlabel('PC 1')
    ax_main.set_ylabel('PC 2')

    # Colorbar
    cbar = fig.colorbar(sc_main, cax=ax_cbar)
    cbar.set_label(phenotype_col)

    # Remove grid and add black border to subplots
    for ax in [ax_main, ax_cbar]:
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(0.8)

    plt.savefig(f"{out_path}/{output_file}.png")
    plt.close()


def plot_joint_pca(out_path, output_file, train_df, train_metadata, test_df, test_metadata, phenotype_col):
    """
    This function plots the joint PCA of the training and test dataframes, including the regression
    direction and projections of both sets onto the regression line.
    :param out_path: The output path where the plot will be saved.
    :param output_file: The name of the output file (without extension).
    :param train_df: The training dataframe to plot.
    :param train_metadata: The metadata for the training dataframe.
    :param test_df: The test dataframe to plot.
    :param test_metadata: The metadata for the test dataframe.
    :param phenotype_col: The metadata column name that contains the phenotype to predict.
    :return: Save a PCA plot with regression direction and projections of both train and test sets.
    """
    train_df = train_df.reset_index(drop=True)
    train_metadata = train_metadata.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    test_metadata = test_metadata.reset_index(drop=True)

    X_train = train_df
    y_train = train_metadata[phenotype_col]
    X_test = test_df
    y_test = test_metadata[phenotype_col]

    # PCA on training data
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Linear regression in original space (training data only)
    reg = LinearRegression().fit(X_train, y_train)
    reg_dir = reg.coef_

    # Project regression direction into PCA space
    reg_pca_dir = pca.components_ @ reg_dir
    reg_pca_dir = reg_pca_dir / np.linalg.norm(reg_pca_dir)

    # Center (from training data)
    center = X_train_pca.mean(axis=0)

    # Projections of train and test on regression line in PCA space
    proj_train = (X_train_pca - center) @ reg_pca_dir
    proj_test = (X_test_pca - center) @ reg_pca_dir
    projected_train = np.outer(proj_train, reg_pca_dir) + center
    projected_test = np.outer(proj_test, reg_pca_dir) + center

    # Define regression line across full range of projections (both train and test)
    all_proj = np.concatenate([proj_train, proj_test])
    min_proj, max_proj = all_proj.min(), all_proj.max()
    line_range = np.linspace(min_proj, max_proj, 300)
    line_points = np.outer(line_range, reg_pca_dir) + center

    # Plot with GridSpec (no upper plot)
    fig = plt.figure(figsize=(9, 10))
    gs = GridSpec(1, 2, width_ratios=[20, 1], wspace=0.1)

    ax_main = fig.add_subplot(gs[0, 0])
    ax_cbar = fig.add_subplot(gs[0, 1])  # Colorbar on the right

    # Bottom plot: PCA with regression direction
    sc_main_train = ax_main.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='viridis',
                                    edgecolor='black', linewidth=0.5, s=60, alpha=0.6, label='Train', zorder=2)
    sc_main_test = ax_main.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap='viridis',
                                   edgecolor='black', linewidth=0.5, s=60, marker='^', alpha=0.6, label='Test', zorder=2)

    # Regression line
    for i in range(len(line_points) - 1):
        color_val = (line_range[i] - min_proj) / (max_proj - min_proj)
        ax_main.plot(
            line_points[i:i+2, 0],
            line_points[i:i+2, 1],
            color=cm.viridis(color_val),
            linewidth=2,
            zorder=1
        )

    # Lines from points to projections
    for i in range(len(X_train_pca)):
        ax_main.plot(
            [X_train_pca[i, 0], projected_train[i, 0]],
            [X_train_pca[i, 1], projected_train[i, 1]],
            color='gray', linestyle='--', alpha=0.3, zorder=0
        )
    for i in range(len(X_test_pca)):
        ax_main.plot(
            [X_test_pca[i, 0], projected_test[i, 0]],
            [X_test_pca[i, 1], projected_test[i, 1]],
            color='gray', linestyle='--', alpha=0.3, zorder=0
        )

    ax_main.set_xlabel('PC 1')
    ax_main.set_ylabel('PC 2')

    # Colorbar (can use either sc_main_train or sc_main_test since colormap is shared)
    cbar = fig.colorbar(sc_main_train, cax=ax_cbar)
    cbar.set_label(phenotype_col)

    # Remove grid and add black border to subplots
    for ax in [ax_main, ax_cbar]:
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(0.8)

    # plt.tight_layout()
    plt.savefig(f"{out_path}/{output_file}.png")
    plt.close()

