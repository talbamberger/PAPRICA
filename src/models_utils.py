from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from statsmodels.stats.multitest import multipletests

import torch.nn as nn


def initialize_weights(m):
    """
    This function initializes the weights of a linear layer using Xavier uniform initialization
    :param m: The layer to initialize.
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def plot_losses(ax,
                loss_list,
                title,
                color="navy"):
    """
    This function plots the losses during training (of the train or the test set).
    :param ax: The axis to plot the losses on.
    :param loss_list: The list of losses to plot.
    :param criterion_name: The name of the criterion used for the losses (e.g., "D Loss").
    :param title: The title of the plot.
    :param color: The color of the plot line.
    :return: A plot of the losses.
    """
    ax.plot(loss_list, color=color)
    ax.set_title(f'{title}')
    ax.set_xlabel('Epochs')
    ax.set_ylabel("Loss")

def get_rf_prediction(data_dict, train_metadata, test_metadata, phenotype_col, i=""):
    """
    This function trains a Random Forest model to predict the phenotype_col
    on the original, latent, and decoded representations of the data
    :param data_dict: Dictionary containing the train and test dataframes for each space
    ("Original", "Latent"), resulting from the trained model.
    :param train_metadata: The metadata for the training set, which includes the phenotype_col.
    :param test_metadata: The metadata for the test set, which includes the phenotype_col.
    :param phenotype_col: The name of the column in the metadata that contains the phenotype to
    predict. Expected to be continuous but can also be categorical.
    :param i: Model identifier in case of multiple latent representations (as in the X, Pd and Pdp
    models, default is an empty string that match the d and y models).
    :return: Dictionary containing the Random Forest scores for each space. In case of a continuous
    phenotype, the score is the Spearman correlation between the predicted and true values. In case
    of a categorical phenotype, the score is the ROC AUC score.
    """
    rf_dict = {}

    for space in ["Original", "Latent"]:
        y_train = train_metadata[phenotype_col]
        y_test = test_metadata[phenotype_col]

        X_train = data_dict[space][f"train_df{i}"]
        X_test = data_dict[space][f"test_df{i}"]

        # In case the phenotype_col is categorical, perform classification
        if len(train_metadata[phenotype_col].unique()) == 2:
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            rf_score = roc_auc_score(y_test, y_pred)
        # Otherwise, perform regression (as expected)
        else:
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            rf_score = spearmanr(y_pred, y_test)[0]

        rf_dict[f"{space}{i}"] = rf_score

    return rf_dict


def get_omic_prediction(real_omic, predicted_omic, pred_output_path):
    """
    This function calculates the Spearman correlation between the real and predicted values given
     omic and saves the results to a CSV file.
    :param real_omic: The real values of the omic.
    :param predicted_omic: The predicted values of the omic.
    :param pred_output_path: The path to save the results CSV file.
    :return: Number of significant features with FDR < 0.1 and r > 0.3.
    """

    # Store correlations and corresponding column names

    correlations = []
    p_values = []

    for col_idx in range(real_omic.shape[1]):
        correlation, p_value = spearmanr(real_omic.iloc[:, col_idx],
                                         predicted_omic.iloc[:, col_idx])
        correlations.append(correlation)
        p_values.append(p_value)

    # Perform FDR correction
    fdr_corrected = multipletests(p_values, method='fdr_bh')[1]

    # Identify significant correlations with FDR < 0.1 and r > 0.3
    significant_indices = [i for i, (r, fdr) in enumerate(zip(correlations, fdr_corrected)) if \
                           r > 0.3 and fdr < 0.1]
    num_significant = len(significant_indices)

    # Create a DataFrame with all results
    results_df = pd.DataFrame({
        "Feature": real_omic.columns.tolist(),
        "Spearman_r": correlations,
        "P_value": p_values,
        "FDR": fdr_corrected
    })

    # Save the results to CSV
    results_df.to_csv(f"{pred_output_path}.csv", index=False)

    # Return the number of significant features and number of omic features
    return num_significant


def build_encoder(hidden_layers, latent_layer_size, n_input_features):
    """
    This function builds the encoder architecture based on the number of hidden layers and the
    number of input features and latent layer size.
    The encoder uses a progressive halving structure, where each hidden layer has roughly twice the
    number of neurons as the next. For example, with a latent size of 50 and an input of 400, the
    layer sizes would be [400, 200, 100, 50]. If the input is smaller (e.g., 200), the intermediate
    layers are adjusted proportionally (e.g., [200, 150, 100, 50]).
    :param hidden_layers: The number of hidden layers in the encoder.
    :param latent_layer_size: The size of the latent layer.
    :param n_input_features: The number of input features in the data.
    :return: List containing the number of neurons in each layer of the encoder.
    """
    # First case, if there are no hidden layers, the encoder is simply the input size and the latent
    # layer size.
    if hidden_layers == 0:
        ENCODER = [n_input_features, latent_layer_size]
    # Second case, if there is one hidden layer, the encoder has three layers: input size,
    elif hidden_layers == 1:
        # If the input size is larger than twice the latent layer size, the encoder has three layers:
        # input size, twice the latent layer size, and the latent layer size.
        if latent_layer_size * 2 < n_input_features:
            ENCODER = [n_input_features, latent_layer_size * 2, latent_layer_size]
        # Otherwise, the layer sizes are adjusted to have a mean size between the input features
        # and the latent layer size.
        else:
            ENCODER = [n_input_features, int(np.mean([n_input_features, latent_layer_size])),
                       latent_layer_size]
    # Third case, if there are two hidden layers
    else:
        if hidden_layers > 2:
            # Warn if the number of hidden layers is greater than 2 (the model did not tested
            # with more than 2)
            warnings.warn(
                "The model does not support hidden_layers > 2 (this configuration was not tested). "
                "Setting hidden_layers to 2.")
        # If the input size is larger than four times the latent layer size, the encoder has four
        # layers:         # input size, four times the latent layer size, twice the latent layer
        # size, and the latent layer size.
        if latent_layer_size * 4 < n_input_features:
            ENCODER = [n_input_features, latent_layer_size * 4, latent_layer_size * 2,
                       latent_layer_size]
        # Otherwise, the layer sizes are adjusted to have a mean size between the input features.
        else:
            diff = n_input_features - latent_layer_size
            ENCODER = [n_input_features, int(latent_layer_size + ((2 / 3) * diff)),
                       int(latent_layer_size + ((1 / 3) * diff)), latent_layer_size]
    return ENCODER