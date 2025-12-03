
#######################################################################
#                              D Model                                #
#######################################################################

# This file is part of the D model for cross-omic prediction.
# The D model employs a simple encoder-decoder architecture designed to predict one omic profile
# from another. The encoder embeds the first omic profile into a lower-dimensional latent space,
# and the decoder predicts the second omic profile from this latent representation.
# This architecture was used in Le et al.

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
import torch
from torch import device
from torch import nn
from torch.cuda import is_available
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor
from sklearn.model_selection import KFold
from joblib import Parallel, delayed

# Local utilities
import src.models_utils as models_utils


#######################################################################
#                             TRAINING                                #
#######################################################################


def grid_search(param_grid, get_data_func, omic1, omic2, phenotype_col,
                output_path=".", run_identifier="", num_processes=8, quiet_mode=True):
    """
    This function performs a grid search over the specified hyperparameters for the D model.
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
    :return: None. The results of the grid search will be saved to a CSV file in the specified
    output path.
    """

    output_path = f"{output_path}/results/{run_identifier}/d_{omic1}_{omic2}_{phenotype_col}"
    if not quiet_mode:
        print("Running D model...")
        print(f"The results will be saved to: {output_path}")

    if not os.path.exists(f"{output_path}/"):
        os.makedirs(f"{output_path}")
        os.makedirs(f"{output_path}/loss")
        os.makedirs(f"{output_path}/cross_omic_predictions")
        os.makedirs(f"{output_path}/processed")

    param_combinations = list(enumerate(product(*param_grid.values())))
    random.shuffle(param_combinations)
    omics_df, metadata = get_data_func(omic1, omic2, phenotype_col)
    partial_func = partial(train_predict_cv_parallel, omics_df=omics_df,
                           metadata=metadata, phenotype_col=phenotype_col,
                           output_path=output_path, quiet_mode=quiet_mode)
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


def train_predict_cv_parallel(params, omics_df, metadata, phenotype_col, output_path, quiet_mode):
    """
    This function trains the D model on the given omics data and metadata.
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
    :return: A dictionary containing the results of the training and predictions for this set of
    hyperparameters.
    """
    n_folds, LR, batch_size, N, hidden_layers, latent_layer_size, rep, weight_decay, eps =  params[1]

    df1 = omics_df[0]
    df2 = omics_df[1]

    n_input_features1 = df1.shape[1] - 1
    n_input_features2 = df2.shape[1] - 1

    ENCODER = models_utils.build_encoder(hidden_layers, latent_layer_size, n_input_features1)
    DECODER = models_utils.build_encoder(hidden_layers, latent_layer_size, n_input_features2)
    subject_df = metadata.groupby('subject', as_index=False)[phenotype_col].mean()

    # Initialize the stratifies KFold object
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    kf_split = kf.split(range(subject_df.shape[0]))

    rf_original = []
    rf_latent = []
    significant_correlations_arr = []

    for fold, (train_index, test_index) in enumerate(kf_split):

        if not quiet_mode:
            print(f"Fold {fold + 1}/{n_folds} with parameters: "
                  f"LR={LR}, batch_size={batch_size}, N={N}, "
                  f"hidden_layers={hidden_layers}, latent_layer_size={latent_layer_size}, "
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
        params_string = f"f-{fold}_N-{N}_LR-{LR}_batchsize-{batch_size}_hiddenlayers-{hidden_layers}_latentlayersize-{latent_layer_size}-rep-{rep}"

        model, train_loss, test_loss = init_and_train(train_df1=train_df1,
                                                      test_df1=test_df1,
                                                      train_df2=train_df2,
                                                      test_df2=test_df2,
                                                      batch_size=batch_size,
                                                      ENCODER=ENCODER,
                                                      DECODER=DECODER,
                                                      criterion=D_loss(),
                                                      N=N,
                                                      LR=LR,
                                                      weight_decay=weight_decay,
                                                      eps=eps,
                                                      quiet_mode=quiet_mode)

        # Get the embeddings and decoded representations for the train and test dataframes
        data_dict = {
            name: apply_d_model(name, model, train_df1, test_df1)
            for name in ["Original", "Latent", "Decoded"]
        }

        pred_output_path = f"{output_path}/cross_omic_predictions/{params_string}.csv"
        n_predicted = models_utils.get_omic_prediction(test_df2, data_dict["Decoded"]\
            ["test_df"], pred_output_path)
        
        if not quiet_mode:

            # Plot the training and test losses
            fig, axs = plt.subplots(1, 2, figsize=(12, 6),
                                    gridspec_kw={'hspace': 0.3, 'wspace': 0.3})
            models_utils.plot_losses(axs[0], train_loss, title="Training loss")
            models_utils.plot_losses(axs[1], test_loss, title="Test loss")
            # Write the losses to text files and save the plots
            for split, losses in [("train", train_loss), ("test", test_loss)]:
                with open(f"{output_path}/loss/{split}_loss_{params_string}.txt", "w") as f:
                    f.write("\n".join(map(str, losses)))
            plt.savefig(
                f"{output_path}/loss/{params_string}.png")

        # Predict the phenotype using Random Forest on the original, latent, and decoded representations
        rf_dict = models_utils.get_rf_prediction(data_dict, train_metadata, test_metadata,
                                                 phenotype_col)

        # Save the train and test data  for each representation
        processed_path = f"{output_path}/processed"
        for split in ["train", "test"]:
            for space in ["Original", "Latent"]:
                df = data_dict[space][f"{split}_df"]
                filename = f"{split}_{space.lower()}_{params_string}.csv"
                df.to_csv(f"{processed_path}/{filename}")

        # Save the results on this fold (Random Forest scores and number of significant correlations
        # of the cross omic predictions)
        rf_original.append(rf_dict["Original"])
        rf_latent.append(rf_dict["Latent"])
        significant_correlations_arr.append(n_predicted)

    # Generate a new row with the results to add to the final hyperparameter dataframe
    new_row = {"n_folds": n_folds,
               "LR": LR,
               "batch_size": batch_size,
               "N": N,
               "hidden_layers": hidden_layers,
               "latent_layer_size": latent_layer_size,
               "weight_decay": weight_decay,
               "rep": rep,
               "rf_orig_cor": np.mean(rf_original),
               "rf_latent_cor": np.mean(rf_latent),
               "significant_correlations": np.mean(significant_correlations_arr),
               "number_of_features2": n_input_features2,
               }

    return new_row

def init_and_train(train_df1,
                   test_df1,
                   train_df2,
                   test_df2,
                   batch_size,
                   ENCODER,
                   DECODER,
                   criterion,
                   N,
                   LR,
                   weight_decay,
                   eps,
                   quiet_mode):
    """
    This function initializes the D model and trains it on the provided dataframes.
    :param train_df1: Train dataframe for the first omic (input omic).
    :param test_df1: Test dataframe for the first omic (input omic).
    :param train_df2: Train dataframe for the second omic (output omic).
    :param test_df2: Test dataframe for the second omic (output omic).
    :param batch_size: The batch size to use for training.
    :param ENCODER: The sizes of the encoder layers (list of integers).
    :param DECODER: The sizes of the decoder layers (list of integers).
    :param criterion: Loss function to use for training the model.
    :param N: Number of epochs to train the model for.
    :param LR: The learning rate for the optimizer.
    :param weight_decay: The weight decay (L2 regularization) for the optimizer.
    :param eps: The epsilon value for the optimizer to avoid division by zero.
    :param quiet_mode: Boolean flag to suppress output messages during training.
    :return: The model, training loss, and test loss.
    """

    # Set the seed for NumPy (used by PyTorch's DataLoader for shuffling)
    np.random.seed(42)

    # Init the DataLoader
    loader = DataLoader(dataset=D_Dataset(train_df1, train_df2), batch_size=int(batch_size),
                        shuffle=True)


    # Init all the model parameters
    dev = device("cuda" if is_available() else "cpu")
    if not quiet_mode:
        print(f"Device: {dev}")
    model = D(encoder_sizes=ENCODER, decoder_sizes=DECODER).to(dev)


    # Apply the weight initialization to the model
    model.apply(models_utils.initialize_weights)
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=weight_decay, eps=eps)
    transform = Compose([ToTensor()])

    train_loss = []
    test_loss = []

    # Train loop
    for epoch in range(N):

        sum_final_loss = 0

        # Iterate over the DataLoader
        for input, output in loader:

            # Move input and output tensors to the device
            input = input.view(-1, input.shape[1]).to(dev)
            output = output.view(-1, output.shape[1]).to(dev)

            # Apply the transformation to the input and output tensors
            optimizer.zero_grad()
            latent, predicted = model(input)
            loss = criterion(output, predicted)
            loss.backward()
            optimizer.step()
            # Store the loss
            sum_final_loss += loss.item()

        # Add the average loss for this epoch to the list
        train_loss.append(sum_final_loss / len(loader))

        if not quiet_mode:
            print(f'epoch : {epoch + 1}/{N}, epoch loss = {train_loss[-1]:.6f}')

        # Evaluate on test set
        test_input_tensor = torch.tensor(test_df1.values.astype(np.float32)).float()
        test_output_tensor = torch.tensor(test_df2.values.astype(np.float32)).float()
        test_latent, test_predicted = model(test_input_tensor.to(dev))
        loss_test = criterion(test_output_tensor, test_predicted.to(dev))
        test_loss.append(loss_test.cpu().detach().numpy().item())

    return model, train_loss, test_loss


#######################################################################
#                               MODEL                                 #
#######################################################################

class D(torch.nn.Module):
    """
    D model for cross-omic prediction.
    """
    def __init__(self, encoder_sizes, decoder_sizes):
        super().__init__()
        self.encoder_sizes = encoder_sizes

        # Encoder
        enc_layers = []
        for i in range(len(encoder_sizes) - 1):
            enc_layers.append(nn.Linear(encoder_sizes[i], encoder_sizes[i + 1]))
            enc_layers.append(nn.ReLU(inplace=True))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder
        dec_layers = []
        for i in reversed(range(1, len(decoder_sizes))):
            dec_layers.append(nn.Linear(decoder_sizes[i], decoder_sizes[i - 1]))
            dec_layers.append(nn.ReLU(inplace=True))
        dec_layers[-1] = nn.Tanh()
        self.decoder = nn.Sequential(*dec_layers[:-1])

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed


#######################################################################
#                           LOSS FUNCTION                             #
#######################################################################


class D_loss(nn.Module):
    """
    Loss function for the D model.
    """
    def __init__(self):
        super(D_loss, self).__init__()

    def forward(self, output, predicted):
        dev = device("cuda" if is_available() else "cpu")
        return torch.nn.functional.mse_loss(output.to(dev), predicted.to(dev))



#######################################################################
#                             Data loader                             #
#######################################################################


class D_Dataset(Dataset):
    """
    Custom Dataset for the D model.
    """

    def __init__(self, df1, df2):
        """
        Initializes the dataset with two dataframes.
        :param df1: Input dataframe (1st omic).
        :param df2: Output dataframe (2nd omic).
        """
        self.train1 = torch.tensor(df1.values.astype(float), dtype=torch.float32)
        self.train2 = torch.tensor(df2.values.astype(float), dtype=torch.float32)

    def __len__(self):
        """
        Returns the length of the dataset.
        :return: The number of samples in the dataset.
        """
        return self.train1.shape[0]

    def __getitem__(self, idx):
        """
        Returns the input and output tensors for a given index.
        :param idx: Given index to retrieve the data.
        :return: Tuple of input and output tensors.
        """
        return self.train1[idx], self.train2[idx]


#######################################################################
#                           HELPER FUNCTIONS                          #
#######################################################################

def apply_d_model(space, model, train_df, test_df):
    """
    This function applies the trained model to the train and test dataframes
    :param space: The space to apply the model to. Can be "Original", "Latent", or "Decoded".
    :param model: The trained model to apply.
    :param train_df: The training dataframe to apply the model to.
    :param test_df: The test dataframe to apply the model to.
    :return: The transformed train and test dataframes as a dictionary.
    """
    dev = device("cuda" if is_available() else "cpu")

    if space in ["Latent", "Decoded"]:
        train_latent, train_decoded = model(torch.Tensor(train_df.values.astype(np.float32))\
                                            .to(dev))
        train_df = pd.DataFrame(train_latent.cpu().detach().numpy())
        test_latent, test_decoded = model(torch.Tensor(test_df.values.astype(np.float32)).to(dev))
        test_df = pd.DataFrame(test_latent.cpu().detach().numpy())

        if space == "Decoded":
            train_df = pd.DataFrame(train_decoded.cpu().detach().numpy())
            test_df = pd.DataFrame(test_decoded.cpu().detach().numpy())

    return {"train_df": train_df, "test_df": test_df}