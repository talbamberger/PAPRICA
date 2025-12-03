"""
DAP Data Utilities

This module provides utilities for processing and loading multi-omic data from the DAP cohort, including microbiome, metabolome, and methylation data. It handles data preprocessing, filtering, transformation, and integration for downstream analysis.

Functions:
- filter_df: Preprocess omics data (prevalence, correlation, normalization, log, scaling)
- get_data: Load and integrate multi-omic data with metadata

"""

import pandas as pd
import numpy as np
from skbio.stats.composition import clr

# === DAP Data Paths (update as needed) ===
DAP_DATA_DIR = "../git/DATA/DAP"
METABOLOME_FILE = f"{DAP_DATA_DIR}/mtbs_shared.csv"
MICROBIOME_FILE = f"{DAP_DATA_DIR}/ra_kraken_sp_shared.csv"
METADATA_FILE = f"{DAP_DATA_DIR}/metadata_age.csv"


def filter_df(df, cor_threshold=0.6, norm_scale=True, log_scale=False, quiet_mode=False):
    """
    Filter and preprocess a DataFrame for omics data analysis.
    Steps:
        1. Remove features with >50% zero values (prevalence filtering)
        2. Replace zeros with half the minimum non-zero value per feature
        3. Remove highly correlated features (above cor_threshold)
        4. Optional log transformation
        5. Optional normalization/scaling
    Args:
        df (pd.DataFrame): Input data with samples as rows and features as columns.
        cor_threshold (float): Correlation threshold for feature removal.
        norm_scale (bool): Whether to normalize and scale features.
        log_scale (bool): Whether to apply log transformation.
        quiet_mode (bool): If False, print shape info.
    Returns:
        pd.DataFrame: Filtered and preprocessed DataFrame.
    """
    df = df.copy()
    raw_shape = df.shape
    df = df.loc[:, (df != 0).sum() > len(df) * 0.5]
    filtered_shape = df.shape
    df = df.fillna(0)
    df = df.astype(float)
    df = df.replace(0, (df[df != 0].min() / 2))
    # Remove highly correlated features
    correlation_matrix = df.corr()
    columns = correlation_matrix.columns
    columns_to_drop = []
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            if correlation_matrix.loc[columns[i], columns[j]] > cor_threshold:
                columns_to_drop.append(columns[j])
    df = df[[col for col in df.columns if col not in columns_to_drop]]
    dropped_shape = df.shape
    if log_scale:
        df = np.log(df)
    if norm_scale:
        df = (df - df.mean()) / df.std()
    if not quiet_mode:
        print(f"Raw data shape: {raw_shape}")
        print(f"After prevalence filter: {filtered_shape}")
        print(f"After correlation filter (>{cor_threshold}): {dropped_shape}")
    return df


def get_metabolome_data():
    """
    Load and preprocess metabolome data from the DAP cohort.
    Returns:
        pd.DataFrame: Processed metabolome data with 'sample' column.
    """
    df = pd.read_csv(METABOLOME_FILE).rename(columns={"dap_dog_id": "sample"})
    return filter_df(df)


def get_microbiome_data():
    """
    Load and preprocess microbiome data from the DAP cohort (Kraken species-level).
    Applies CLR transformation for compositional analysis.
    Returns:
        pd.DataFrame: Processed microbiome data with 'sample' column.
    """
    mgx = pd.read_csv(MICROBIOME_FILE).rename(columns={"dap_dog_id": "sample"})
    sample_column = mgx['sample']
    feature_columns = mgx.iloc[:, 1:]
    clr_transformed = clr(feature_columns + 1e-9)
    clr_transformed_df = pd.DataFrame(clr_transformed, columns=feature_columns.columns)
    mgx = pd.concat([sample_column, clr_transformed_df], axis=1)
    return filter_df(mgx)


def get_data(omic1, omic2, metadata_col):
    """
    Load and integrate multi-omic data from the DAP cohort.
    Args:
        omic1 (str): First omics data type ('microbiome', 'metabolome', 'methylation').
        omic2 (str): Second omics data type.
        metadata_col (str): Metadata column/phenotype to use as target.
    Returns:
        tuple: ([omic1_data, omic2_data], metadata)
    """
    omics_dict = {}
    if omic1 == "metabolome" or omic2 == "metabolome":
        omics_dict["metabolome"] = get_metabolome_data()
    if omic1 == "microbiome" or omic2 == "microbiome":
        omics_dict["microbiome"] = get_microbiome_data()
    metadata = pd.read_csv(METADATA_FILE).rename(columns={"dap_dog_id": "sample"})
    metadata["subject"] = metadata["sample"]
    # Filter out samples missing the specified metadata column
    metadata = metadata.dropna(subset=[metadata_col])
    # Merge omics datasets on sample IDs (inner join)
    merged_df = pd.merge(omics_dict[omic1], omics_dict[omic2], on='sample', how='inner')
    # Prepare metadata: keep only columns not in omics data (except 'sample')
    metadata = metadata[[col for col in metadata.columns if (col not in merged_df.columns) or (col == "sample")]]
    # Final merge: combine omics and metadata (inner join for complete cases only)
    merged_with_meta = pd.merge(merged_df, metadata, on='sample', how='inner')
    # Extract individual omics datasets with consistent sample sets
    omics_df = [merged_with_meta[list(omics_dict[omic1].columns)], merged_with_meta[list(omics_dict[omic2].columns)]]
    metadata = merged_with_meta[metadata.columns]
    return omics_df, metadata

# Example usage/test
dfs, metadata = get_data("metabolome", "microbiome", "age_at_stool_collection")
print(dfs[0].shape, dfs[1].shape, metadata.shape)


