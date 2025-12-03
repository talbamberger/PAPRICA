"""
IBD (Franzosa et al.) Data Utilities

This module provides utilities for processing and loading multi-omic data from the Franzosa dataset,
including microbiome and metabolome data. It handles data preprocessing, filtering, transformation,
and integration for downstream analysis.

The module includes functions for:
- Loading and processing metabolomic data with optional class-level aggregation
- Loading and processing microbiome data from Kraken taxonomic profiles
- Metadata loading and preprocessing
- Data integration across omics and metadata
- Applying centered log-ratio (CLR) transformations
- Filtering features based on prevalence and correlation

"""

import pandas as pd
from skbio.stats.composition import clr

# === Franzosa/IBD Data Paths (update as needed) ===
DATA_DIR = "../git/DATA/FRANZOSA"
MTB_METADATA_FILE = f"{DATA_DIR}/mtb_metadata.csv"
MTB_RAW_FILE = f"{DATA_DIR}/mtb_from_sup.csv"
SRA_RUN_TABLE_FILE = f"{DATA_DIR}/SraRunTable.txt"
FEAT_METADATA_FILE = f"{DATA_DIR}/feat_metadata_from_paper.txt"
KRAKEN_SPECIES_FILE = f"{DATA_DIR}/kraken_species_level_taxonomy_RA.tsv"


def sum_mtb_clusters(data_dir, mtb):
    """
    Aggregate metabolite features by their putative chemical class.
    This function groups individual metabolite features into their respective chemical
    classes and sums their abundances, reducing the dimensionality of the metabolomic
    data while preserving biological interpretability.
    :param data_dir: Path to the directory containing metabolite metadata files.
    :param mtb: Metabolomic data with samples as rows and metabolite features as columns.
    Must contain a 'sample' column.
    :return: Aggregated metabolomic data with samples as rows and chemical
    classes as columns. Contains a 'sample' column and columns for
    each putative chemical class.
    """
    mtb_metadata = pd.read_csv(MTB_METADATA_FILE) # Read metabolite metadata
    mtb_metadata = mtb_metadata[["Metabolomic Feature", "Putative Chemical Class"]] # Keep only relevant columns
    mtb = mtb.set_index("sample") # Set sample as index
    mtb = mtb.T # Transpose
    mtb = mtb.reset_index().rename(columns={"# Feature / Sample": "Metabolomic Feature"}) # Reset index
    mtb = pd.merge(mtb, mtb_metadata, on="Metabolomic Feature") # Merge with metabolite metadata
    mtb = mtb.drop(columns=["Metabolomic Feature"]) # Drop metabolite feature column
    mtb = mtb.groupby("Putative Chemical Class").sum().T # Group by class, sum and transpose again
    mtb = mtb.reset_index().rename(columns={"index": "sample"}) # Reset index
    return mtb

def get_mtb(data_dir, group_by_class=True, filter_by_correlation=True, apply_clr=True):
    """
    Load and preprocess metabolomic data from the Franzosa dataset.
    This function performs comprehensive preprocessing of metabolomic data including:
    - Loading raw metabolomic data from CSV files
    - Sample name standardization
    - Prevalence-based feature filtering
    - Zero-value imputation with half-minimum values
    - Optional chemical class aggregation
    - Optional correlation-based feature filtering
    - Optional centered log-ratio (CLR) transformation
    :param data_dir: Path to the directory containing metabolomic data files.
    :param group_by_class:  Whether to aggregate metabolites by chemical class. Defaults to True.
    :param filter_by_correlation: Whether to remove highly correlated features (>0.8 correlation).
    Defaults to True.
    :param apply_clr: Whether to apply CLR transformation. Defaults to True.
    :return:  Processed metabolomic data with samples as rows and metabolite
                     features/classes as columns. Contains a 'sample' column.
    """
    # Load metabolomic data and transpose to have samples as rows
    mtb = pd.read_csv(MTB_RAW_FILE, index_col=0).T
    mtb = mtb.reset_index().rename(columns={"index": "sample"})

    # Standardize sample names by removing dataset-specific identifiers
    mtb["sample"] = mtb["sample"].str.replace("|", "_", regex=False).str.replace("Validation_", "",regex=False).str.replace("IBD", "", regex=False)

    # Remove rare features (more than 50% of the measurements were zero)
    threshold = 0.5 # Prevalence threshold (more than 50% of samples with non-zero values)
    prevalence = (mtb.iloc[:, 1:] > 0).mean()  # Calculate prevalence for each feature (excluding the first column)
    filtered_mask = prevalence > threshold # Boolean mask for columns to keep
    mtb = mtb.loc[:, [True] + filtered_mask.tolist()]  # Include the first column (Sample)

    # Handle zero values by imputation
    mtb = mtb.fillna(0)  # Fill NaN values with 0
    mtb.iloc[:, 1:] = mtb.iloc[:, 1:].astype(float)  # Ensure numeric data types
    # Replace zeros with half of the minimal non-zero value in each column to avoid log(0) in CLR
    mtb.iloc[:, 1:] = mtb.iloc[:, 1:].replace(0, (mtb.iloc[:, 1:][mtb.iloc[:, 1:] != 0].min() / 2))

    if group_by_class:
        # Sum metabolite cluster-level abundance into class-level abundance
        mtb = sum_mtb_clusters(data_dir, mtb)

    # Optional: Remove highly correlated features to reduce redundancy
    cor_threshold = 0.8
    if filter_by_correlation:
        # Calculate correlation matrix excluding the first column
        correlation_matrix = mtb.iloc[:, 1:].corr()
        columns = correlation_matrix.columns
        columns_to_drop = []
        # Identify highly correlated columns to drop (keep first occurrence)
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                if correlation_matrix.loc[columns[i], columns[j]] > cor_threshold:
                    columns_to_drop.append(columns[j])
        # Keep the first column and filter out redundant columns
        mtb = mtb.loc[:, ~mtb.columns.isin(columns_to_drop)]

    # Optional: Apply centered log-ratio transformation for compositional data
    if apply_clr:
        sample_column = mtb['sample']
        # Apply CLR to the feature columns
        feature_columns = mtb.iloc[:, 1:]
        clr_transformed = clr(feature_columns) # CLR transformation from scikit-bio
        # Convert back to a DataFrame
        clr_transformed_df = pd.DataFrame(clr_transformed, columns=feature_columns.columns)
        # Combine with the excluded column
        mtb = pd.concat([sample_column, clr_transformed_df], axis=1)
    return mtb


def get_metadata(data_dir, phenotype_col="Fecal.Calprotectin"):
    """
    Load and preprocess metadata from the Franzosa dataset.
    This function loads sample metadata and clinical phenotype data, performing
    necessary preprocessing including sample ID standardization and phenotype
    value capping for outliers.
    :param data_dir: Path to the directory containing metadata files.
    :param phenotype_col: Name of the phenotype column to extract.
    Currently only supports "Fecal.Calprotectin", defaults to "Fecal.Calprotectin".
    :return: Processed metadata with standardized sample IDs and cleaned
    phenotype values. Contains columns for sample information,
    sequencing run details, and clinical phenotypes.
    """
    # Load SRA run table and feature metadata, then merge them
    map_df = pd.read_csv(SRA_RUN_TABLE_FILE, sep=",")
    map_df = map_df.rename(columns={"Library Name": "SRA_metagenome_name"})
    metadata = pd.read_csv(FEAT_METADATA_FILE, sep="\t", index_col=0).T
    metadata = pd.merge(map_df, metadata, on="SRA_metagenome_name")
    
    # Standardize sample IDs by removing dataset-specific suffixes
    metadata["sample"] = metadata["host_subject_id"].str.replace("_NLIBD", "").str.replace("_PRISM", "").str.replace("IBD", "")
    metadata["subject"] = metadata["sample"]  # Create subject identifier
    
    # Handle phenotype column validation and selection
    if phenotype_col != "Fecal.Calprotectin":
        print(
            f"Metadata column {phenotype_col} not recognized. Using Fecal.Calprotectin as default.")
        phenotype_col = "Fecal.Calprotectin"
    
    # Clean phenotype data
    metadata = metadata.dropna(subset=[phenotype_col])  # Remove samples without phenotype data
    metadata[phenotype_col] = metadata[phenotype_col].astype(float)
    # Cap extreme outliers at 625 (samples with values >2000)
    metadata[phenotype_col] = metadata[phenotype_col].apply(lambda x: 625 if x > 2000 else x)
    return metadata


def get_microbiome(data_dir, metadata, apply_clr=True):
    """
    Load and preprocess microbiome data from Kraken taxonomic profiles.
    This function loads species-level taxonomic relative abundance data and performs
    comprehensive preprocessing including sample matching, abundance filtering,
    prevalence filtering, and optional CLR transformation.
    :param data_dir: Path to the directory containing microbiome data files.
    :param metadata: Sample metadata containing 'sample' and 'Run' columns for matching
    sequencing runs to sample IDs.
    :param apply_clr: Whether to apply CLR transformation. Defaults to True.
    :return: Processed microbiome data with samples as rows and microbial species as columns.
    Contains a 'sample' column and columns for each retained microbial feature.
    """
    # Load Kraken species-level taxonomic data and transpose to have samples as rows
    df = pd.read_csv(KRAKEN_SPECIES_FILE, sep="\t", index_col=0).T
    
    # Extract SRR identifiers from sample names for matching with metadata
    df["SRR"] = df.index
    df["SRR"] = df["SRR"].str.split("_", expand=True)[0]  # Extract SRR prefix
    
    # Merge with metadata to get standardized sample IDs
    df = pd.merge(df, metadata[["sample", "Run"]], left_on="SRR", right_on="Run").drop(columns=["Run", "SRR"])
    df = (df[["sample"] + [col for col in df.columns if col != "sample"]])  # Reorder columns

    # Convert relative abundances from percentages to proportions
    df.iloc[:, 1:] = df.iloc[:, 1:] / 100

    # Set filtering thresholds for feature selection
    abundance_threshold = 0.001  # Mean abundance threshold (0.1%)
    prevalence_threshold = 0.02  # Prevalence threshold (2% of samples)

    # Apply dual filtering: abundance AND prevalence criteria
    mean_abundance = df.iloc[:, 1:].mean()  # Calculate mean abundance for feature columns
    prevalence = (df.iloc[:, 1:] > 0).mean()  # Calculate prevalence for feature columns

    # Filter columns based on criteria (mean_abundance > 0.1% AND prevalence > 2%)
    filtered_columns = (mean_abundance > abundance_threshold) & (prevalence > prevalence_threshold)
    df = df.loc[:, [True] + filtered_columns.tolist()]  # Include the first column (Sample)

    # Optional: Apply CLR transformation for compositional data analysis
    if apply_clr:
        sample_column = df['sample']
        # Apply CLR to the feature columns
        feature_columns = df.iloc[:, 1:]
        clr_transformed = clr(feature_columns+1e-9)  # Add small pseudocount to avoid log(0)
        clr_transformed_df = pd.DataFrame(clr_transformed, columns=feature_columns.columns)
        df = pd.concat([sample_column, clr_transformed_df], axis=1)

    return df



def get_data(omic1, omic2, metadata_col):
    """
    Load and integrate multi-omic data from the Franzosa dataset.
    This is the main function for loading and integrating microbiome and metabolome
    data along with associated metadata. It handles data loading, preprocessing,
    and merging across data types.
    :param omic1: First omics data type to load. Must be "microbiome" or "metabolome".
    :param omic2: Second omics data type to load. Must be "microbiome" or "metabolome".
    :param metadata_col: Name of the metadata column/phenotype to extract.
    :return: A tuple containing:
            - list: Two DataFrames for the requested omics data types [omic1_data, omic2_data]
            - pd.DataFrame: Merged metadata for samples present in both omics datasets
    """
    data_dict = {}
    data_dir = DATA_DIR
    
    # Load metadata first as it's needed for microbiome data processing
    metadata = get_metadata(data_dir, metadata_col)

    # Load requested omics data types
    if omic1 == "microbiome" or omic2 == "microbiome":
        microbiome = get_microbiome(data_dir, metadata, apply_clr=True)
        data_dict["microbiome"] = microbiome

    if omic1 == "metabolome" or omic2 == "metabolome":
        metabolome = get_mtb(data_dir, group_by_class=True, filter_by_correlation=True, apply_clr=True)
        data_dict["metabolome"] = metabolome

    # Merge all datasets on sample column (inner join - only samples present in all datasets)
    df = pd.merge(pd.merge(data_dict["microbiome"], data_dict["metabolome"], on="sample"),
                  metadata, on="sample")

    # Extract individual datasets from merged dataframe to ensure consistent sample sets
    metadata = df[metadata.columns]
    data_dict["microbiome"] = df[data_dict["microbiome"].columns]
    data_dict["metabolome"] = df[data_dict["metabolome"].columns]
    
    # Return data in the order requested by user
    return [data_dict[omic1], data_dict[omic2]], metadata

#
# dfs, metadata  = get_data("microbiome", "metabolome", "Fecal.Calprotectin")
# print(dfs[0].shape, dfs[1].shape, metadata.shape)