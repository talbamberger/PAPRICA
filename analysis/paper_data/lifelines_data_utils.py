"""
Lifelines Data Utilities

NOTE: This code was run on the Lifelines server. Paths and data access are specific to the Lifelines cluster environment.

This module provides utilities for processing and loading multi-omic data from the Lifelines
cohort study, including microbiome and metabolome data. It handles data preprocessing,
filtering, transformation, and integration for downstream analysis.

The module includes functions for:
- Loading and processing metabolomic NMR data from blood samples
- Loading and processing microbiome data from Kraken taxonomic profiles
- Data filtering based on prevalence and correlation
- Applying centered log-ratio (CLR) transformations
- Integrating multi-omic data with clinical metadata

The Lifelines cohort is a large population-based study from the Netherlands that includes
comprehensive health data and multi-omic measurements from participants.

"""

import numpy as np
import pandas as pd
from skbio.stats.composition import clr

LIFELINES_DATA_DIR = "../git/DATA/LifeLines"
NMR_LIPIDS_FILE = f"{LIFELINES_DATA_DIR}/LLD_bloodlipids_nmr.txt.gz"
METADATA_FILE = f"{LIFELINES_DATA_DIR}/1a_q_1_results.csv"
DEEP_LINK_FILE = f"{LIFELINES_DATA_DIR}/OV22_00666_deep_linkage_file-v2.csv"
KRAKEN_SPECIES_FILE = f"{LIFELINES_DATA_DIR}/kraken_species_level_taxonomy.tsv"
MGS_LINK_FILE = f"{LIFELINES_DATA_DIR}/linkage_file_MGS.txt"
SAMPLE_FILE = f"{LIFELINES_DATA_DIR}/f/sample_file.tsv"


def filter_df(df):
    """
    Filter and preprocess a DataFrame for omics data analysis.
    This function performs comprehensive preprocessing including prevalence filtering,
    zero-value imputation, correlation-based feature removal, and optional transformations.
    Processing Steps:
        1. Remove features with >50% zero values (prevalence filtering)
        2. Replace zeros with half the minimum non-zero value per feature
    :param df: Input data with samples as rows and features as columns.
    Features with correlation above this threshold are removed. Defaults to 0.6.
    :return: Filtered and preprocessed DataFrame with reduced features and applied transformations.
    """
    # Filter features by prevalence: keep metabolites with non-zero values in >50% of samples
    df = df.loc[:, (df != 0).sum() > len(df) * 0.5 ]
    # Handle missing values and ensure numeric data types
    df = df.fillna(0)
    df = df.astype(float)
    # Replace zeros with half of the minimal non-zero value in each column to avoid log(0) in CLR
    df = df.replace(0, (df[df != 0].min() / 2))

    return df


def get_metabolome_data():
    """
    Load and preprocess NMR metabolomics data from the Lifelines cohort.
    This function loads blood lipids NMR data from the Lifelines deep phenotyping
    project and applies preprocessing including filtering and CLR transformation.
    Processing Steps:
        1. Load NMR blood lipids data from Lifelines deep phenotyping release
        2. Apply filtering (no correlation filtering, no scaling/log transform)
        3. Reset index and rename ID column to 'sample'
        4. Apply centered log-ratio (CLR) transformation for compositional analysis
    Note:
        - Data path is hardcoded to Lifelines cluster location
        - Requires access to Lifelines deep phenotyping data releases
    :return: Processed metabolomics data with samples as rows and NMR
                     features as columns. Contains a 'sample' column and CLR-transformed
                     metabolite measurements.
    """
    # Load NMR blood lipids data from Lifelines deep phenotyping project
    nmr = pd.read_csv(NMR_LIPIDS_FILE, sep="\t")#.head(100) 
    nmr.set_index('id', inplace=True)
    
    # Apply filtering
    nmr = filter_df(nmr)
    
    # Reset index and standardize column names
    nmr.reset_index(inplace=True)
    nmr.rename(columns={'id': 'sample'}, inplace=True)
    
    # Apply CLR transformation for compositional data analysis
    if True:  # Always apply CLR for metabolomics data
        sample_column = nmr['sample']
        # Apply CLR to the feature columns
        feature_columns = nmr.iloc[:, 1:]
        clr_transformed = clr(feature_columns)  # CLR transformation from scikit-bio
        # Convert back to a DataFrame with original column names
        clr_transformed_df = pd.DataFrame(clr_transformed, columns=feature_columns.columns)
        # Combine sample IDs with CLR-transformed features
        nmr = pd.concat([sample_column, clr_transformed_df], axis=1)
    return nmr

def get_data(omic1, omic2, metadata_col):
    """
    Load and integrate multi-omic data from the Lifelines cohort.
    This is the main function for loading and integrating microbiome and metabolome
    data along with associated clinical metadata from the Lifelines study. It handles
    data loading, preprocessing, and merging across data types.
    :param omic1: First omics data type to load. Must be "microbiome" or "metabolome".
    :param omic2: Second omics data type to load. Must be "microbiome" or "metabolome".
    :param metadata_col: Name of the metadata column/phenotype to extract from clinical
    questionnaire data.
    :return: A tuple containing:
            - list: Two DataFrames for the requested omics data types [omic1_data, omic2_data]
            - pd.DataFrame: Merged clinical metadata for samples present in both omics datasets
    """
    # Initialize dictionary to store omics data
    omics_dict = {}
    
    # Load requested omics data types
    if omic1 == "microbiome" or omic2 == "microbiome":
        omics_dict["microbiome"] = get_kraken_species()

    if omic1 == "metabolome" or omic2 == "metabolome":
        omics_dict["metabolome"] = get_metabolome_data()

    # Load clinical metadata from Lifelines questionnaire data
    metadata  = pd.read_csv(METADATA_FILE)
    
    # Load linkage file to connect deep phenotyping IDs with project IDs
    deep_link = pd.read_csv(DEEP_LINK_FILE)
    
    # Merge clinical data with linkage information
    metadata  = pd.merge(deep_link, metadata, on="project_pseudo_id", how="inner")
    metadata.rename(columns={'LLDEEP_ID': 'sample'}, inplace=True)
    metadata["subject"] =  metadata["sample"]  # Create subject identifier
    
    # Merge omics datasets on sample IDs (inner join - only shared samples)
    merged_df = pd.merge(omics_dict[omic1], omics_dict[omic2], on='sample', how='inner')
    
    # Prepare metadata: keep only columns not in omics data (except 'sample')
    metadata = metadata[[col for col in metadata.columns if (col not in merged_df.columns) or (col == "sample")]]
    
    # Filter out samples missing the specified metadata column
    metadata = metadata.dropna(subset=[metadata_col])
    
    # Final merge: combine omics and metadata (inner join for complete cases only)
    merged_with_meta = pd.merge(merged_df, metadata, on='sample', how='inner')
    
    # Extract individual omics datasets with consistent sample sets
    omics_df = [merged_with_meta[list(omics_dict[omic1].columns)], merged_with_meta[list(omics_dict[omic2].columns)]]
    metadata = merged_with_meta[metadata.columns]
                                
    return omics_df, metadata

def get_kraken_species():
    """
    Load and preprocess microbiome data from Kraken species-level taxonomic profiles.
    
    This function loads pre-filtered Kraken taxonomic data from the Lifelines
    metagenomics sequencing (MGS) project and applies CLR transformation for
    compositional analysis.
    
    Returns:
        pd.DataFrame: Processed microbiome data with samples as rows and microbial
                     species as columns. Contains a 'sample' column (LLDEEP_ID) and
                     CLR-transformed relative abundance values.
    
    Processing Steps:
        1. Load pre-filtered Kraken species-level taxonomic data
        2. Load MGS linkage file to connect sequencing IDs with participant IDs
        3. Load sample metadata file with file accession information
        4. Merge datasets to map sequencing files to participant IDs
        5. Apply CLR transformation with small pseudocount (1e-9) to avoid log(0)
        6. Return data with standardized sample identifiers (LLDEEP_ID)
    
    Note:
        - Data paths are hardcoded to Lifelines cluster locations
        - Uses pre-filtered taxonomic data (filtering applied upstream)
        - CLR transformation helps with compositional microbiome data analysis
        - Requires access to Lifelines MGS data and metadata files
        - Sample aliases are standardized by removing 'LL' prefix
    """
    # Load pre-filtered Kraken species-level taxonomic data
    ra = pd.read_csv(KRAKEN_SPECIES_FILE)
    
    # Load MGS linkage file to connect sequencing IDs with participant IDs
    link = pd.read_csv(MGS_LINK_FILE, header=None, sep=" ")
    link.columns = ['LLDEEP_ID', 'sample_alias']
    link['sample_alias'] = link['sample_alias'].str.replace('LL', '', regex=False)
    
    # Load sample metadata with file accession information
    sample_file = pd.read_csv(SAMPLE_FILE, sep="\t")
    sample_file['sample_alias'] = sample_file['sample_alias'].str.replace('LL', '', regex=False)
    
    # Merge linkage and sample files to get participant-to-file mapping
    merged = pd.merge(link, sample_file, on="sample_alias", how="inner")
    
    # Merge taxonomic data with participant IDs, replacing file IDs with participant IDs
    ra = pd.merge(merged[["LLDEEP_ID", "file_accession_id"]], ra, left_on = "file_accession_id", right_on="sample", how="inner").drop(columns=["file_accession_id", "sample"])
    ra.rename(columns={'LLDEEP_ID': 'sample'}, inplace=True)
    
    # Apply CLR transformation for compositional microbiome data analysis
    sample_column = ra['sample']
    # Apply CLR to the feature columns
    feature_columns = ra.iloc[:, 1:]
    clr_transformed = clr(feature_columns+1e-9)  # Add small pseudocount to avoid log(0)
    clr_transformed_df = pd.DataFrame(clr_transformed, columns=feature_columns.columns)
    ra = pd.concat([sample_column, clr_transformed_df], axis=1)
    return ra


dfs, metadata = get_data("microbiome", "metabolome", "age")
omic1_data = dfs[0]
omic2_data = dfs[1] 
print(f"\nMicrobiome data shape: {omic1_data.shape}")
print(f"Metabolome data shape: {omic2_data.shape}")
print(f"Metadata shape: {metadata.shape}")