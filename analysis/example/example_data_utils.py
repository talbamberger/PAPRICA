"""
Example Data Utilities

This module provides a simple template for creating custom data loading functions
for new datasets. It demonstrates the expected data format and structure that
should be returned by the get_data function.

Use this as a template when adapting the models to work with your own datasets.
Simply replace the placeholder data generation with your actual data loading logic.

"""

import pandas as pd
import numpy as np


def get_data(omic1, omic2, phenotype_col):
    """
    Load and integrate multi-omic data with metadata.
    
    This is a template function that demonstrates the expected format for the get_data
    function. Replace the placeholder data generation with your actual data loading logic.
    
    Args:
        omic1 (str): First omics data type to load (e.g., "microbiome", "metabolome").
        omic2 (str): Second omics data type to load (e.g., "microbiome", "metabolome").
        phenotype_col (str): Name of the metadata column/phenotype to use as target.
    
    Returns:
        tuple: A tuple containing:
            - list: Two DataFrames for the requested omics data types [omic1_data, omic2_data]
            - pd.DataFrame: Metadata with the specified phenotype column
    
    Expected Data Format:
        - Each omics DataFrame should have 'sample' as the first column
        - Feature columns should contain numerical data (preprocessed/transformed)
        - Metadata DataFrame should include 'sample' and 'subject' columns
        - All DataFrames should have consistent sample identifiers
        
    Example Usage:
    # >>> omic1_data, omic2_data, metadata = get_data("microbiome", "metabolome", "disease_score")
    """
    
    # TODO: Replace this section with your actual data loading logic
    # ================================================================
    
    # Example: Generate simple placeholder data to demonstrate format
    n_samples = 100
    sample_ids = [f"sample_{i:03d}" for i in range(n_samples)]
    
    # Placeholder microbiome data (samples x microbial_features)
    microbiome_data = pd.DataFrame({
        'sample': sample_ids,
        # TODO: Replace with actual microbiome feature loading
        # Example: Load from CSV, database, or other data source
        'Bacteroides_fragilis': np.random.randn(n_samples),
        'Prevotella_copri': np.random.randn(n_samples),
        'Faecalibacterium_prausnitzii': np.random.randn(n_samples),
        'Escherichia_coli': np.random.randn(n_samples),
        'Bifidobacterium_longum': np.random.randn(n_samples),
        # Add more microbial features as needed...
    })
    
    # Placeholder metabolome data (samples x metabolite_features)
    metabolome_data = pd.DataFrame({
        'sample': sample_ids,
        # TODO: Replace with actual metabolome feature loading
        # Example: Load from CSV, database, or other data source
        'Amino_acids': np.random.randn(n_samples),
        'Fatty_acids': np.random.randn(n_samples),
        'Carbohydrates': np.random.randn(n_samples),
        'Organic_acids': np.random.randn(n_samples),
        'Lipids': np.random.randn(n_samples),
        # Add more metabolite features as needed...
    })
    
    # Placeholder metadata
    metadata = pd.DataFrame({
        'sample': sample_ids,
        'subject': sample_ids,  # Can be different if multiple samples per subject
        # TODO: Replace with actual metadata loading
        phenotype_col: np.random.uniform(0, 100, n_samples),  # Target phenotype
        'age': np.random.uniform(20, 80, n_samples),
        'sex': np.random.choice(['Male', 'Female'], n_samples),
        'bmi': np.random.uniform(18, 35, n_samples),
        # Add more clinical variables as needed...
    })
    
    # ================================================================
    # End of placeholder section
    
    # Create data dictionary for easy access
    data_dict = {
        'microbiome': microbiome_data,
        'metabolome': metabolome_data
    }
    
    # Return the requested omics data in the specified order
    return [data_dict[omic1], data_dict[omic2]], metadata


# Example of how to customize this function for your data:
"""
def get_data(omic1, omic2, phenotype_col):
    # Step 1: Load your microbiome data
    microbiome_data = pd.read_csv("path/to/your/microbiome_data.csv")
    # Ensure first column is named 'sample'
    
    # Step 2: Load your metabolome data  
    metabolome_data = pd.read_csv("path/to/your/metabolome_data.csv")
    # Ensure first column is named 'sample'
    
    # Step 3: Load your metadata
    metadata = pd.read_csv("path/to/your/metadata.csv")
    # Ensure it has 'sample', 'subject', and your phenotype columns
    
    # Step 4: Apply any necessary preprocessing
    # - Handle missing values
    # - Apply transformations (log, CLR, etc.)
    # - Filter features
    # - Normalize/scale data
    
    # Step 5: Merge datasets on sample IDs (optional)
    # merged = pd.merge(microbiome_data, metabolome_data, on='sample')
    # metadata = pd.merge(metadata, merged[['sample']], on='sample')
    
    # Step 6: Return in expected format
    data_dict = {'microbiome': microbiome_data, 'metabolome': metabolome_data}
    return [data_dict[omic1], data_dict[omic2]], metadata
"""


if __name__ == "__main__":
    # Test the function
    print("Testing example data loading...")
    
    data_list, metadata = get_data("microbiome", "metabolome", "disease_progression")
    omic1_data = data_list[0]
    omic2_data = data_list[1]

    print(f"\nMicrobiome data shape: {omic1_data.shape}")
    print(f"Metabolome data shape: {omic2_data.shape}")
    print(f"Metadata shape: {metadata.shape}")
    
    print(f"\nMicrobiome columns: {list(omic1_data.columns[:6])}")
    print(f"Metabolome columns: {list(omic2_data.columns[:6])}")
    print(f"Metadata columns: {list(metadata.columns)}")
    
    print("\nData loading test completed successfully!")
