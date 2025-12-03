# How to Run the PAPRICA Models

This guide provides step-by-step instructions for running all five models from the "Phenotype-driven parallel embedding for microbiome multi-omic data integration" paper, includinc the PAPRICA model. For detailed model descriptions, see the [README](./README.md).

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Quick Start with Example Data](#quick-start-with-example-data)
3. [Using Your Own Data](#using-your-own-data)
4. [Model Selection Guide](#model-selection-guide)
5. [Parameter Tuning](#parameter-tuning)
6. [Understanding Results](#understanding-results)
7. [Common Workflows](#common-workflows)
8. [Troubleshooting](#troubleshooting)

## Quick Reference

| Script                                                   | Purpose                                                 | Notes / Output Location                          |
|----------------------------------------------------------|---------------------------------------------------------|--------------------------------------------------|
| `analysis/example/train_all_models.py`                   | Train selected model(s) on the example synthetic data   | Writes results to `results/<run_identifier>/...` |
| `analysis/example/generate_model_summary_plots.py`       | Create summary figures after training (example)         | Writes to `results/<run_identifier>/summary_.../`|
| `analysis/paper_data/ibd_train_all_models.py`            | Paper-specific training pipeline (IBD dataset example)  | Writes results to `results/<run_identifier>/...` |

## Environment Setup

> **Recommended:** Python 3.8 or newer. If using conda, you may need to specify the correct channel for PyTorch (see https://pytorch.org/get-started/locally/).

### 1. Install Dependencies

```bash
# Using pip
pip install pandas numpy scikit-learn torch scikit-bio scipy matplotlib seaborn

# Or using conda
conda install pandas numpy scikit-learn pytorch scikit-bio scipy matplotlib seaborn
```

### 2. Verify Installation

```bash
python -c "import pandas, numpy, sklearn, torch, skbio; print('All packages installed successfully!')"
```

### 3. Download the Repository

```bash
git clone https://github.com/your-username/PAPRICA.git
cd PAPRICA
```

## Quick Start with Example Data

> **⚠️ IMPORTANT**: The example data is **completely synthetic** and is only used to demonstrate the correct data format and structure. It is **not real biological data**. You **must** implement your own `get_data` function to load your actual datasets.
> If you see errors about missing columns, you must implement your own get_data function as described below.

### Step 1: Test with Synthetic Data

```bash
cd analysis/example
python train_all_models.py --model d
```

This will:
- Generate **synthetic (fake) microbiome and metabolome data** for format demonstration
- Train the D model on this synthetic data
- Save results in the current directory

### Step 2: Try Different Models

```bash
# D model (cross-omic prediction: first omic → second omic)
python train_all_models.py --model d

# Y model (cross-omic prediction + first omic reconstruction)
python train_all_models.py --model y

# X model (bidirectional prediction with shared latent space)
python train_all_models.py --model x

# Pd model (parallel autoencoders with  with inter-omic distance loss)
python train_all_models.py --model pd

# Pdp model (PAPRICA: parallel autoencoders  with inter-omic distance loss and phenotype-distance loss)
python train_all_models.py --model pdp
```

### Step 3: Check Results

Results will be saved under the `results` directory, organized by the `run_identifier` parameter (default is `example` in the script). Each model run creates a subdirectory named `<model>_<omic1>_<omic2>_<phenotype_col>` inside `results/<run_identifier>/`.

For example, with `run_identifier="example"`, `model="d"`, `omic1="microbiome"`, `omic2="metabolome"`, and `phenotype_col="disease_progression"`, results will be saved in:

- `results/example/d_microbiome_metabolome_disease_progression/`

> **Note**: These results are from synthetic data and have no biological meaning. They only demonstrate the output format and directory structure.

### Step 4: Generate Summary Plots

After training the models, you can generate publication-style summary figures using the provided script. 

To run the script:

```bash
cd analysis/example
python generate_model_summary_plots.py
```

This script will read the results from all trained models and create summary visualizations in:

- `results/<run_identifier>/summary_<omic1>_<omic2>_<phenotype_col>/`

For example, with the default settings:
- `results/example/summary_microbiome_metabolome_disease_progression/`

    The generated plots include several summary visualizations. For a detailed description of each figure, see the documentation at the top of `generate_model_summary_plots.py`.

These figures allow you to visually compare model performance and feature relevance, as shown in the publication.

## Using Your Own Data

> **Checklist:**
> - All files must contain a `sample` column for sample IDs.
> - Metadata file must contain a `subject` column.
> - Metadata file must contain the phenotype column you wish to predict.
> - Sample IDs must match across all files. Use pandas `.merge()` to check for mismatches.

> **⚠️ CRITICAL**: You **MUST** write your own `get_data` function. The current `example_data_utils.py` only contains synthetic data for demonstration purposes.

### Step 1: Prepare Your Data Files
Organize your data in CSV format:

#### microbiome_data.csv
```
sample,Bacteroides_fragilis,Prevotella_copri,Faecalibacterium_prausnitzii,...
sample_001,0.234,-0.123,0.456,...
sample_002,-0.567,0.789,0.012,...
...
```

#### metabolome_data.csv
```
sample,Amino_acids,Fatty_acids,Carbohydrates,Organic_acids,...
sample_001,1.234,0.567,-0.890,...
sample_002,-0.345,1.678,0.234,...
...
```

#### metadata.csv
```
sample,subject,disease_score,age,sex,bmi,...
sample_001,subj_001,45.6,35,Female,24.5,...
sample_002,subj_002,67.8,42,Male,28.1,...
...
```

### Step 2: **REQUIRED** - Write Your Own Data Loading Function

**You must completely replace the synthetic data generation in `example_data_utils.py`** with your actual data loading code:

```python
def get_data(omic1, omic2, phenotype_col):
    """Load your actual data here - REPLACE ALL THE SYNTHETIC DATA CODE"""
    
    # Load your data files
    microbiome_data = pd.read_csv("path/to/your/microbiome_data.csv")
    metabolome_data = pd.read_csv("path/to/your/metabolome_data.csv") 
    metadata = pd.read_csv("path/to/your/metadata.csv")
    
    # Optional: Apply preprocessing
    # - Remove low-prevalence features
    # - Apply CLR transformation
    # - Handle missing values
    # - Normalize data
    
    # Create data dictionary
    data_dict = {
        'microbiome': microbiome_data,
        'metabolome': metabolome_data
    }
    
    # Return in expected format
    return [data_dict[omic1], data_dict[omic2]], metadata
```

### Step 3: Test Your Data

```bash
# Test data loading (will fail if you haven't replaced the synthetic data)
python example_data_utils.py

# Test model training with your actual data
python train_all_models.py --model d
```

> **Important**: If you haven't replaced the synthetic data generation code, the models will still run but will only train on meaningless synthetic data.

### Step 4: Adjust Parameters for Your Dataset

> **Tip:** Start with fewer epochs (`N`) and repetitions (`rep`) for quick debugging, then scale up for final runs. The `rep` parameter controls model stability and is not a random seed.

Edit the parameter grids in `train_all_models.py`:

> **Note:** There are separate parameter grids for single network models (`d`, `y`, `x`) and for parallel models (`pd`, `pdp`). Use `params_for_single_network` for single models and `params_for_parallel_models` for parallel models.

```python
params_for_single_network = {
    'n_folds': [5],
    'LR': [0.001, 0.0001, 0.00001],  # Learning rates
    'batch_size': [16, 32, 64],  # Batch sizes
    'N': [500, 1000, 1500],  # Number of epochs
    'hidden_layers': [0], # Number of hidden layers in the encoder (and decoder)
    'latent_layer_size': [16, 32], # Size of the latent layer
    'rep': [i for i in range(10)], # Number of repetitions for the model stability
    'weight_decay': [1e-5], # Keep this value low to avoid overfitting
    'eps': [1e-8], # Epsilon for numerical stability, keep 1e-8}
```

## Model Selection Guide

### Model Architectures

#### **D Model (Diagonal)**
The D model employs a simple encoder-decoder architecture designed to predict one omic profile from another. The encoder embeds the first omic profile into a lower-dimensional latent space, and the decoder predicts the second omic profile from this latent representation.


#### **Y Model** 
The Y model utilizes an encoder-decoder architecture designed to predict one omic profile from another (as in the D model above), but does so while also reconstructing the first omic profile. This dual objective ensures that the latent space captures not only the features relevant for predicting the second omic dataset but also those unique to the first omic dataset.


#### **X Model**
The X model employs an encoder-decoder architecture designed to predict and reconstruct both omics, using a shared latent space, with the input being either the first omic profile or the second omic profile. The model is trained with alternating input (using the first omic during even epochs and the second omic during odd epochs), allowing the model to cope with missing omic data, leverage complementary information between the two omics when available, and predict each omic from the other while learning both shared and unique features.


#### **Parallel Models (Pd and Pdp)**
The parallel models share a common architectural framework, consisting of two (or more) autoencoders trained in parallel, with each autoencoder corresponding to a specific omic dataset. This approach offers a unique benefit of generating a distinct latent space representation for each omic dataset, thus allowing the autoencoders to preserve the distinct biological structure of each omic and any modality-specific patterns.

##### **Pd Model (Parallel with Distance-based Coupling)**
The Pd model aims to align the structure and distribution of the different omics' latent spaces by augmenting the loss function with an omics distance loss component, ensuring that samples close to one another in one omic's latent space are also close in the other latent spaces.


##### **Pdp Model (PAPRICA - Phenotype Aware Parallel Representation for Integrative omiC Analysis)**
The Pdp model further aims to align omic-specific latent spaces with a phenotype space, by augmenting the loss function with a phenotype distance loss component. This component ensures that the latent spaces also capture distances between samples in the phenotype space, integrating available phenotypic information during training to enhance the biological relevance of the latent representations.



## Parameter Tuning

### Key Parameters to Adjust

#### Learning Rate (`LR`)
- **Start with**: `[0.001]`
- **If overfitting**: Add smaller values `[0.0001, 0.00001]`
- **If underfitting**: Add larger values `[0.01]`

#### Batch Size (`batch_size`)
- **Small datasets**: `[32]`
- **Medium datasets**: `[32, 64]`
- **Large datasets**: `[64, 128, 256]`

#### Epochs (`N`)
- **Quick testing**: `[100, 200]`
- **Normal training**: `[500, 1000]`
- **Thorough training**: `[1000, 1500, 2000]`

#### Architecture (`hidden_layers`, `latent_layer_size`)
- **Simple**: `hidden_layers: [0], latent_layer_size: [16]`
- **Medium**: `hidden_layers: [0, 1], latent_layer_size: [16, 32]`
- **Complex**: `hidden_layers: [1, 2], latent_layer_size: [32, 64]`

### Example Parameter Configurations

#### Fast Testing Configuration
```python
params_for_single_network = {
    'n_folds': [3],
    'LR': [0.001],
    'batch_size': [32],
    'N': [100],
    'hidden_layers': [0],
    'latent_layer_size': [16],
    'rep': [i for i in range(2)],
    'weight_decay': [1e-5],
    'eps': [1e-8],
}
```

#### Parameters for Parallel Models (`pd`, `pdp`)
For parallel models (`pd`, `pdp`), you need to specify parameters for each omic network separately. Example configuration:

```python
params_for_parallel_models = {
    'n_folds': [5],
    'LR1': [0.001],  # Learning rate for first omic network
    'LR2': [0.001],  # Learning rate for second omic network
    'batch_size': [32],
    'N': [500],
    'hidden_layers1': [0],         # Hidden layers for first omic
    'hidden_layers2': [0],         # Hidden layers for second omic
    'latent_layer_size1': [16],    # Latent size for first omic
    'latent_layer_size2': [16],    # Latent size for second omic
    'rep': [i for i in range(10)],
    'weight_decay': [1e-5],
    'eps': [1e-8],
}
```
Set these in your script when running `pd` or `pdp` models.

### Required Training Script Parameters (Non-Hyperparameters)
Before running your models and understanding the results, you must set the following parameters in your training script (`train_all_models.py`). These are not hyperparameters, but control the data, output, and training behavior:

- `omic1`: Name of the first omic dataset (e.g., "microbiome"). This must exactly match the key used in your `get_data` function.
- `omic2`: Name of the second omic dataset (e.g., "metabolome"). This must exactly match the key used in your `get_data` function.
- `phenotype_col`: Name of the continuous phenotype column in your metadata (e.g., "disease_progression"). 
- `run_identifier`: Label for this run; results will be saved under `results/<run_identifier>/` (e.g., "example")
- `output_path`: Root directory for outputs (usually ".")
- `num_processes`: Number of parallel processes for training (increase for speed, decrease for memory limits)
- `quiet_mode`: If `True`, suppresses most console output during training and disables plotting of loss components during training (useful for faster runs or when running on a server).
- `same_hyperparameters`: For parallel models (`pd`, `pdp`), if `True`, both omic networks use the same hyperparameters (those set for the first omic); if `False`, the grid search will independently optimize hyperparameters for each omic network.

Set these values at the top of your training script to match your dataset and analysis needs before running any models.

---

### Parallelization: Number of Processes
You can control parallelization (number of CPU processes used for training) via the `num_processes` parameter in `train_all_models.py`. Increase for faster grid search on multi-core machines, decrease if you experience memory issues.

```python
num_processes = 10  # Default is 10, adjust as needed
```
Pass this value when calling the grid search functions.

### Quiet Mode
Set `quiet_mode = True` to suppress most console output during training. This is useful for batch runs or when logging output to files.

```python
quiet_mode = True  # Suppresses verbose output
```
Pass this flag to the grid search functions to enable quiet mode.

### same_hyperparameters Flag
For parallel models, `same_hyperparameters=True` ensures both omic networks use identical hyperparameters (learning rate, architecture, etc.). Set to `False` to allow independent tuning for each omic.

```python
same_hyperparameters = True  # Both omic networks use the same hyperparameters
```
Set to `False` if you want to grid search different settings for each omic.



---
### Visualizing Results (as in the paper)

After running your models, you can generate summary plots in the same format as the publication by running:

```bash
cd analysis
python plot_models_comparison.py
```

This will create summary figures in the appropriate `summary_{omic1}_{omic2}_{phenotype}/` directory, allowing you to validate and compare your results visually.



## Understanding Results


### Output Directory Structure

| Folder/File         | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| `final.csv`         | Main results summary                                                        |
| `cross_omic_predictions/` | Cross-modal prediction files                                            |
| `loss/`             | Training loss curves as .txt files and plots (only if `quiet_mode=False`)    |
| `processed/`        | Original input data and latent representations for each fold/run             |
| `summary_.../`      | Summary figures (generated after running plotting script)                    |

Results are saved under the `results` directory, organized by the `run_identifier` parameter (set in `train_all_models.py`). Each model run creates a subdirectory named `<model>_<omic1>_<omic2>_<phenotype_col>` inside `results/<run_identifier>/`.

For example, with `run_identifier="example"`, `model="d"`, `omic1="microbiome"`, `omic2="metabolome"`, and `phenotype_col="disease_progression"`, results will be saved in:

```
results/example/d_microbiome_metabolome_disease_progression/
```

Inside each model directory, you will find:
- `final.csv`                 # Main results summary
- `cross_omic_predictions/`   # Cross-modal prediction files
- `loss/`                     # Contains training loss curves as .txt files and plots. These files are only generated if `quiet_mode` is set to `False`; when `quiet_mode=True`, loss curves and plots are skipped to speed up training and reduce output clutter.
- `processed/`                # Contains both the original input data and the corresponding latent representations for each fold and run (i.e., after splitting and model training).

Additionally, summary figures are saved in:
```
results/example/summary_microbiome_metabolome_disease_progression/
```
with files such as:
- `phenotype_prediction.png`      # Performance visualization
- `second_omic_prediction.png`    # Cross-omic performance
- `heatmap_all_second_features_Spearman_r.png`
- `heatmap_significant_second_features_Spearman_r.png`

> **Note**: These figures are generated only after running the `generate_model_summary_plots.py` script (see instructions above).

