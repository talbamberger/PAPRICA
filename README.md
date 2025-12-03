# Phenotype-driven parallel embedding for microbiome multi-omic data integration

**PAPRICA** (Phenotype-Aware Parallel Representation for Integrative omiC Analysis) is a flexible encoder-decoder framework for microbiome multi-omic integration. It embeds each omic into its own latent space, jointly studies their relationships, and supports cross-omic and phenotype prediction, preserving omic-specific signals while capturing phenotype-associated variation. 

This codebase implements the five neural network models compared in our study: PAPRICA and four baseline architectures representing successive advances in multi-omic integration. These models enable accurate, scalable integration and analysis of complex microbiome data, supporting both cross-omic prediction and phenotype prediction tasks.

<div align="center">
  <img src="models_for_the_git.png" alt="Model Architectures" width="600"/>
</div>

**Repository Structure:**
- `src/`: Model architecture code
- `analysis/`: Example scripts (using demo data), data loading utilities, and scripts to run models and generate results described in the paper

## Model Architectures

### D Model (Diagonal) *(see Panel A)*
The D model employs a simple encoder-decoder architecture designed to predict one omic profile from another. The encoder embeds the first omic profile into a lower-dimensional latent space, and the decoder predicts the second omic profile from this latent representation.

### Y Model *(see Panel B)*
The Y model utilizes an encoder-decoder architecture designed to predict one omic profile from another (as in the D model above), but does so while also reconstructing the first omic profile. This dual objective ensures that the latent space captures not only the features relevant for predicting the second omic dataset but also those unique to the first omic dataset.

### X Model *(see Panel C)*
The X model employs an encoder-decoder architecture designed to predict and reconstruct both omics, using a shared latent space, with the input being either the first omic profile or the second omic profile. The model is trained with alternating input (using the first omic during even epochs and the second omic during odd epochs), allowing the model to cope with missing omic data, to leverage complementary information between the two omics when available, and to predict each omic from the other while learning both shared and unique features.

### Parallel Models (Pd and Pdp) *(see Panel D)*
The parallel models share a common architectural framework, consisting of two (or more) autoencoders trained in parallel, with each autoencoder corresponding to a specific omic dataset. This approach offers a unique benefit of generating a distinct latent space representation for each omic dataset, thus allowing the autoencoders to preserve the distinct biological structure of each omic and any modality-specific patterns.

#### Pd Model (Parallel with Distance-based Coupling)
The Pd model aims to align the structure and distribution of the different omics' latent spaces by augmenting the loss function with an omics distance loss component, ensuring that samples close to one another in one omic's latent space are also close in the other latent spaces.

#### Pdp Model (PAPRICA - Phenotype Aware Parallel Representation for Integrative omiC Analysis)
The Pdp model further aims to align omic-specific latent spaces with a phenotype space, by augmenting the loss function with a phenotype distance loss component. This component ensures that the latent spaces also capture distances between samples in the phenotype space, integrating available phenotypic information during training to enhance the biological relevance of the latent representations.

*Cross-omic prediction results for Pd and Pdp can be seen in Panel E; phenotype prediction results for Pd and Pdp can be seen in Panel F.*

For installation and usage instructions, please refer to the [RUNNING_GUIDE.md](RUNNING_GUIDE.md).

---

This repository includes all code related to the publication:
<!-- Add publication reference here when available -->
