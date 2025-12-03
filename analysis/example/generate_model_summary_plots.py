"""
Generate Model Summary Plots
===========================

This script aggregates results from all trained models and generates publication-style summary figures for multi-omic prediction experiments.

Usage:
    python generate_model_summary_plots.py

Outputs:
    - Summary plots saved in results/<run_identifier>/summary_<omic1>_<omic2>_<phenotype_col>/
    - Plots include:
        * phenotype_prediction.png: Boxplot of phenotype prediction performance (correlation between true phenotype and predicted phenotype from latent space) for each model and omic. Includes statistical significance stars for pairwise model comparisons.
        * second_omic_prediction.png: Boxplot showing the number of well-predicted second omic features (cross-omic prediction) for each model. Each point represents a hyperparameter configuration; lines connect identical configurations across models.
        * second_omic_prediction_with_err_bars.png: Same as above, but with error bars showing standard deviation across repetitions for each configuration.
        * second_omic_prediction_percentage.png: Boxplot of the percentage of significant second omic features predicted by each model, normalized by total features. Useful for comparing relative model performance.
        * second_omic_prediction_by_feature.png: Stacked barplot showing, for each second omic feature, how many times it was well-predicted (significant) by each model across all hyperparameter settings and folds.
        * second_omic_prediction_by_feature_top_50.png: Stacked horizontal barplot for the top 50 most well-predicted second omic features, highlighting model-specific strengths.
        * heatmap_of_significant_features_Spearman_r.png: Heatmap of Spearman correlation coefficients for significant second omic features across models, hyperparameters, and folds. Asterisks mark significant correlations (FDR <0.1).

Parameters:
    - omic1, omic2, phenotype_col, run_identifier: Set in the script or modify as needed

Notes:
    - The script expects model results to be present in results/<run_identifier>/<model>_<omic1>_<omic2>_<phenotype_col>/final.csv
    - Figures are generated for all models: d, y, x, pd, pdp
"""

import os
import sys
import matplotlib.patches as patches

from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from statannotations.Annotator import Annotator
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.patches as mpatches
import numpy as np
import re

# Set these parameters as needed to be similar to the training script
OMIC1 = "microbiome"
OMIC2 = "metabolome"
PHENOTYPE_COL = "disease_progression"
RUN_IDENTIFIER = "example"  # Identifier for the run, can be used to save results in a specific folder
LOCAL_PATH = "."

def second_omic_prediction_percentage(df, out_dir):
    """
    Generate a boxplot showing the percentage of significant second omic features predicted by each model.
    Each box represents the distribution of well-predicted features (as a percentage of total features) for a given model across hyperparameter settings.
    Output: second_omic_prediction_percentage.png
    Interpretation: Higher percentages indicate better cross-omic prediction performance.
    """

    # Plot the results from the first repetition
    df = df[df["rep"] == 0]

    df['LR_new'] = df['LR'].combine_first(df['LR1'])

    # df['significant_correlations2'] *= 100 / df['number_of_features2']
    df.loc[:,'significant_correlations2'] = \
        df.loc[:,'significant_correlations2'] * 100 / df.loc[:,'number_of_features2']

    # Aggregate duplicates: compute mean and standard deviation
    agg_df = df.groupby(['LR_new', 'latent_layer_size_1', 'model', 'N'], as_index=False).agg(
        mean_value=('significant_correlations2', 'mean'),
        std_value=('significant_correlations2', 'std')
    )

    fig, ax = plt.subplots(figsize=(6.8, 5.0))

    agg_df = agg_df[[col for col in agg_df.columns if col not in ["std_value"]]]

    # Create boxplot
    sns.boxplot(
        data=df, x="model", y="significant_correlations2",
        dodge=True, palette=["#D89F9C"] * df["model"].nunique(), ax=ax
    )

    # Add jittered mean points with error bars
    model_categories = df["model"].unique()  # Ensure consistent category ordering

    for _, row in agg_df.iterrows():
        if row["model"] in model_categories:
            x_pos = np.where(model_categories == row["model"])[0][0]  # Get categorical index safely
            ax.errorbar(x=x_pos, y=row["mean_value"], 
                        fmt='o', color="black", alpha=0.2, capsize=3)

    # Extract scatter points for connecting related models
    offsets = {cat: i for i, cat in enumerate(model_categories)}
    scatter_points = {tuple(row): (offsets[row["model"]], row["mean_value"]) 
                      for _, row in agg_df.iterrows() if row["model"] in offsets}

    # Draw lines between identical (LR, latent_layer_size_1) mean points across different boxes
    grouped = agg_df.groupby(["LR_new", "latent_layer_size_1", "N"])
    for _, group in grouped:
        if len(group) > 1:
            points = [scatter_points[tuple(row)] for _, row in group.iterrows() if tuple(row) in scatter_points]
            points = sorted(points, key=lambda x: x[0])
            for i in range(len(points) - 1):
                ax.plot(
                    [points[i][0], points[i+1][0]], 
                    [points[i][1], points[i+1][1]], 
                    color="gray", linestyle="-", linewidth=0.5, alpha=0.6
                )

    # Customize labels and legend
    ax.set_ylabel('Well-predicted 2-nd omic features (%)')

    custom_handle_test = mpatches.Patch(
        facecolor="#D89F9C", label="Second omic", edgecolor="black", linewidth=0.5
    )
    ax.legend(handles=[custom_handle_test], title="Predicted omic", loc="upper left", frameon=True)

    plt.xticks(range(len(model_categories)), model_categories)  # Ensure correct x-tick labels
    plt.tight_layout()
    plt.savefig(f'{out_dir}/second_omic_prediction_percentage.png')
    plt.close()

def second_omic_prediction(df, out_dir):
    """
    Generate a boxplot showing the number of well-predicted second omic features for each model.
    Each point represents a hyperparameter configuration; lines connect identical configurations across models.
    Output: second_omic_prediction.png
    Interpretation: Models with more well-predicted features are more effective at cross-omic prediction.
    """

    # Plot the results from the first repetition
    df = df[df["rep"] == 0]

    # Handle the single network and multiple networks case together. Note that when the double
    # network is used, the latent_layer_size_1 and latent_layer_size_2 need to be the same for this
    # plot.
    df['LR_new'] = df['LR'].combine_first(df['LR1'])


    # Aggregate duplicates: compute mean and standard deviation
    # The hyperparameters used for the aggregation are: LR, latent_layer_size_1, and N.
    agg_df = df.groupby(['LR_new', 'latent_layer_size_1', 'model', 'N'], as_index=False).agg(
        mean_value=('significant_correlations2', 'mean'),
        std_value=('significant_correlations2', 'std')
    )

    fig, ax = plt.subplots(figsize=(6.8, 5.0))

    agg_df = agg_df[[col for col in agg_df.columns if col not in ["std_value"]]]

    # Create boxplot
    sns.boxplot(
        data=df, x="model", y="significant_correlations2",
        dodge=True, palette=["#D89F9C"] * df["model"].nunique(), ax=ax
    )

    # Add jittered mean points with error bars
    model_categories = df["model"].unique()  # Ensure consistent category ordering

    for _, row in agg_df.iterrows():
        if row["model"] in model_categories:
            x_pos = np.where(model_categories == row["model"])[0][0]  # Get categorical index safely
            ax.errorbar(x=x_pos, y=row["mean_value"], 
                        fmt='o', color="black", alpha=0.2, capsize=3)

    # Extract scatter points for connecting related models
    offsets = {cat: i for i, cat in enumerate(model_categories)}
    scatter_points = {tuple(row): (offsets[row["model"]], row["mean_value"]) 
                      for _, row in agg_df.iterrows() if row["model"] in offsets}

    # Draw lines between identical (lr, latent_layer_size_1, loss_weight) mean points across different boxes
    grouped = agg_df.groupby(["LR_new", "latent_layer_size_1", "N"])
    for _, group in grouped:
        if len(group) > 1:
            points = [scatter_points[tuple(row)] for _, row in group.iterrows() if tuple(row) in scatter_points]
            points = sorted(points, key=lambda x: x[0])
            for i in range(len(points) - 1):
                ax.plot(
                    [points[i][0], points[i+1][0]], 
                    [points[i][1], points[i+1][1]], 
                    color="gray", linestyle="-", linewidth=0.5, alpha=0.6
                )
            # break

    # Customize labels and legend
    ax.set_ylabel('Well-predicted 2nd omic features(n)')

    custom_handle_test = mpatches.Patch(
        facecolor="#D89F9C", label="Second omic", edgecolor="black", linewidth=0.5
    )
    ax.legend(handles=[custom_handle_test], title="Predicted omic", loc="upper left", frameon=True)

    plt.xticks(range(len(model_categories)), model_categories)  # Ensure correct x-tick labels
    plt.tight_layout()
    plt.savefig(f'{out_dir}/second_omic_prediction.png')
    plt.close()


def second_omic_prediction_err_bar(df, out_dir):
    """
    Generate a boxplot with error bars showing the number of well-predicted second omic features for each model.
    Error bars represent standard deviation across repetitions for each configuration.
    Output: second_omic_prediction_with_err_bars.png
    Interpretation: Allows assessment of model stability and consistency across runs.
    """

    df['LR_new'] = df['LR'].combine_first(df['LR1'])

    # Aggregate duplicates: compute mean and standard deviation
    agg_df = df.groupby(['LR_new', 'latent_layer_size_1', 'model', 'N'], as_index=False).agg(
        mean_value=('significant_correlations2', 'mean'),
        std_value=('significant_correlations2', 'std')
    )

    fig, ax = plt.subplots(figsize=(7, 7))

    # Create boxplot
    sns.boxplot(
        data=df, x="model", y="significant_correlations2",
        dodge=True, palette=["#D89F9C"] * df["model"].nunique(), ax=ax
    )

    # Add jittered mean points with error bars
    model_categories = df["model"].unique()  # Ensure consistent category ordering

    for _, row in agg_df.iterrows():
        if row["model"] in model_categories:
            x_pos = np.where(model_categories == row["model"])[0][0]  # Get categorical index safely
            # Add jitter to x position
            jitter_val = 0.2
            jitter = np.random.uniform(-jitter_val, jitter_val)  # Small jitter range
            x_pos += jitter
            ax.errorbar(x=x_pos, y=row["mean_value"], yerr=row["std_value"], 
                            fmt='o', color="black", alpha=0.2, capsize=3)

    # Customize labels and legend
    ax.set_ylabel('Well-predicted 2nd omic features (n)')

    custom_handle_test = mpatches.Patch(
        facecolor="#D89F9C", label="Second omic", edgecolor="black", linewidth=0.5
    )
    ax.legend(handles=[custom_handle_test], title="Predicted omic", loc="upper left", frameon=True)

    plt.xticks(range(len(model_categories)), model_categories)  # Ensure correct x-tick labels
    plt.tight_layout()
    plt.savefig(f'{out_dir}/second_omic_prediction_with_err_bars.png')
    plt.close()


def phenotype_prediction(df, out_dir):
    """
    Generate a boxplot of phenotype prediction performance (correlation between true phenotype and predicted phenotype from latent space) for each model and omic.
    Includes statistical significance stars for pairwise model comparisons.
    Output: phenotype_prediction.png
    Interpretation: Higher correlations indicate better phenotype prediction; significance stars show statistically significant differences between models.
    """
    # keep rep==0 and tidy
    df = df[df["rep"] == 0].copy()

    first_df = df[["model", "rf_latent_cor1"]].rename(columns={"rf_latent_cor1": "rf_latent_cor"})
    first_df["omic"] = "first"
    second_df = df[["model", "rf_latent_cor2"]].rename(columns={"rf_latent_cor2": "rf_latent_cor"})
    second_df["omic"] = "second"

    df = pd.concat([first_df, second_df], axis=0, ignore_index=True)
    df["model"] = df["model"].replace({"d": "D", "y": "Y", "x": "X", "pd": "Pd", "pdp": "Pdp"})
    df = df.dropna(subset=["rf_latent_cor"])  # avoid NaNs reaching the plotter

    order = ["D", "Y", "X", "Pd", "Pdp"]
    hue_order = ["first", "second"]
    custom_palette = {"first": "#4E6C97", "second": "#8D4A55"}

    plt.figure(figsize=(6.8, 5))
    ax = sns.boxplot(
        data=df, x="model", y="rf_latent_cor", hue="omic",
        order=order, hue_order=hue_order, dodge=True, palette=custom_palette
    )
    # jittered points (single color; avoid hue palette length mismatch)
    sns.stripplot(
        data=df, x="model", y="rf_latent_cor",
        order=order, hue="omic", hue_order=hue_order,
        dodge=True, alpha=0.25, color="black", linewidth=0.2, jitter=0.2
    )
    # de-dupe legends created by both plots
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], title="Input omic", loc="upper left")

    # define pairs and keep only those with data on both sides
    all_pairs = [
        (("D", "first"), ("Y", "first")),
        (("D", "first"), ("X", "first")),
        (("D", "first"), ("Pd", "first")),
        (("D", "first"), ("Pdp", "first")),
        (("Y", "first"), ("X", "first")),
        (("Y", "first"), ("Pd", "first")),
        (("Y", "first"), ("Pdp", "first")),
        (("X", "first"), ("Pd", "first")),
        (("X", "first"), ("Pdp", "first")),
        (("Pd", "first"), ("Pdp", "first")),
        (("X", "second"), ("Pd", "second")),
        (("X", "second"), ("Pdp", "second")),
        (("Pd", "second"), ("Pdp", "second")),
    ]
    def _group_vals(m, o):
        return df[(df["model"] == m) & (df["omic"] == o)]["rf_latent_cor"].values
    pairs = [p for p in all_pairs if len(_group_vals(*p[0])) and len(_group_vals(*p[1]))]

    # compute MWU p-values then FDR-BH
    p_vals = []
    for (g1, g2) in pairs:
        v1 = _group_vals(*g1)
        v2 = _group_vals(*g2)
        stat, p = mannwhitneyu(v1, v2, alternative="two-sided")
        p_vals.append(p)
    _, p_vals_fdr, _, _ = multipletests(p_vals, method="fdr_bh")

    # annotate: force using the boxplot geometry to avoid PathCollection issues
    annotator = Annotator(
        ax, pairs, data=df, x="model", y="rf_latent_cor", hue="omic",
        order=order, hue_order=hue_order, plot='boxplot'
    )
    annotator.configure(test=None, text_format="star", loc="inside",
                        verbose=0, hide_non_significant=True)
    annotator.set_pvalues(p_vals_fdr)
    annotator.annotate()

    # finish
    ax.set_ylabel("Phenotype correlation")
    # ax.set_ylim(0.1, 0.75)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/phenotype_prediction.png", dpi=300)
    plt.close()


def merge_csv_files(directory, model, cross_omic_parallel_models=False):
    # Get all CSV files in the directory
    csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]

    # Initialize an empty DataFrame to store merged data
    merged_df = pd.DataFrame()

    # Iterate over each CSV file and merge into the DataFrame
    for file in csv_files:
        # For the parallel models, ignore the first omic prediction files when merging the cross omic prediction results
        if cross_omic_parallel_models and "first_omic_prediction" in file:
            continue
        file_path = os.path.join(directory, file)
        file_name = file_path.split('/')[-1]
        match = re.search(r"f-(\d+)_", file_name)
        fold = int(match.group(1))
        match = re.search(r"f-\d+_(.*)\.csv", file_name)
        hyper = match.group(1)[:-4]
        if model in ["pd", "pdp"]:
            hyper = re.sub(r"_(LR2|hiddenlayers2|latentlayersize2)-[^_]+", "", hyper)
            hyper = hyper.replace("LR1", "LR")
            hyper = hyper.replace("hiddenlayers1", "hiddenlayers")
            hyper = hyper.replace("latentlayersize1", "latentlayersize")

        df = pd.read_csv(file_path)
        df["fold"] = fold
        df["hyper"] = hyper
        merged_df = pd.concat([merged_df, df], ignore_index=True)
    return merged_df


def plot_significant_stacked_bar(df, out_dir):
    """
    Generate stacked barplots showing, for each second omic feature, how many times it was well-predicted (significant) by each model across all hyperparameter settings and folds.
    Outputs:
        - second_omic_prediction_by_feature.png: Stacked barplot for all features
        - second_omic_prediction_by_feature_top_50.png: Stacked horizontal barplot for top 50 features
    Interpretation: Reveals which features are consistently well-predicted and which models excel for specific features.
    """

    # Keep only rows in the first repetition (col hyper ends with -rep-0)
    df = df[df["hyper"].str.endswith("rep-0")]

    # Add the 'Significant' column
    df['Significant'] = (df['Spearman_r'] > 0.3) & (df['FDR'] < 0.1)
    df["model"] = df["model"].replace({"d": "D", "y": "Y", "x": "X", "pd": "Pd", "pdp": "Pdp"})

    # Filter for significant rows
    significant_df = df[df['Significant']]

    # Group by metabolite and model to count significant times
    plot_data = significant_df.groupby(['Feature', 'model']).size().reset_index(name='Count')

    # Compute the total significant count for each metabolite
    metabolite_totals = plot_data.groupby('Feature')['Count'].sum().reset_index(
        name='TotalCount')
    
    metabolite_totals.to_csv("metabolite_totals.csv")

    # Merge totals back to plot_data and order by TotalCount
    plot_data = plot_data.merge(metabolite_totals, on='Feature')
    plot_data = plot_data.sort_values(by='TotalCount', ascending=False)

    # Pivot the data for a stacked bar plot
    stacked_data = plot_data.pivot(index='Feature', columns='model', values='Count').fillna(0)

    # Order models explicitly
    model_order = ["D", "Y", "X", "Pd", "Pdp"]
    stacked_data = stacked_data.reindex(columns=model_order, fill_value=0)

    # Define custom colors
    custom_colors = ['#335c67', '#99a88c', '#fff3b0', '#e09f3e', '#9e2a2b', '#540b0e']

    # # Order metabolites based on total significant occurrences
    stacked_data = stacked_data.loc[
        metabolite_totals.sort_values('TotalCount', ascending=False)['Feature']]

    # Helper function to generate the plot
    def generate_plot(data, title, file_name, show_xticks=True, rotate=False):
        if rotate:
            ax = data.plot(kind='barh', stacked=True, figsize=(8, 12), color=custom_colors, width=1.0)
            plt.xlabel('Well predicted count')
            plt.ylabel('Feature')
        else:
            ax = data.plot(kind='bar', stacked=True, figsize=(14, 7), color=custom_colors, width=1.0)
            plt.xlabel('Feature')
            plt.ylabel('Well predicted count')
        plt.title(title)
        plt.legend(title='Model', bbox_to_anchor=(1.01, 1), loc='upper left')
        if not show_xticks:
            ax.set_xticklabels([])
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, file_name))

    # Plot for all metabolites (no x-tick labels)
    generate_plot(
        stacked_data,
        title='All Features',
        file_name='second_omic_prediction_by_feature.png',
        show_xticks=False
    )

    stacked_data = stacked_data.loc[
        metabolite_totals.sort_values('TotalCount', ascending=True)['Feature']]
    # Limit to the top 50 features
    top_50_features = stacked_data.tail(50)

    # Plot for top 50 metabolites
    generate_plot(
        top_50_features,
        title='Top 50 Features',
        file_name='second_omic_prediction_by_feature_top_50.png',
        show_xticks=True,
        rotate=True
    )


def plot_heat_sig(df, out_dir, plot_only_significant=True, add_significance_stars=True):
    """
    Generate a heatmap of Spearman correlation coefficients for significant second omic features across models, hyperparameters, and folds.
    Asterisks mark significant correlations (FDR <0.1).
    Output: heatmap_of_significant_features_Spearman_r.png
    Interpretation: Visualizes feature-level prediction strength and model/hyperparameter/fold effects.
    """

    param = "Spearman_r" # Can be also "FDR" "

    # Step 1: Create a unique index for each hyperparameter combination
    hyper_map = {hyper: f"H{idx + 1}" for idx, hyper in enumerate(df["hyper"].unique())}
    df["hyper_index"] = df["hyper"].map(hyper_map)
    df["model"] = df["model"].replace({"d": "D", "y": "Y", "x": "X", "pd": "Pd", "pdp": "Pdp"})

    if plot_only_significant:
        # Step 2: Keep only well-predicted metabolites
        filtered_metabolites = df.loc[
            (df["FDR"] < 0.1) & (df["Spearman_r"] > 0.3), "Feature"
        ]

        df = df[df["Feature"].isin(filtered_metabolites)]

    # Step 3: Pivot the DataFrame
    df_pivot = df.pivot_table(
        index="Feature",
        columns=["model", "hyper_index", "fold"],
        # columns=["model", ],
        values=param
    )

    # Fix Header Display: Convert Multi-Index to Readable Labels
    df_pivot.columns = [
        "{} | {} | Fold {}".format(model, hyper, fold)
        for model, hyper, fold in df_pivot.columns
    ]

    # Step 4: Change the model order
    model_order = ["D", "Y", "X", "Pd", "Pdp"]
    df_pivot = df_pivot[sorted(
        df_pivot.columns,
        key=lambda x: model_order.index(x.split(" | ")[0])
    )]

    # Step 5: Plot the heatmap with the updated figure size
    fig, ax = plt.subplots(figsize=(36, 18))
    sns.heatmap(df_pivot, cmap="BrBG", annot=False, linewidths=0.5, ax=ax)  # coolwarm

    if add_significance_stars:
        # Step 6: Add significance stars to well-predicted cells
        for i, feature in enumerate(df_pivot.index):
            for j, col in enumerate(df_pivot.columns):
                # Extract corresponding model, hyperparameter, and fold
                model, hyper, fold = col.split(" | ")
                # Extract the fold number
                fold = int(fold.split()[-1])
                # Check if this cell is significant
                subset = df[
                    (df["Feature"] == feature) &
                    (df["model"] == model) &
                    (df["hyper_index"] == hyper) &
                    (df["fold"] == fold)
                    ]
                if not subset.empty and subset.iloc[0]["FDR"] < 0.1 and subset.iloc[0][param] > 0.3:
                    ax.text(j + 0.5, i + 0.5, "*", ha='center', va='center', color='black')


    # Step 7: Add rectangles around each model group
    col_positions = {model: [] for model in model_order}
    for idx, col_name in enumerate(df_pivot.columns):
        model_name = col_name.split(" | ")[0]
        col_positions[model_name].append(idx)

    for model, positions in col_positions.items():
        start = min(positions)
        end = max(positions)
        rect = patches.Rectangle(
            (start, 0),
            end - start + 1,
            df_pivot.shape[0],
            linewidth=2, edgecolor='black', facecolor='none'
        )
        ax.add_patch(rect)

    # Step 8: Customize labels
    plt.title(f"{param} Heatmap by Model, Hyperparameter Index, and Fold")
    plt.xlabel("Model → Hyperparameter Index → Fold")
    plt.ylabel("Feature")

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()

    # Save the plot
    if plot_only_significant:
        fig_name = "heatmap_significant_second_features"
    else:
        fig_name = "heatmap_all_second_features"
    if add_significance_stars:
        fig_name += "_with_significance_stars"
    plt.savefig(f"{out_dir}/{fig_name}_{param}.png")


def main():

    # Read all models outputs
    df_dict = {}
    predictions_df_dict = {}
    for model in ["d", "y", "x", "pd", "pdp"]:
        model_output_path = f"{LOCAL_PATH}/results/{RUN_IDENTIFIER}/{model}_{OMIC1}_{OMIC2}_{PHENOTYPE_COL}/final.csv"
        model_df = pd.read_csv(model_output_path)
        model_df["model"] = model
        if model in ["d", "y", "x"]:
            model_df = model_df.rename(columns={"latent_layer_size": "latent_layer_size_1"})
        if model in ["d", "y"]:
            model_df = model_df.rename(columns={"significant_correlations": "significant_correlations2"})
            model_df = model_df.rename(columns={"rf_latent_cor": "rf_latent_cor1"})
        df_dict[model] = model_df

        # Read the second omic prediction statistics for each model for each feature
        second_omic_tables_dir = f"{LOCAL_PATH}/results/{RUN_IDENTIFIER}/{model}_{OMIC1}_{OMIC2}_{PHENOTYPE_COL}/cross_omic_predictions"
        if model in ["d", "y", "x"]:
            predictions_df_dict[model] =  merge_csv_files(second_omic_tables_dir, model)
        else: # model in ["pd", "pdp"]
            predictions_df_dict[model] =  merge_csv_files(second_omic_tables_dir, model, cross_omic_parallel_models=True)
        predictions_df_dict[model]["model"] = model

    df = pd.concat(df_dict)
    predictions_df = pd.concat(predictions_df_dict)

    out_dir = f"{LOCAL_PATH}/results/{RUN_IDENTIFIER}/summary_{OMIC1}_{OMIC2}_{PHENOTYPE_COL}"
    if not os.path.exists(f"{out_dir}/"):
        os.makedirs(f"{out_dir}")

    second_omic_prediction(df.copy(), out_dir)
    second_omic_prediction_percentage(df.copy(), out_dir)
    second_omic_prediction_err_bar(df.copy(), out_dir)
    phenotype_prediction(df.copy(), out_dir)
    plot_significant_stacked_bar(predictions_df.copy(), out_dir)
    plot_heat_sig(predictions_df.copy(), out_dir, True, False)


if __name__ == "__main__":
    main()

