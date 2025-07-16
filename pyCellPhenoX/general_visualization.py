import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

def plot_cna_umap_panels(adata_filtered, query_column, query_value, baseline_value, plot_dir="./", sig_value=0.1):
    """
    Plot a two-panel UMAP visualization showing CNA coefficient significance and cell type annotations.

    Parameters
    ----------
    adata_filtered : AnnData
        Annotated data matrix with `obsm["X_umap_ccmt"]` and relevant `.obs` columns.
    
    query_column : str
        The name of the column used in the CNA model (e.g., "sample_type").
    
    query_value : str
        The class being predicted in the CNA model.
    
    baseline_value : str
        The reference class used for contrast in the CNA model.
    
    plot_dir : str, optional (default="./")
        Path to directory where the plot will be saved.
    
    sig_value : float, optional (default=0.1)
        Significance threshold for filtering `coef_fdr` values.

    Returns
    -------
    None
        Saves a two-panel UMAP figure and displays it.
    """
    # Extract UMAP coordinates and metadata
    umap_coords = adata_filtered.obsm["X_umap_ccmt"]
    umap_df = pd.DataFrame(umap_coords, columns=["UMAP1", "UMAP2"], index=adata_filtered.obs_names)
    umap_df["coef"] = adata_filtered.obs["coef"]
    umap_df["coef_fdr"] = adata_filtered.obs["coef_fdr"]
    umap_df["grimes.l1_label"] = adata_filtered.obs["grimes.l1_label"]

    # Create mask for significant coefficients
    sig_mask = umap_df["coef_fdr"] < sig_value

    # Set up figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [4, 3.2]})

    # --- LEFT PLOT: Coefficient significance ---
    axs[0].scatter(
        umap_df["UMAP1"], umap_df["UMAP2"],
        color="lightgrey", s=10, alpha=0.5
    )
    sc0 = axs[0].scatter(
        umap_df.loc[sig_mask, "UMAP1"],
        umap_df.loc[sig_mask, "UMAP2"],
        c=umap_df.loc[sig_mask, "coef"],
        cmap="seismic", s=10, alpha=0.9
    )
    axs[0].set_title(f'UMAP colored by CNA coefficient for predicting\n{query_value} from {baseline_value} in {query_column}\nFDR < {sig_value}')
    axs[0].set_xlabel("UMAP 1")
    axs[0].set_ylabel("UMAP 2")
    axs[0].set_xticks([]); axs[0].set_yticks([])
    cbar0 = fig.colorbar(sc0, ax=axs[0], label="coef")
    cbar0.set_alpha(1)

    # --- RIGHT PLOT: Cell type annotations ---
    cell_types = umap_df["grimes.l1_label"].astype(str)
    unique_types = cell_types.unique()
    colors = plt.cm.tab20.colors
    type_to_color = {k: colors[i % len(colors)] for i, k in enumerate(unique_types)}
    point_colors = cell_types.map(type_to_color)

    axs[1].scatter(
        umap_df["UMAP1"], umap_df["UMAP2"],
        c=point_colors, s=10, alpha=0.8
    )
    axs[1].set_title("Cell Type Annotation (grimes.l1_label)")
    axs[1].set_xlabel("UMAP 1")
    axs[1].set_ylabel("UMAP 2")
    axs[1].set_xticks([]); axs[1].set_yticks([])

    # Add legend
    legend_elements = [Patch(color=type_to_color[k], label=k) for k in sorted(unique_types)]
    axs[1].legend(
        handles=legend_elements,
        title="Cell Type",
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        fontsize='small'
    )

    plt.tight_layout()
    plt.savefig(f"{plot_dir}/CNA_coef_FDR{sig_value}.png", bbox_inches="tight", pad_inches=0.3, transparent=False)
    plt.show()


def plot_boxplot_cell_type_categories(adata_filtered, cell_type_use, sig_thresh = 0.1, fig_width=10, level="cell", plot_dir="./"):
    """
    Plot the CNA correlation coefficients, while considering the significance threshold of coef_fdr.

    This function generates a boxplot (with overlaid jittered points) of interpretable scores 
    grouped by cell type. It allows for plotting either at the individual cell level or by 
    summarizing (median) scores at the sample level. The plot is saved to disk and also shown.

    Parameters
    ----------
    adata_filtered : AnnData
        The AnnData object containing `.obs` with interpretable scores and metadata.
    
    cell_type_use : str
        The column name in `adata_filtered.obs` that defines the cell type categories.

    sig_thresh: float, optional (default=0.1)
        Threshold for coef_fdr to consider significant and therefore include as colored

    fig_width : int, optional (default=10)
        Width of the output plot in inches.

    level : str, optional (default="cell")
        Level at which to compute the plot. Must be one of:
        - "cell": interpretable scores per individual cell
        - "sample": median interpretable scores per sample per cell type

    plot_dir : str, optional (default="./")
        Path to the directory where the plot will be saved.

    model_type : str, optional (default="LR")
        The model identifier used to extract interpretable scores from `.obs`. 
        E.g., "LR" for logistic regression or "RF" for random forest.
        The column used will be `interpretable_score_{model_type}`.

    Returns
    -------
    None
        Saves and displays a boxplot of interpretable scores by cell type.
    """
    if level == "sample":
        # Sample-level aggregation
        obs_df = adata_filtered.obs[["coef", "coef_fdr", cell_type_use, "sample_id"]].dropna()

        summary_df = (
            obs_df
            .groupby(["sample_id", cell_type_use])
            .agg(
                coef=("coef", "median"),
                coef_fdr=("coef_fdr", "median")
            )
            .reset_index()
            .rename(columns={cell_type_use: "cell_type"})
        )
        summary_df = summary_df.dropna()

        group_counts = (
            summary_df.groupby("cell_type")["sample_id"]
            .nunique()
            .to_dict()
            )
        sig_counts = (
            summary_df[summary_df["coef_fdr"] < sig_thresh]
            .groupby("cell_type")["sample_id"]
            .nunique()
            .to_dict()
        )


    elif level == "cell":
        # Cell-level data
        summary_df = pd.DataFrame({
            "coef": adata_filtered.obs["coef"],
            "coef_fdr": adata_filtered.obs["coef_fdr"],
            "cell_type": adata_filtered.obs[cell_type_use]
        }).dropna()

        group_counts = summary_df["cell_type"].value_counts().to_dict()
        sig_counts = summary_df[summary_df["coef_fdr"] < sig_thresh]["cell_type"].value_counts().to_dict()
    else:
        raise ValueError("`level` must be either 'cell' or 'sample'")


    # Plotting
    plt.figure(figsize=(fig_width, 6))
    ax = plt.gca()

    sns.boxplot(
        data=summary_df,
        x="cell_type",
        y="coef",
        palette="Set3",
        showfliers=False,
        ax=ax
    )

    sns.stripplot(
        data=summary_df,
        x="cell_type",
        y="coef",
        color="black",
        alpha=0.3,
        jitter=0.25,
        size=4 if level == "sample" else 2,
        ax=ax
    )

    # Annotation
    ymin = ax.get_ylim()[0]
    for i, tick in enumerate(ax.get_xticks()):
        label = ax.get_xticklabels()[i].get_text()
        n_total = group_counts.get(label, 0)
        n_sig = sig_counts.get(label, 0)

        ax.text(i, ymin + 0.05, f"n={n_total}", ha="center", va="bottom", fontsize=9, color="gray")
        if n_sig > 0:
            ax.text(i, ymin + 0.10, f"n_sig={n_sig}", ha="center", va="bottom", fontsize=9, color="darkred")

    # Highlight significant cell types
    for i, label in enumerate(ax.get_xticklabels()):
        name = label.get_text()
        if sig_counts.get(name, 0) > 0:
            label.set_color("darkred")
            label.set_fontweight("bold")

    # Axes and styling
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel(f"Cell Type ({cell_type_use})")
    ylabel = "Mean CNA Coefficient per Sample" if level == "sample" else "CNA Coefficient"
    plt.ylabel(ylabel)
    plt.title(f"CNA Coefficients by Cell Type ({level.capitalize()} Level)\nFDR < {sig_thresh} shown in red")
    plt.tight_layout()
    if level == "sample":
        plt.savefig("".join([plot_dir, "/CNA_coef_by_Celltype_", cell_type_use, "_sample.png"]), 
        bbox_inches="tight", pad_inches=0.3, transparent=False)
    elif level == "cell":
        plt.savefig("".join([plot_dir, "/CNA_coef_by_Celltype_", cell_type_use, "_cell.png"]), 
        bbox_inches="tight", pad_inches=0.3, transparent=False)
    plt.show()

def plot_violin_cell_type_categories(adata_filtered, cell_type_use, sig_thresh = 0.1, fig_width=10, level="cell", plot_dir="./"):
    """
    Plot the CNA correlation coefficients, while considering the significance threshold of coef_fdr.

    This function generates a boxplot (with overlaid jittered points) of interpretable scores 
    grouped by cell type. It allows for plotting either at the individual cell level or by 
    summarizing (median) scores at the sample level. The plot is saved to disk and also shown.

    Parameters
    ----------
    adata_filtered : AnnData
        The AnnData object containing `.obs` with interpretable scores and metadata.
    
    cell_type_use : str
        The column name in `adata_filtered.obs` that defines the cell type categories.

    sig_thresh: float, optional (default=0.1)
        Threshold for coef_fdr to consider significant and therefore include as colored

    fig_width : int, optional (default=10)
        Width of the output plot in inches.

    level : str, optional (default="cell")
        Level at which to compute the plot. Must be one of:
        - "cell": interpretable scores per individual cell
        - "sample": median interpretable scores per sample per cell type

    plot_dir : str, optional (default="./")
        Path to the directory where the plot will be saved.

    model_type : str, optional (default="LR")
        The model identifier used to extract interpretable scores from `.obs`. 
        E.g., "LR" for logistic regression or "RF" for random forest.
        The column used will be `interpretable_score_{model_type}`.

    Returns
    -------
    None
        Saves and displays a boxplot of interpretable scores by cell type.
    """
    if level == "sample":
        # Sample-level aggregation
        obs_df = adata_filtered.obs[["coef", "coef_fdr", cell_type_use, "sample_id"]].dropna()

        summary_df = (
            obs_df
            .groupby(["sample_id", cell_type_use])
            .agg(
                coef=("coef", "median"),
                coef_fdr=("coef_fdr", "median")
            )
            .reset_index()
            .rename(columns={cell_type_use: "cell_type"})
        )
        summary_df = summary_df.dropna()

        group_counts = (
            summary_df.groupby("cell_type")["sample_id"]
            .nunique()
            .to_dict()
            )
        sig_counts = (
            summary_df[summary_df["coef_fdr"] < sig_thresh]
            .groupby("cell_type")["sample_id"]
            .nunique()
            .to_dict()
        )


    elif level == "cell":
        # Cell-level data
        summary_df = pd.DataFrame({
            "coef": adata_filtered.obs["coef"],
            "coef_fdr": adata_filtered.obs["coef_fdr"],
            "cell_type": adata_filtered.obs[cell_type_use]
        }).dropna()

        group_counts = summary_df["cell_type"].value_counts().to_dict()
        sig_counts = summary_df[summary_df["coef_fdr"] < sig_thresh]["cell_type"].value_counts().to_dict()
    else:
        raise ValueError("`level` must be either 'cell' or 'sample'")


    # Plotting
    plt.figure(figsize=(fig_width, 6))
    ax = plt.gca()

    sns.violinplot(
        data=summary_df,
        x="cell_type",
        y="interpretable_score",
        palette="Set3",
        inner="box",  # show median line
        cut=0,
        scale="width",
        ax=ax
    )


    # Annotation
    ymin = ax.get_ylim()[0]
    for i, tick in enumerate(ax.get_xticks()):
        label = ax.get_xticklabels()[i].get_text()
        n_total = group_counts.get(label, 0)
        n_sig = sig_counts.get(label, 0)

        ax.text(i, ymin + 0.05, f"n={n_total}", ha="center", va="bottom", fontsize=9, color="gray")
        if n_sig > 0:
            ax.text(i, ymin + 0.10, f"n_sig={n_sig}", ha="center", va="bottom", fontsize=9, color="darkred")

    # Highlight significant cell types
    for i, label in enumerate(ax.get_xticklabels()):
        name = label.get_text()
        if sig_counts.get(name, 0) > 0:
            label.set_color("darkred")
            label.set_fontweight("bold")

    # Axes and styling
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel(f"Cell Type ({cell_type_use})")
    ylabel = "Mean CNA Coefficient per Sample" if level == "sample" else "CNA Coefficient"
    plt.ylabel(ylabel)
    plt.title(f"CNA Coefficients by Cell Type ({level.capitalize()} Level)\nFDR < {sig_thresh} shown in red")
    plt.tight_layout()
    if level == "sample":
        plt.savefig("".join([plot_dir, "/CNA_coef_by_Celltype_", cell_type_use, "_sample.png"]), 
        bbox_inches="tight", pad_inches=0.3, transparent=False)
    elif level == "cell":
        plt.savefig("".join([plot_dir, "/CNA_coef_by_Celltype_", cell_type_use, "_cell.png"]), 
        bbox_inches="tight", pad_inches=0.3, transparent=False)
    plt.show()




def plot_boxplot_cell_type_categories_int(adata_filtered, cell_type_use, fig_width=10, level="cell", plot_dir="./", model_type="LR"):
    """
    Plot interpretable scores by cell type at the cell or sample level.

    This function generates a boxplot (with overlaid jittered points) of interpretable scores 
    grouped by cell type. It allows for plotting either at the individual cell level or by 
    summarizing (median) scores at the sample level. The plot is saved to disk and also shown.

    Parameters
    ----------
    adata_filtered : AnnData
        The AnnData object containing `.obs` with interpretable scores and metadata.
    
    cell_type_use : str
        The column name in `adata_filtered.obs` that defines the cell type categories.

    fig_width : int, optional (default=10)
        Width of the output plot in inches.

    level : str, optional (default="cell")
        Level at which to compute the plot. Must be one of:
        - "cell": interpretable scores per individual cell
        - "sample": median interpretable scores per sample per cell type

    plot_dir : str, optional (default="./")
        Path to the directory where the plot will be saved.

    model_type : str, optional (default="LR")
        The model identifier used to extract interpretable scores from `.obs`. 
        E.g., "LR" for logistic regression or "RF" for random forest.
        The column used will be `interpretable_score_{model_type}`.

    Returns
    -------
    None
        Saves and displays a boxplot of interpretable scores by cell type.
    """
    if level == "sample":
        # Sample-level aggregation
        obs_df = adata_filtered.obs[[f"interpretable_score_{model_type}",  cell_type_use, "sample_id"]].dropna()

        summary_df = (
            obs_df
            .groupby(["sample_id", cell_type_use])
            .agg(
                interpretable_score=(f"interpretable_score_{model_type}", "median"),
            )
            .reset_index()
            .rename(columns={cell_type_use: "cell_type"})
        )
        summary_df = summary_df.dropna()

        group_counts = (
            summary_df.groupby("cell_type")["sample_id"]
            .nunique()
            .to_dict()
            )

    elif level == "cell":
        # Cell-level data
        summary_df = pd.DataFrame({
            "interpretable_score": adata_filtered.obs[f"interpretable_score_{model_type}"],
            "cell_type": adata_filtered.obs[cell_type_use]
        }).dropna()

        group_counts = summary_df["cell_type"].value_counts().to_dict()
    else:
        raise ValueError("`level` must be either 'cell' or 'sample'")


    # Plotting
    plt.figure(figsize=(fig_width, 6))
    ax = plt.gca()

    sns.boxplot(
        data=summary_df,
        x="cell_type",
        y="interpretable_score",
        palette="Set3",
        showfliers=False,
        ax=ax
    )

    sns.stripplot(
        data=summary_df,
        x="cell_type",
        y="interpretable_score",
        color="black",
        alpha=0.3,
        jitter=0.25,
        size=4 if level == "sample" else 2,
        ax=ax
    )

    # Annotation
    ymin = ax.get_ylim()[0]
    for i, tick in enumerate(ax.get_xticks()):
        label = ax.get_xticklabels()[i].get_text()
        n_total = group_counts.get(label, 0)

        ax.text(i, ymin + 0.05, f"n={n_total}", ha="center", va="bottom", fontsize=9, color="gray")


    # Axes and styling
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel(f"Cell Type ({cell_type_use})")
    ylabel = "Median Interpretable Score per Sample" if level == "sample" else "Interpretable Score"
    plt.ylabel(ylabel)
    plt.title(f"Interpretable Scores by Cell Type ({level.capitalize()} Level: {model_type})")
    plt.tight_layout()
    if level == "sample":
        plt.savefig("".join([plot_dir, "/Interpretable_Score_by_Celltype_", cell_type_use, "_", model_type, "_sample.png"]), 
        bbox_inches="tight", pad_inches=0.3, transparent=False)
    elif level == "cell":
        plt.savefig("".join([plot_dir, "/Interpretable_Score_by_Celltype_", cell_type_use, "_", model_type, "_cell.png"]), 
        bbox_inches="tight", pad_inches=0.3, transparent=False)
    plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_violin_cell_type_categories_int(adata_filtered, cell_type_use, fig_width=10, level="cell", plot_dir="./", model_type="LR"):
    """
    Plot interpretable scores by cell type at the cell or sample level using violin plots.

    This function generates a violin plot of interpretable scores grouped by cell type.
    It allows plotting at the individual cell level or by summarizing (median) scores at 
    the sample level. Median values are shown as lines within violins. The plot is saved 
    to disk and also displayed.

    Parameters
    ----------
    adata_filtered : AnnData
        The AnnData object containing `.obs` with interpretable scores and metadata.
    
    cell_type_use : str
        The column name in `adata_filtered.obs` that defines the cell type categories.

    fig_width : int, optional (default=10)
        Width of the output plot in inches.

    level : str, optional (default="cell")
        Level at which to compute the plot. Must be one of:
        - "cell": interpretable scores per individual cell
        - "sample": median interpretable scores per sample per cell type

    plot_dir : str, optional (default="./")
        Path to the directory where the plot will be saved.

    model_type : str, optional (default="LR")
        The model identifier used to extract interpretable scores from `.obs`. 
        E.g., "LR" for logistic regression or "RF" for random forest.
        The column used will be `interpretable_score_{model_type}`.

    Returns
    -------
    None
        Saves and displays a violin plot of interpretable scores by cell type.
    """
    if level == "sample":
        obs_df = adata_filtered.obs[[f"interpretable_score_{model_type}", cell_type_use, "sample_id"]].dropna()
        summary_df = (
            obs_df
            .groupby(["sample_id", cell_type_use])
            .agg(interpretable_score=(f"interpretable_score_{model_type}", "median"))
            .reset_index()
            .rename(columns={cell_type_use: "cell_type"})
        )
        group_counts = (
            summary_df.groupby("cell_type")["sample_id"]
            .nunique()
            .to_dict()
            )
    elif level == "cell":
        summary_df = pd.DataFrame({
            "interpretable_score": adata_filtered.obs[f"interpretable_score_{model_type}"],
            "cell_type": adata_filtered.obs[cell_type_use]
        }).dropna()
        #group_counts = summary_df["cell_type"].value_counts().to_dict()
        group_counts = (
            summary_df.groupby("cell_type")
            .nunique()
            .to_dict()
            )
    else:
        raise ValueError("`level` must be either 'cell' or 'sample'")
    # Plotting
    plt.figure(figsize=(fig_width, 6))
    ax = plt.gca()

    sns.violinplot(
        data=summary_df,
        x="cell_type",
        y="interpretable_score",
        palette="Set3",
        inner="box",  # show median line
        cut=0,
        scale="width",
        ax=ax
    )

    # Annotation
    # Ensure we only annotate visible ticks with valid labels
    ymin = ax.get_ylim()[0]
    for i, tick in enumerate(ax.get_xticks()):
        try:
            label_obj = ax.get_xticklabels()[i]
            label = label_obj.get_text()
            if not label:  # Skip empty labels
                continue
            n_total = group_counts.get(label, 0)
            ax.text(i, ymin*0.95, f"n={n_total}", ha="center", va="bottom", fontsize=9, color="gray")
        except IndexError:
            continue 

    # Axes and styling
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel(f"Cell Type ({cell_type_use})")
    ylabel = "Median Interpretable Score per Sample" if level == "sample" else "Interpretable Score"
    plt.ylabel(ylabel)
    plt.title(f"Interpretable Scores by Cell Type ({level.capitalize()} Level: {model_type})")
    #plt.tight_layout()

    filename = f"{plot_dir}/Interpretable_Score_by_Celltype_{cell_type_use}_{model_type}_{level}.png"
    plt.savefig(filename, bbox_inches="tight", pad_inches=0.3, transparent=False, dpi=200)
    plt.show()
