import pandas as pd

def summarize_shap_by_expression_quantiles(cell_score_df, shap_df):
    """
    Summarizes SHAP values for each gene program by stratifying cells based on gene-program expression levels.

    For each gene program:
    - Cells are divided into Low, Medium, and High expression groups using quantile-based binning.
    - The mean SHAP values for Logistic Regression (_shap_LR) and/or Random Forest (_shap_RF) 
      are calculated within each expression group, depending on availability.

    Parameters
    ----------
    cell_score_df : pandas.DataFrame
        DataFrame with shape (n_cells, n_gene_programs) where values are expression scores for each gene program.

    shap_df : pandas.DataFrame
        DataFrame with shape (n_cells, n_shap_features) where columns may include:
        '{gene_program}_shap_LR' and/or '{gene_program}_shap_RF'.

    Returns
    -------
    result_df : pandas.DataFrame
        A summary DataFrame indexed by gene program with the following columns if available:
        ['Low_Shap_LR', 'Med_Shap_LR', 'High_Shap_LR', 
         'Low_Shap_RF', 'Med_Shap_RF', 'High_Shap_RF']
    """
    summary_rows = []

    for gp in cell_score_df.columns:
        shap_lr_col = f"{gp}_shap_LR"
        shap_rf_col = f"{gp}_shap_RF"

        has_lr = shap_lr_col in shap_df.columns
        has_rf = shap_rf_col in shap_df.columns
        if not has_lr and not has_rf:
            print("Gene-PROGRAM", gp, "is not included")
            continue  # Skip if neither SHAP column exists

        # Quantile binning of expression levels
        scores = cell_score_df[gp]
        try:
            bins = pd.qcut(scores, q=3, labels=["Low", "Med", "High"])
        except ValueError:
            continue  # Skip if not enough unique values for 3 bins

        df = pd.DataFrame({"Expression_Level": bins})
        if has_lr:
            df["SHAP_LR"] = shap_df[shap_lr_col]
        if has_rf:
            df["SHAP_RF"] = shap_df[shap_rf_col]

        group_means = df.groupby("Expression_Level").mean()

        summary = {"Gene_Program": gp}
        if has_lr:
            summary.update({
                "Low_Shap_LR": group_means.loc["Low", "SHAP_LR"],
                "Med_Shap_LR": group_means.loc["Med", "SHAP_LR"],
                "High_Shap_LR": group_means.loc["High", "SHAP_LR"],
            })
        if has_rf:
            summary.update({
                "Low_Shap_RF": group_means.loc["Low", "SHAP_RF"],
                "Med_Shap_RF": group_means.loc["Med", "SHAP_RF"],
                "High_Shap_RF": group_means.loc["High", "SHAP_RF"],
            })

        summary_rows.append(summary)

    result_df = pd.DataFrame(summary_rows).set_index("Gene_Program")
    return result_df