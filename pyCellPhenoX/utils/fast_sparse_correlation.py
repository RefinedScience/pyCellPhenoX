from scipy.sparse import issparse
from scipy.sparse import csr_matrix
import numpy as np

def fast_sparse_correlation(Y, X_sparse):
    """
    Compute correlation between each column of dense Y (cells × variables)
    and each column of sparse X (cells × genes), using matrix algebra.

    Returns: (n_vars × n_genes) array

    Running fast_sparse_correlation(res.ncorrs.to_numpy().reshape(-1, 1), adata_filtered.X)[0] 
       gives almost identical results to  
       np.corrcoef(res.ncorrs.to_numpy().reshape(1,-1), adata_filtered.X.todense(), rowvar=False)[0,1:]
       (Off by maybe 0.0001)
    """
    if not issparse(X_sparse):
        raise ValueError("X must be a sparse matrix.")

    n_cells = X_sparse.shape[0]
    Y = np.asarray(Y)

    # Center Y and X
    Y_mean = Y.mean(axis=0)
    Y_centered = Y - Y_mean

    X_mean = X_sparse.mean(axis=0).A1  # shape: (n_genes,)
    X_centered = csr_matrix(X_sparse - X_mean)# broadcasts over rows

    # Compute numerator: dot product between centered Y and X
    numer = Y_centered.T @ X_centered  # shape: (n_vars, n_genes)

    # Compute std deviations
    Y_std = Y_centered.std(axis=0, ddof=0)[:, np.newaxis]  # shape: (n_vars, 1)
    X_sq_mean = X_centered.power(2).mean(axis=0).A1
    X_std = np.sqrt(X_sq_mean)  # shape: (n_genes,)

    # Compute correlation matrix
    corr_matrix = numer / (n_cells * (Y_std * X_std[np.newaxis, :]))

    return corr_matrix