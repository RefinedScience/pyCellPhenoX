from scipy.sparse import issparse
from scipy.sparse import csr_matrix
import numpy as np

def fast_sparse_correlation(Y, X_sparse, chunks=True):
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

    n_cells, n_genes = X_sparse.shape
    Y = np.asarray(Y)
    chunk_size = 1000

    # Center Y 
    Y_mean = Y.mean(axis=0)
    Y_centered = Y - Y_mean
    Y_std = Y_centered.std(axis=0, ddof=0)[:, np.newaxis]  # (n_vars, 1)
    
    if chunks:
      corr_chunks = []
      for start in range(0, n_genes, chunk_size):
        end = min(start + chunk_size, n_genes)
        X_chunk = X_sparse[:, start:end].copy()

        # Center the sparse chunk without densifying
        X_mean = X_chunk.mean(axis=0).A1  # mean across cells
        #X_chunk.data -= np.take(X_mean, X_chunk.indices)
        X_chunk.data -= X_mean[X_chunk.indices - start]

        # Dot product between centered Y and X
        numer = Y_centered.T @ X_chunk  # shape: (n_vars, chunk_size)

        # Compute std for each gene in the chunk
        X_chunk_sq = X_chunk.copy()
        X_chunk_sq.data **= 2
        X_std = np.sqrt(X_chunk_sq.mean(axis=0)).A1  # shape: (chunk_size,)

        # Avoid divide-by-zero
        denom = (n_cells * (Y_std * X_std[np.newaxis, :]))
        denom[denom == 0] = np.nan  # optional: mark as NaN where std is 0

        # Correlation for this chunk
        corr_chunk = numer / denom  # shape: (n_vars, chunk_size)
        corr_chunks.append(corr_chunk)
      # Concatenate all chunks
      corr_matrix = np.hstack(corr_chunks)
    else:
      # Center X without densifying
      X_mean = X_sparse.mean(axis=0).A1  # (n_genes,)
      X_centered = X_sparse.copy()
      X_centered.data -= np.take(X_mean, X_centered.indices)

      # Compute numerator: dot product between centered Y and X
      numer = Y_centered.T @ X_centered  # shape: (n_vars, n_genes)
    
      # Compute std of X efficiently
      X_centered_squared = X_centered.copy()
      X_centered_squared.data **= 2
      X_std = np.sqrt(X_centered_squared.mean(axis=0)).A1  # (n_genes,)
    
      # Compute correlation matrix
      corr_matrix = numer / (n_cells * (Y_std * X_std[np.newaxis, :]))

    return corr_matrix
