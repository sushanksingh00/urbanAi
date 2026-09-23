import numpy as np
import scipy.sparse as sp
import pickle
import torch

def load_adjacency_matrix(filepath: str) -> np.ndarray:
    """
    Loads the adjacency matrix from a pickle file.
    The pickle file is expected to contain a tuple: (sensor_ids, sensor_id_to_ind, adj_mx)
    """
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
    except UnicodeDecodeError:
        with open(filepath, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            
    if len(data) == 3:
        _, _, adj_mx = data
    else:
        adj_mx = data
        
    return adj_mx

def calculate_normalized_laplacian(adj: np.ndarray) -> sp.coo_matrix:
    """
    Calculates the normalized Laplacian: L = I - D^-1/2 A D^-1/2
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_scaled_laplacian(adj_mx: np.ndarray, lambda_max: float = 2, undirected: bool = True) -> sp.csr_matrix:
    """
    Calculates the scaled Laplacian for Chebyshev polynomials approximation.
    """
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    
    L = calculate_normalized_laplacian(adj_mx)
    
    if lambda_max is None:
        # Compute the largest eigenvalue
        lambda_max, _ = sp.linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
        
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32)

def calculate_cheb_poly(L: sp.csr_matrix, Ks: int) -> torch.Tensor:
    """
    Calculates Chebyshev polynomials up to order Ks.
    Returns a tensor of shape (Ks, N, N).
    """
    n = L.shape[0]
    # Convert L to dense for easy tensor creation
    L_dense = L.todense()
    
    cheb_polynomials = [np.identity(n), L_dense.copy()]
    for i in range(2, Ks):
        cheb_polynomials.append(2 * np.dot(L_dense, cheb_polynomials[i - 1]) - cheb_polynomials[i - 2])
        
    return torch.tensor(np.array(cheb_polynomials), dtype=torch.float32)
