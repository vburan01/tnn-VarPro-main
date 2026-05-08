import torch
from tnn.tensor_utils import facewiseSVD as fSVD
from tnn.tensor_utils.modek_product import modek_product



# Computes the tSVDM of a tensor A w.r.t an orthogonal matrix M with optional truncation to rank k. 

def tSVDM(A,M, k= None): 
  
    # Apply transformation to obtain tranform domain representation of A 
    A_t = modek_product(A, M, k=-1)

    # Compute facewise SVD of transformed A
    U, S, V = fSVD.facewiseSVD(A_t, k)

    # Apply inverse transformation to U, S, V to return to original domain 
    U = modek_product(U, M.T, k=-1)
    S = modek_product(S, M.T, k=-1)
    V = modek_product(V, M.T, k=-1)

    return U, S, V