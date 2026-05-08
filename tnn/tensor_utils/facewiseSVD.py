import torch

#function for computing Facewise SVD of tensor A. 

def facewiseSVD(A, k=None):
   
    # Get the shape of the input tensor A
    orig_shape = A.shape
    #Extracts the last three dimensions: n1, n2, n3
    n1, n2, n3 = orig_shape[-3:]
    # Gets the batch shape (all dimensions except the last three)
    batch_shape = orig_shape[:-3]
    # Calculate the batch size (product of batch_shape, or 1 if no batch)
    batch_size = 1
    if batch_shape:
        batch_size = int(torch.prod(torch.tensor(batch_shape)))
    # Reshape A to (batch_size, n1, n2, n3) for easier processing
    A_batch = A.reshape(batch_size, n1, n2, n3)

    # Prepare lists to collect U, S, V for each slice
    U_list, S_list, V_list = [], [], []
    # Loop over each frontal slice along the last dimension (n3)
    for idx in range(n3):
        # Select the idx-th frontal slice: shape (batch_size, n1, n2)
        A_slice = A_batch[..., :, :, idx]
        # Compute SVD for this slice (batched if batch_size > 1)
        U, S, Vh = torch.linalg.svd(A_slice, full_matrices=False)
        # If k is specified and > 0, truncate to rank k
        if k is not None and k > 0:
            U = U[..., :,:k]  # Take first k columns of U
            S = S[..., :k]    # Take first k singular values
            Vh = Vh[..., :k, :]  # Take first k rows of Vh
            p = k
        else:
            # Otherwise, use full or economic SVD (p = min(n1, n2))
            p = min(n1, n2)
        # Append U, S, V (V = Vh^T) for this slice to the lists
        U_list.append(U)
        S_list.append(S)
        V_list.append(Vh.transpose(-2, -1))


    #U_list, S_list, V_list are lists svds for each slice. 
    # Stack the results along the last dimension 
    U = torch.stack(U_list, dim=-1)  # Has shape (batch_size, n1, p, n3)
    S = torch.stack(S_list, dim=-1)  # Has shape (batch_size, p, n3)
    V = torch.stack(V_list, dim=-1)  # Has shape (batch_size, n2, p, n3)

    # Build the diagonal S matrix for each slice
    S_diag = torch.zeros(batch_size, p, p, n3, dtype=S.dtype, device=S.device)
    for i in range(p):
        # Place the i-th singular value along the diagonal for each slice
        S_diag[:, i, i, :] = S[:, i, :]

    # Reshape U, S_diag, V back to original batch shape 
    if batch_shape: 
        U = U.reshape(*batch_shape, n1, p, n3)
        S_diag = S_diag.reshape(*batch_shape, p, p, n3)
        V = V.reshape(*batch_shape, n2, p, n3)
    else:
        # If no batch, remove the batch dimension
        U = U.squeeze(0)
        S_diag = S_diag.squeeze(0)
        V = V.squeeze(0)
    # Return U, S, V
    return U, S_diag, V

