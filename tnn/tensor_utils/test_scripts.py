
import torch 
from tnn.tensor_utils import facewiseSVD as fSVD
from tnn.tensor_utils import tSVDM as tsvdm
from tnn.tensor_utils import m_product as mp

if __name__ == "__main__":
    # Tests the facewiseSVD function
    A = torch.randn(4, 5, 3)  # Example tensor of shape (n1, n2, n3)
    U, S, V = fSVD.facewiseSVD(A)  # Truncated SVD with k=2
    print("U shape:", U.shape)  # Should be (4, 4, 3)
    print("S shape:", S.shape)  # Should be (2, 4, 3)
    print("V shape:", V.shape)  # Should be (5, 4, 3)


    # test on Identity transformation: tSVDM should match facewiseSVD
    U_t, S_t, V_t = tsvdm.tSVDM(A, M=torch.eye(A.shape[-1]))
    print("[tSVDM, identity] U shape:", U_t.shape)
    print("[tSVDM, identity] S shape:", S_t.shape)
    print("[tSVDM, identity] V shape:", V_t.shape)
    print("Max diff U:", (U - U_t).abs().max().item())
    print("Max diff S:", (S - S_t).abs().max().item())
    print("Max diff V:", (V - V_t).abs().max().item())

    # 2. Orthogonal transformation: tSVDM with random orthogonal M
    n3 = A.shape[-1]
    Q, _ = torch.linalg.qr(torch.randn(n3, n3))  # Random orthogonal matrix
    Uo, So, Vo = tsvdm.tSVDM(A, M=Q, k=2)
    print("[tSVDM, orthogonal] U shape:", Uo.shape)
    print("[tSVDM, orthogonal] S shape:", So.shape)
    print("[tSVDM, orthogonal] V shape:", Vo.shape)

    # 3. Check reconstruction
    # Reconstruct A from U, S, V
    A_rec = torch.zeros_like(A)
    for i in range(A.shape[-1]):
        A_rec[..., :, :, i] = (U_t[..., :, :, i] @ S_t[..., :, :, i] @ V_t[..., :, :, i].transpose(-2, -1))
    print("Reconstruction error:", torch.norm(A - A_rec).item())


