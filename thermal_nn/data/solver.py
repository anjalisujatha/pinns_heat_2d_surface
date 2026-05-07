import numpy as np
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import cg


def solve_fdm_3d(kappa, Q, res=32, dx=1.0):
    """
    Vectorized 3D FDM Solver using Conjugate Gradient.
    kappa: (res, res, res) - Thermal conductivity
    Q: (res, res, res) - Heat source
    """
    # Flatten for sparse matrix operations
    k = kappa.flatten()

    # 1D finite difference stencil: [-1, 2, -1]
    main_diag = 2.0 * np.ones(res)
    off_diag = -1.0 * np.ones(res - 1)
    L1D = diags([off_diag, main_diag, off_diag], [-1, 0, 1]) / (dx ** 2)

    # 3D Laplacian via Kronecker products (for constant kappa simplification)
    # For variable kappa, we use a more complex sparse assembly
    I = eye(res)
    L3D = kron(I, kron(I, L1D)) + kron(I, kron(L1D, I)) + kron(L1D, kron(I, I))

    # Simple Conductivity mapping (approximation for demo)
    A = L3D.multiply(k[:, np.newaxis])

    # Solve Ax = b
    T_flat, info = cg(A, Q.flatten(), rtol=1e-6)
    return T_flat.reshape((res, res, res))


if __name__ == "__main__":
    res = 16
    rng = np.random.default_rng(0)
    kappa = rng.uniform(0.5, 2.0, (res, res, res))
    Q = np.zeros((res, res, res))
    Q[res // 2, res // 2, res // 2] = 1.0

    T = solve_fdm_3d(kappa, Q, res=res)
    print(f"Solved {res}^3 grid — T min={T.min():.4f}, max={T.max():.4f}, mean={T.mean():.4f}")
