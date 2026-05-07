import torch


def get_polar_inverse_symplectic_matrix():
    """
    Returns the 6x6 inverse symplectic matrix (f_ij^2)^-1
    based on the basic brackets: {r, Pr}=1, {rho, theta}=1,
    {lambda, Ptheta}=1, {lambda, rho}=1[cite: 135, 136].
    """
    # State vector order: [r, Pr, theta, Ptheta, lambda, rho]
    J = torch.zeros((6, 6))

    # Brackets from Equation 28 in the paper [cite: 137]
    J[0, 1] = 1.0;
    J[1, 0] = -1.0  # {r, Pr} = 1
    J[2, 5] = -1.0;
    J[5, 2] = 1.0  # {theta, rho} = -1
    J[3, 4] = -1.0;
    J[4, 3] = 1.0  # {Ptheta, lambda} = -1
    J[4, 5] = 1.0;
    J[5, 4] = -1.0  # {lambda, rho} = 1

    return J