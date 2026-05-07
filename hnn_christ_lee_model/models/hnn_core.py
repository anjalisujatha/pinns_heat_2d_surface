import torch
import torch.nn as nn
try:
    from .symplectic_utlis import get_polar_inverse_symplectic_matrix
except ImportError:
    from symplectic_utlis import get_polar_inverse_symplectic_matrix


class ChristLeeHNN(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        # The MLP learns the scalar Hamiltonian function H(zeta)
        self.hamiltonian_net = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.register_buffer('J', get_polar_inverse_symplectic_matrix())

    def forward(self, x):
        x = x.requires_grad_(True)
        H = self.hamiltonian_net(x)

        # dH/d_zeta
        grad_h = torch.autograd.grad(H.sum(), x, create_graph=True)[0]

        # Symplectic evolution: dot(zeta) = J @ grad_h [cite: 62]
        # We use matrix multiplication to enforce the FJ structure
        z_dot = grad_h @ self.J.t()
        return z_dot, H