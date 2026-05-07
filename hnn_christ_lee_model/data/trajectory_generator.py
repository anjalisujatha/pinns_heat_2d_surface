import numpy as np
import torch


def generate_trajectories(n_points=2000):
    t = np.linspace(0, 20, n_points)
    # Simple harmonic potential V(r) = 0.5 * r^2
    r = 2.0 + 0.5 * np.cos(t)
    pr = -0.5 * np.sin(t)  # Since Pr = dot(r) [cite: 48]

    # Gauge fixed variables [cite: 117]
    theta = np.zeros_like(t)
    ptheta = np.zeros_like(t)

    # Lagrange multiplier interpretations from Eq. 33-34 [cite: 162]
    # dot(lambda) = -dot(theta), dot(rho) = Ptheta
    lam = np.zeros_like(t)
    rho = np.zeros_like(t)

    # State: [r, Pr, theta, Ptheta, lambda, rho]
    states = np.stack([r, pr, theta, ptheta, lam, rho], axis=1)

    # Compute numerical derivatives (dots) for supervision
    dots = np.gradient(states, t, axis=0)

    return torch.tensor(states, dtype=torch.float32), torch.tensor(dots, dtype=torch.float32)