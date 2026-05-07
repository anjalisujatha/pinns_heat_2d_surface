import os
import numpy as np
import torch
from scipy.integrate import solve_ivp

# --- 1. Physical & Geometric Parameters [cite: 55, 65, 68] ---
R = 3.0  # Major radius
r = 1.0  # Minor radius
a = np.sqrt(R ** 2 - r ** 2)
m = 1.0  # Mass
p, q = 2, 3  # Knot winding numbers (must be relatively prime)
eta = np.arccosh(R / r)  # Constant toroidal surface


# --- 2. Faddeev-Jackiw Equations of Motion [cite: 69, 517-525] ---
def system_dynamics(t, state):
    # state = [theta, phi, P_theta, P_phi]
    theta, phi, p_theta, p_phi = state

    # Pre-calculate common terms from the Lagrangian [cite: 69]
    denom = (np.cosh(eta) - np.cos(theta)) ** 2
    factor = denom / (m * a ** 2)

    # Constraint denominator from Symplectic Analysis [cite: 313, 517]
    c_inv_factor = (p ** 2 * np.sinh(eta) ** 2 + q ** 2)

    # Auxiliary variable 'b' from BRST formalism [cite: 520]
    # Here we assume a simplified case for data generation where b balances the constraint
    diff_p = (q * p_theta - p * p_phi)

    # Equations of Motion (Eq. 69 in your paper)
    d_theta = factor * (q * diff_p) / c_inv_factor
    d_phi = -factor * (p * diff_p) / c_inv_factor

    # Momentum derivatives [cite: 524, 689]
    # In the constrained case, these follow the symplectic structure
    d_p_theta = - ((np.cosh(eta) - np.cos(theta)) * np.sin(theta) / (m * a ** 2)) * (p_theta ** 2)
    d_p_phi = 0  # Symmetry in phi for the free particle on knot

    return [d_theta, d_phi, d_p_theta, d_p_phi]


# --- 3. Cartesian Mapping for e3nn  ---
def to_cartesian(theta, phi):
    common_denom = np.cosh(eta) - np.cos(theta)
    x = (a * np.sinh(eta) * np.cos(phi)) / common_denom
    y = (a * np.sinh(eta) * np.sin(phi)) / common_denom
    z = (a * np.sin(theta)) / common_denom
    return np.array([x, y, z])


# --- 4. Integration Loop ---
# Initial condition: theta=0, phi=0 (satisfies p*theta + q*phi = 0) [cite: 67]
initial_state = [0.0, 0.0, 1.0, - (p / q) * 1.0]  # P_phi set to satisfy initial tertiary constraint
t_span = (0, 50)
t_eval = np.linspace(t_span[0], t_span[1], 2000)

solution = solve_ivp(system_dynamics, t_span, initial_state, t_eval=t_eval, method='RK45')

# --- 5. Processing & Saving for GNN ---
positions_toroidal = solution.y[:2, :].T  # [theta, phi]
momenta_toroidal = solution.y[2:, :].T  # [P_theta, P_phi]

cartesian_pos = []
for theta, phi in positions_toroidal:
    cartesian_pos.append(to_cartesian(theta, phi))

# Final Dictionary for PyTorch
data_dict = {
    "pos": torch.tensor(np.array(cartesian_pos), dtype=torch.float32),
    "toroidal": torch.tensor(positions_toroidal, dtype=torch.float32),
    "momenta": torch.tensor(momenta_toroidal, dtype=torch.float32),
    "params": {"p": p, "q": q, "R": R, "r": r}
}

out_path = os.path.join(os.path.dirname(__file__), "torus_knot_data.pt")
torch.save(data_dict, out_path)
print(f"Data generation complete for ({p},{q})-knot. Saved to {out_path}")
