import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
import numpy as np
from model.architecture import model

# --- 1. Load Phase 1 Data ---
data_path = os.path.join(os.path.dirname(__file__), "..", "data", "torus_knot_data.pt")
data = torch.load(data_path)
positions = data["pos"]
p_knot = data["params"]["p"]
q_knot = data["params"]["q"]
R_maj = data["params"]["R"]
r_min = data["params"]["r"]
a_param = np.sqrt(R_maj ** 2 - r_min ** 2)
eta_c = np.arccosh(R_maj / r_min)

# Calculate Ground Truth energy for Hamiltonian Regularization [cite: 75, 509]
target_vels = positions[1:] - positions[:-1]
initial_energy = torch.mean(torch.sum(target_vels ** 2, dim=1))


# --- 2. Enhanced Physics-Informed Loss Function ---
def physics_loss(pred_pos, target_pos, current_pos, initial_energy):
    # a. Reconstruction Loss
    mse_loss = F.mse_loss(pred_pos, target_pos)

    # b. Stiff Topological Constraint
    # Forces the model to stay on the 1D knot string
    theta_pred = torch.atan2(pred_pos[:, 2], torch.sqrt(pred_pos[:, 0] ** 2 + pred_pos[:, 1] ** 2) - R_maj)
    phi_pred = torch.atan2(pred_pos[:, 1], pred_pos[:, 0])

    topo_residual = p_knot * theta_pred + q_knot * phi_pred
    # High weight (100.0) to reduce the 6.27 Mean Knot Residual
    topo_loss = 100.0 * torch.mean(topo_residual ** 2)

    # c. Hamiltonian Regularization [cite: 75, 415]
    # Penalize kinetic energy drift to respect the symplectic potential
    pred_vel = pred_pos - current_pos
    energy_pred = torch.mean(torch.sum(pred_vel ** 2, dim=1))
    energy_loss = F.mse_loss(energy_pred, initial_energy)

    return mse_loss + topo_loss + 10.0 * energy_loss


# --- 3. Training Loop ---
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Lower LR for stability

model.train()
# Increased epochs to 1000 to resolve high-curvature details
for epoch in range(1001):
    optimizer.zero_grad()

    cur_pos = positions[:-1]
    N = cur_pos.shape[0]

    input_data = {
        "pos": cur_pos,
        "x": torch.cat([torch.ones(N, 1), cur_pos], dim=1),
        "z": torch.ones(N, 1)
    }

    out = model(input_data)

    loss = physics_loss(out, positions[1:], cur_pos, initial_energy)

    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.8f}")

weights_path = os.path.join(os.path.dirname(__file__), "..", "trained_torus_gnn.pth")
torch.save(model.state_dict(), weights_path)
print(f"Saved model weights to {weights_path}")
