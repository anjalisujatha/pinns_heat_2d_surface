import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import matplotlib.pyplot as plt
from model.architecture import model

# --- 1. Load Data & Trained Weights ---
data_path = os.path.join(os.path.dirname(__file__), "..", "data", "torus_knot_data.pt")
weights_path = os.path.join(os.path.dirname(__file__), "..", "trained_torus_gnn.pth")

data = torch.load(data_path)
model.load_state_dict(torch.load(weights_path))
model.eval()

# Extract parameters [cite: 68-70]
p_knot = data["params"]["p"]
q_knot = data["params"]["q"]
R = data["params"]["R"]
pos = data["pos"]

# --- 2. Long-Term Rollout (Autoregressive) ---
# We start at t=0 and see if the model can stay on the knot for 500 steps
steps = 500
current_pos = pos[0:1] # Start with the first frame
predictions = [current_pos]

def make_data(p):
    N = p.shape[0]
    x = torch.cat([torch.ones(N, 1), p], dim=1)  # [N, 4] matching "1x0e + 1x1o"
    z = torch.ones(N, 1)
    return {"pos": p, "x": x, "z": z}

with torch.no_grad():
    for _ in range(steps):
        # Predict next position
        next_pos = model(make_data(current_pos))
        predictions.append(next_pos)
        current_pos = next_pos

rollout_tensor = torch.cat(predictions, dim=0)

# --- 3. Physical Metric: Knot Residual [cite: 67] ---
# Calculate R = |p*theta + q*phi| to see if the model "fell off" the knot
def get_toroidal_from_cartesian(p_cart):
    x, y, z = p_cart[:, 0], p_cart[:, 1], p_cart[:, 2]
    # Invert the mapping from Eq. 1 [cite: 53]
    phi = torch.atan2(y, x)
    theta = torch.atan2(z, torch.sqrt(x**2 + y**2) - R)
    return theta, phi

theta_pred, phi_pred = get_toroidal_from_cartesian(rollout_tensor)
knot_residual = torch.abs(p_knot * theta_pred + q_knot * phi_pred)

print(f"Mean Knot Residual over {steps} steps: {torch.mean(knot_residual).item():.6f}")

# --- 4. Equivariance Test (Symmetry Audit) ---
# Rotate input and check if output rotates exactly the same way
random_rot = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=torch.float32) # 90 deg rotation
rotated_input = pos[0:1] @ random_rot.T
rotated_output = model(make_data(rotated_input))
expected_output = (model(make_data(pos[0:1])) @ random_rot.T)

equiv_error = torch.mean(torch.abs(rotated_output - expected_output))
print(f"Equivariance Error: {equiv_error.item():.8f}")

# --- 5. Visualization ---
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot ground truth
ax.plot(pos[:steps, 0], pos[:steps, 1], pos[:steps, 2], label="FJ Ground Truth", alpha=0.5)
# Plot model rollout
ax.plot(rollout_tensor[:, 0], rollout_tensor[:, 1], rollout_tensor[:, 2], '--', label="GNN Rollout", color='red')

ax.set_title(f"({p_knot},{q_knot})-Torus Knot Dynamics Rollout")
ax.legend()
plt.show()
