import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from models.hnn_core import ChristLeeHNN
from data.trajectory_generator import generate_trajectories


def train():
    states, target_dots = generate_trajectories()
    model = ChristLeeHNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(500):
        optimizer.zero_grad()
        pred_dots, pred_H = model(states)

        # 1. Standard Dynamics Loss
        loss_dyn = torch.mean((pred_dots - target_dots) ** 2)

        # 2. Gauge Invariance Loss (The "Anjali-S" Correction)
        # Shift states along the zero mode: delta_theta = kappa, delta_lambda = -kappa
        kappa = 0.05
        #random_kappa = (torch.rand(states.shape[0], 1) - 0.5) * 0.2
        gauge_shifted_states = states.clone()
        gauge_shifted_states[:, 2] += kappa  # theta shift [cite: 150]
        gauge_shifted_states[:, 4] -= kappa  # lambda shift [cite: 142, 145]

        # 2. Physics Constraint Loss:
        # In polar coordinates, Eq. 11 of your paper shows Omega = P_theta = 0
        loss_constraint = torch.mean(states[:, 3] ** 2)  # Index 3 corresponds to P_theta

        _, shifted_H = model(gauge_shifted_states)
        loss_gauge = torch.mean((pred_H - shifted_H) ** 2)  # H must be invariant [cite: 152]

        # 3. Total Physics-Aware Loss
        total_loss = loss_dyn + (5.0 * loss_gauge) + (0.5 * loss_constraint)
        loss_interpretation = torch.mean((pred_dots[:, 4] + pred_dots[:, 2]) ** 2)

        # Update total loss
        total_loss += 1.0 * loss_interpretation
        total_loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss.item():.6f}")


if __name__ == "__main__":
    train()