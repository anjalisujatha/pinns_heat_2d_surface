import torch
import numpy as np


def generate_samples(num_samples=10, res=32):
    inputs = []
    targets = []
    for _ in range(num_samples):
        # Create random conductivity field (Silicon=150, Air=0.026)
        kappa = np.full((res, res, res), 0.026)
        # Stamp a "chip"
        x, y, z = np.random.randint(5, res - 10, 3)
        kappa[x:x + 8, y:y + 8, z:z + 2] = 150.0

        # Random heat source
        Q = np.zeros((res, res, res))
        Q[x + 2:x + 6, y + 2:y + 6, z + 1] = 100.0

        # Solve (Note: In Phase 1, you'd call solver.py)
        # For now, we simulate a target for structure
        T = np.random.rand(res, res, res)

        # Combine input channels: [kappa, Q]
        sample_input = np.stack([kappa, Q], axis=0)
        inputs.append(sample_input)
        targets.append(T[np.newaxis, ...])

    return torch.tensor(np.array(inputs), dtype=torch.float32), \
        torch.tensor(np.array(targets), dtype=torch.float32)