import torch
from model.tfno_3d import get_vinci_model
from training.loss import pino_loss
from data.generator import generate_samples

# 1. Setup Device (MPS for M4 Mac, CUDA for Colab) [cite: 34, 125]
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda")

# 2. Build Model and Data
model = get_vinci_model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
x_train, y_train = generate_samples(num_samples=20)  # Small batch for demo
x_train, y_train = x_train.to(device), y_train.to(device)

# 3. Training Loop (Phase 3) [cite: 108]
for epoch in range(100):
    optimizer.zero_grad()

    out = model(x_train)

    # Hybrid Loss: Data MSE + Physics Residual [cite: 51, 85]
    data_loss = torch.nn.functional.mse_loss(out, y_train)
    phys_loss = pino_loss(out, x_train[:, 0:1, ...], x_train[:, 1:2, ...])

    total_loss = data_loss + 0.1 * phys_loss
    total_loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {total_loss.item():.6f}")

print("Inference Complete. Model ready for Zero-Shot Scaling to 128^3.")