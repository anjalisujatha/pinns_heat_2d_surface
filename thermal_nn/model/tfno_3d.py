from neuralop.models import TFNO

def get_vinci_model(res=32):
    model = TFNO(
        n_modes=(12, 12, 12),      # Frequency modes kept [cite: 67]
        hidden_channels=32,        # Latent space width [cite: 23]
        in_channels=2,             # [Conductivity, Source]
        out_channels=1,            # [Temperature]
        factorization='tucker',    # Tucker decomposition for memory [cite: 57, 81]
        rank=0.4                   # Compression rank
    )
    return model