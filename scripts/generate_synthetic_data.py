import torch
import pandas as pd
from models.generator import Generator

# Load the trained generator model
latent_dim = 100
output_dim = 22
generator = Generator(latent_dim, output_dim)
generator.load_state_dict(torch.load('models/generator.pth'))
generator.eval()

# Generate synthetic data
num_samples = 1000
z = torch.randn(num_samples, latent_dim)
synthetic_data = generator(z).detach().numpy()

# Save the synthetic data
pd.DataFrame(synthetic_data).to_csv('data/synthetic_data.csv', index=False)
