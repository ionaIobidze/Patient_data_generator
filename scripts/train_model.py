import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from models import *

# Hyperparameters
latent_dim = 100
output_dim = 22  # Number of features in the preprocessed dataset
learning_rate = 0.0002
batch_size = 64
epochs = 10000

# Load preprocessed data
train_data = pd.read_csv('../data/train_preprocessed.csv')
train_data = torch.tensor(train_data.values).float()
train_loader = DataLoader(TensorDataset(train_data), batch_size=batch_size, shuffle=True)

# Initialize the models
generator = Generator(latent_dim, output_dim)
discriminator = Discriminator(output_dim)

# Optimizers
optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate)

# Loss function
criterion = nn.BCELoss()

# Training loop
for epoch in range(epochs):
    for real_data, in train_loader:
        batch_size = real_data.size(0)

        # Train Discriminator
        optimizer_d.zero_grad()
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        outputs = discriminator(real_data)
        d_loss_real = criterion(outputs, real_labels)

        z = torch.randn(batch_size, latent_dim)
        fake_data = generator(z)
        outputs = discriminator(fake_data.detach())
        d_loss_fake = criterion(outputs, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_d.step()

        # Train Generator
        optimizer_g.zero_grad()
        outputs = discriminator(fake_data)
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()
        optimizer_g.step()

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")
