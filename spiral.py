import torch
import torch.nn as nn
import generate_data
import random
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class Dataset:
    def __init__(self, x, y):
        self.x = torch.tensor(x).float()
        self.y = torch.tensor(y).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x = self.x[idx]
        if self.y is not None:
            y = self.y[idx]
            sample = {'x': x, 'y': y}
        else:
            sample = {'x': x}
        return sample


class Discriminator(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super(Discriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.3),
            nn.Linear(hidden_dim, out_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x.to(device))


class Generator(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super(Generator, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(Z_DIM, in_dim),
            nn.Linear(in_dim, 128),
            nn.LeakyReLU(0.3),
            nn.Linear(128, 32),
            nn.LeakyReLU(0.3),
            nn.Linear(32, out_dim),
        )

    def forward(self, x):
        return self.layers(x)


def get_dataset(num_samples=1000, data_type='line'):
    x = torch.tensor(generate_data.get_data(num_samples, data_type)).float()
    y = torch.tensor([[REAL, REAL] for _ in range(num_samples)]).float()
    return Dataset(x, y)


def train(epochs: int = 10):
    G_losses = []
    D_losses = []

    for epoch in range(epochs):
        for i, data in enumerate(dataloader, 0):
            x, y = data

            # Discriminator training: real data
            D.zero_grad()
            label = torch.full((BATCH_SIZE,), REAL, dtype=torch.float, device=device)
            output = D(data[x]).view(-1)
            D_real_loss = criterion(output, label)
            D_real_loss.backward()
            D_x = output.mean().item()

            # Discriminator training: fake data
            noise = torch.randn(BATCH_SIZE, Z_DIM, device=device)
            fake = G(noise)
            label.fill_(FAKE)
            output = D(fake.detach()).view(-1)

            D_fake_loss = criterion(output, label)
            D_fake_loss.backward()
            D_G_z1 = output.mean().item()
            D_loss = torch.mean(D_real_loss + D_fake_loss)
            D_optimizer.step()

            if epoch % 2 == 0:
                # Generator training
                G.zero_grad()
                label.fill_(REAL)  # fake labels are real for the generator!
                output = D(fake).view(-1)
                G_loss = criterion(output, label)

                G_loss.backward()
                D_G_z2 = output.mean().item()
                G_optimizer.step()
            if i == len(dataloader) - 1:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch + 1, epochs, i, len(dataloader), D_loss.item(), G_loss.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(G_loss.item())
            D_losses.append(D_loss.item())

            if i == len(dataloader) - 1:
                with torch.no_grad():
                    generator_results = []
                    for _ in range(1000):
                        noise = torch.randn(1, Z_DIM, device=device)
                        # noise = torch.tensor(np.random.uniform(low=-1, high=1, size=(1, Z_DIM))).float().to(device)
                        fake = G(noise).detach().cpu()
                        generator_results.append(fake.tolist()[0])

                    points = generate_data.get_data(1000, 'spiral')
                    plt.plot(points[:, 0], points[:, 1], '.', label='original')
                    plt.plot([x[0] for x in generator_results], [x[1] for x in generator_results], '.',
                             label='generator')

                    plt.title('training set')
                    plt.legend()
                    plt.show()
                    generator_results = []


if __name__ == '__main__':
    seed = 42
    print("Seed: ", seed)
    random.seed(seed)
    torch.manual_seed(seed)
    BATCH_SIZE = 32
    Z_DIM = 1
    G_IN, G_HID, G_OUT = 2, 64, 2
    D_IN, D_HID, D_OUT = 2, 32, 1

    BETA_1 = 0.5
    REAL, FAKE = 1, 0
    NUM_SAMPLES = 10000
    D_TYPE = 'spiral'

    # Decide which device we want to run on
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load data
    dataset = get_dataset(NUM_SAMPLES, D_TYPE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # Models
    G = Generator(G_IN, G_HID, G_OUT).to(device)
    D = Discriminator(D_IN, D_HID, D_OUT).to(device)

    # Optimizers
    G_optimizer = torch.optim.Adam(G.parameters(), lr=0.001, betas=(BETA_1, 0.999))
    D_optimizer = torch.optim.Adam(D.parameters(), lr=0.003, betas=(BETA_1, 0.999))

    criterion = nn.BCELoss()
    train(epochs=250)
