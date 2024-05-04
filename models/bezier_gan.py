import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import optim
import numpy as np

class GAN(nn.Module):
    def __init__(self, latent_dim=4, noise_dim=10, n_points=192, bezier_degree=31, bounds=(0.0, 1.0)):
        super(GAN, self).__init__()

        self.latent_dim = latent_dim
        self.noise_dim = noise_dim
        self.X_shape = (n_points, 2, 1)
        self.bezier_degree = bezier_degree
        self.bounds = bounds

        self.depth_cpw = 32*8
        self.dim_cpw = int((bezier_degree + 1) / 8)
        self.kernel_size = (4, 3)

        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

    def build_generator(self):
        generator = nn.Sequential(
            nn.Linear(self.latent_dim + self.noise_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(1024, self.dim_cpw * 3 * self.depth_cpw),
            nn.BatchNorm1d(self.dim_cpw * 3 * self.depth_cpw),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Unflatten(1, (self.dim_cpw, 3, self.depth_cpw)),

            nn.ConvTranspose2d(self.depth_cpw, int(self.depth_cpw / 2), self.kernel_size, stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(int(self.depth_cpw / 2)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(int(self.depth_cpw / 2), int(self.depth_cpw / 4), self.kernel_size, stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(int(self.depth_cpw / 4)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(int(self.depth_cpw / 4), int(self.depth_cpw / 8), self.kernel_size, stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(int(self.depth_cpw / 8)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(int(self.depth_cpw / 8), 1, (1, 2), padding=(0, 0))
        )
        return generator

    def build_discriminator(self):
        discriminator = nn.Sequential(
            nn.Conv2d(1, self.depth_cpw, (4, 2), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(self.depth_cpw),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.depth_cpw, self.depth_cpw * 2, (4, 2), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(self.depth_cpw * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.depth_cpw * 2, self.depth_cpw * 4, (4, 2), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(self.depth_cpw * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.depth_cpw * 4, self.depth_cpw * 8, (4, 2), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(self.depth_cpw * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.depth_cpw * 8, self.depth_cpw * 16, (4, 2), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(self.depth_cpw * 16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.depth_cpw * 16, self.depth_cpw * 32, (4, 2), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(self.depth_cpw * 32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(self.depth_cpw * 32 * (self.X_shape[0] // 64), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        return discriminator

    def forward(self, c, z):
        cz = torch.cat((c, z), dim=1)
        cpw = self.generator(cz)
        return cpw
    
    def train(self, X_train, train_steps=2000, batch_size=256, save_interval=0, directory='.'):
        X_train = torch.tensor(X_train, dtype=torch.float32)

        # Optimizers
        d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
        g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0001, betas=(0.5, 0.999))

        # Loss function
        criterion = nn.BCELoss()

        for t in range(train_steps):
            # Train discriminator
            d_optimizer.zero_grad()
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)

            # Real data
            real_outputs = self.discriminator(X_train[np.random.choice(len(X_train), batch_size, replace=False)])
            d_loss_real = criterion(real_outputs, real_labels)

            # Fake data
            latent = torch.randn(batch_size, self.latent_dim)
            noise = torch.randn(batch_size, self.noise_dim)
            fake_outputs = self.forward(latent, noise).detach()
            fake_outputs = fake_outputs.view(-1, *self.X_shape)
            fake_outputs = fake_outputs.to(X_train.device)
            fake_outputs = fake_outputs.clone().detach()
            fake_outputs.requires_grad_(True)
            fake_outputs = self.discriminator(fake_outputs)
            d_loss_fake = criterion(fake_outputs, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

            # Train generator
            g_optimizer.zero_grad()
            latent = torch.randn(batch_size, self.latent_dim)
            noise = torch.randn(batch_size, self.noise_dim)
            fake_outputs = self.forward(latent, noise)
            fake_outputs = fake_outputs.view(-1, *self.X_shape)
            fake_outputs = fake_outputs.to(X_train.device)
            fake_outputs = fake_outputs.clone().detach()
            fake_outputs.requires_grad_(True)
            fake_outputs = self.discriminator(fake_outputs)
            g_loss = criterion(fake_outputs, real_labels)
            g_loss.backward()
            g_optimizer.step()

            # Show messages
            log_mesg = "%d: [D] real %f fake %f" % (t+1, d_loss_real.item(), d_loss_fake.item())
            log_mesg = "%s  [G] fake %f" % (log_mesg, g_loss.item())
            print(log_mesg)

            if save_interval > 0 and (t + 1) % save_interval == 0:
                torch.save(self.state_dict(), f"{directory}/model_{t+1}.pt")
                print(f"Model saved in path: {directory}/model_{t+1}.pt")

    def restore(self, directory='.'):
        self.load_state_dict(torch.load(f"{directory}/model.pt"))

    def synthesize(self, latent, noise=None, return_cp=False):
        if noise is None:
            noise = torch.randn(latent.size(0), self.noise_dim)
        with torch.no_grad():
            cpw = self.forward(latent, noise)
            if return_cp:
                return cpw.cpu().numpy()
            else:
                return cpw.cpu().numpy()

    def embed(self, X):
        with torch.no_grad():
            outputs = self.discriminator(X)
            return outputs.cpu().numpy()

if __name__ == '__main__':
    gan = GAN()
    print(gan)
