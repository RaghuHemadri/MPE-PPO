from turtle import forward
import torch.nn as nn
import torch

class reparamatrization(nn.Module):
    def __init__(self, input_dim) -> None:
        super(reparamatrization, self).__init__()
        self.input_dim = input_dim
        self.fc_mu = nn.Linear(self.input_dim, self.input_dim)
        self.fc_var = nn.Linear(self.input_dim, self.input_dim)

    def forward(self, x):
        mu, log_var = self.fc_mu(x), self.fc_var(x)

        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        return z, mu, log_var