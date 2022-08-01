from turtle import forward
import torch.nn as nn
import torch

class reparamatrization(nn.Module):
    def __init__(self, input_dim, mu_coef, var_coef) -> None:
        super(reparamatrization, self).__init__()
        init_w = 3e-3
        self.input_dim = input_dim
        self.mu_coef = mu_coef
        self.var_coef = var_coef
        self.fc_mu = nn.Linear(self.input_dim, self.input_dim)
        self.fc_var = nn.Linear(self.input_dim, self.input_dim)
        # self.fc_mu.weight.data.uniform_(-init_w, init_w)
        # self.fc_mu.bias.data.uniform_(-init_w, init_w)
        # self.fc_var.weight.data.uniform_(-init_w, init_w)
        # self.fc_var.bias.data.uniform_(-init_w, init_w)

    def forward(self, x):
        mu, log_var = torch.mul(self.mu_coef, torch.tanh(self.fc_mu(x))), torch.mul(self.var_coef, torch.tanh(self.fc_var(x)))

        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        return z, mu, log_var