from turtle import forward
import torch.nn as nn
import torch

from on_policy.utils.utils import store_args

class ReparamLayer(nn.Module):
    @store_args
    def __init__(self,
                input_dim:int, 
                mu_coef:float, 
                var_coef:float, 
                use_activation:bool=True,
                use_init_weights:bool=True,
                init_w:float=3e-3) -> None:
        super(ReparamLayer, self).__init__()
        self.fc_mu = nn.Linear(self.input_dim, self.input_dim)
        self.fc_var = nn.Linear(self.input_dim, self.input_dim)
        if self.use_init_weights:
            self.fc_mu.weight.data.uniform_(-init_w, init_w)
            self.fc_mu.bias.data.uniform_(-init_w, init_w)
            self.fc_var.weight.data.uniform_(-init_w, init_w)
            self.fc_var.bias.data.uniform_(-init_w, init_w)

    def forward(self, x:torch.FloatTensor) -> tuple(torch.FloatTensor, 
                                                    torch.FloatTensor, 
                                                    torch.FloatTensor):
        if self.use_activation:
            mu      = torch.tanh(self.fc_mu(x)) * self.mu_coef
            log_var = torch.tanh(self.fc_var(x)) * self.var_coef
        else:
            mu      = self.fc_mu(x)
            log_var = self.fc_var(x)

        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        return z, mu, log_var