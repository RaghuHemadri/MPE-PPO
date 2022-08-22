import torch.nn as nn
import torch

from utils.utils import store_args
from typing import Tuple

class ReparamLayer(nn.Module):
    @store_args
    def __init__(self,
                input_dim:int, 
                mu_coef:float, 
                var_coef:float, 
                use_activation:bool=True,
                use_init_weights:bool=False,
                init_w:float=3e-3) -> None:
        super(ReparamLayer, self).__init__()
        self.fc_mu = nn.Linear(self.input_dim, self.input_dim)
        self.fc_var = nn.Linear(self.input_dim, self.input_dim)
        self.softplus = nn.Softplus()
        if self.use_init_weights:
            self.fc_mu.weight.data.uniform_(-init_w, init_w)
            self.fc_mu.bias.data.uniform_(-init_w, init_w)
            self.fc_var.weight.data.uniform_(-init_w, init_w)
            self.fc_var.bias.data.uniform_(-init_w, init_w)

    def forward(self, x:torch.FloatTensor) -> Tuple[torch.FloatTensor, 
                                                    torch.FloatTensor, 
                                                    torch.FloatTensor]:
        if self.use_activation:
            # mu      = torch.tanh(self.fc_mu(x)) * self.mu_coef
            # log_var = torch.tanh(self.fc_var(x)) * self.var_coef
            mu      = self.fc_mu(x)
            log_var = self.softplus(self.fc_var(x))
        else:
            mu      = self.fc_mu(x)
            log_var = self.fc_var(x)

        std = torch.exp(torch.add(log_var, 1e-16))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        # print("mu: ", mu, "\nlog_var: ", log_var)

        return z, mu, log_var