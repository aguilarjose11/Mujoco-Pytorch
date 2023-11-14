import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.base import Network

from typing import Tuple

Tensor = torch.Tensor

class Actor(Network):
    def __init__(self,
                 layer_num,
                 input_dim,
                 output_dim,
                 hidden_dim,
                 activation_function = torch.tanh,
                 last_activation = None,
                 trainable_std = False):
        super(Actor, self).__init__(layer_num, input_dim, output_dim, hidden_dim, activation_function ,last_activation)
        self.trainable_std = trainable_std # Unique to Actor
        if self.trainable_std == True:
            # Add unique standard deviation learnable parameter
            self.logstd = nn.Parameter(torch.zeros(1, output_dim))
    def forward(self, x) -> Tuple[Tensor, Tensor]:
        mu = self._forward(x)
        if self.trainable_std == True:
            std = torch.exp(self.logstd)
        else:
            logstd = torch.zeros_like(mu)
            std = torch.exp(logstd) # Tensor of 1s
        return mu,std

class Critic(Network):
    def __init__(self,
                 layer_num,
                 input_dim,
                 output_dim,
                 hidden_dim,
                 activation_function,
                 last_activation = None):
        super(Critic, self).__init__(layer_num, input_dim, output_dim, hidden_dim, activation_function ,last_activation)
        
    def forward(self, *x) -> Tensor:
        x = torch.cat(x,-1)
        return self._forward(x)
    