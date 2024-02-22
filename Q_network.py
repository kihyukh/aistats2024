import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[256,256], final_activation=None):
        super(MLP, self).__init__()

        if final_activation == 'softmax':
            self.final_activation = nn.Softmax(dim=1)
        elif final_activation == 'tanh':
            self.final_activation = nn.Tanh()
        else:
            self.final_activation = None
        dimensions = [input_dim] + hidden_dims + [output_dim]
        layers = []
        for i in range(len(dimensions) - 1):
            layers.append(
                nn.Linear(dimensions[i], dimensions[i + 1])
            )
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*(layers[:-1]))

    def forward(self, x):
        x = self.layers(x)
        if self.final_activation is not None:
            return self.final_activation(x)
        return x


class Q_network(MLP):
    def __init__(self, input_dim, output_dim, hidden_dims=[256,256], final_activation=None):
        super(Q_network, self).__init__(input_dim, output_dim, hidden_dims, final_activation)

    def forward(self, states, actions):
        x = torch.cat((states, actions), dim=1)
        return super().forward(x)


class GaussianPolicy(MLP):
    def __init__(self, state_dim, action_dim, log_std_min=-10., log_std_max=2.0, hidden_dims=[256,256]):
        super(GaussianPolicy, self).__init__(state_dim, action_dim * 2, hidden_dims, final_activation='tanh')
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, x):
        x = super().forward(x)
        mu, log_std = x.chunk(2, dim=1)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)
        return dist
