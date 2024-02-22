import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from util import Deque
from util import load_offline_data
from tqdm import tqdm
import os

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[128, 128], final_activation=None):
        super(MLP, self).__init__()

        if final_activation == 'softmax':
            self.final_activation = nn.Softmax(dim=1)
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
        x = torch.flatten(x, start_dim=1)
        x = self.layers(x)
        if self.final_activation is not None:
            return self.final_activation(x)
        return x


class COptiDICE(Alg):
    def __init__(self, n_states, n_actions, config):
        super(COptiDICE, self).__init__()

        self.config = config
        self.n_actions = n_actions

        self.gamma = config.get('gamma', 0.95)
        self.alpha = config.get('alpha', 1)
        self.batch_size = config.get('batch_size', 1024)
        self.f_type = config.get('f_type', 'kl')
        self.num_costs = config.get('num_costs', 1) # TODO: this is hacky
        self.c_hats = torch.ones(self.num_costs, device=self.device) * config.get('cost_thresholds', 1)
        self.cost_ub_eps = config.get('cost_ub_epsilon', 0)

        self.nu_network = MLP(n_states, 1, [128, 128]).to(self.device)
        self.chi_network = MLP(n_states, self.num_costs, [128, 128]).to(self.device)
        self.policy_network = MLP(n_states, n_actions, [128, 128], 'softmax').to(self.device)

        self.lamb = torch.zeros(self.num_costs, device=self.device, requires_grad=True)
        self.tau = torch.zeros(self.num_costs, device=self.device, requires_grad=True)

        if 'offline_data' in config:
            self.memory = load_offline_data("./data/dataset/"+config['offline_data'], capacity=int(1e6))

        self.lr=config.get('lr', 0.001)
        self.optimizer = optim.Adam(
            [
                {'params': self.nu_network.parameters()},
                {'params': self.chi_network.parameters()},
            ],
            lr=self.lr,
            weight_decay=config.get('weight_decay', 0),
        )
        self.policy_optimizer = optim.Adam(
            [
                {'params': self.policy_network.parameters()},
            ],
            lr=self.lr,
            weight_decay=config.get('weight_decay', 0),
        )


    def f(self, x):
        if self.f_type == 'chisquare':
            return 0.5 * ((x - 1) ** 2)
        if self.f_type == 'softchi':
            return torch.where(
                x < 1,
                x * (torch.log(x + 1e-10) - 1) + 1,
                0.5 * ((x - 1) ** 2)
            )
        if self.f_type == 'kl':
            return x * torch.log(x + 1e-10)

        raise NotImplementedError('undefined f_type', self.f_type)

    def f_prime_inv(self, x):
        if self.f_type == 'chisquare':
            return x + 1
        if self.f_type == 'softchi':
            return torch.where(
                x < 0,
                torch.exp(torch.minimum(x, 0)),
                x + 1
            )
        if self.f_type == 'kl':
            return torch.exp(x - 1)

        raise NotImplementedError('undefined f_type', self.f_type)

    def update(self):
        s0, s, a, r, c, s_next, dones = list(zip(
            *self.memory.sample(self.batch_size)
        ))
        n = len(a)

        s = torch.tensor(np.array(s), device=self.device, dtype=torch.float)
        a = torch.tensor(a, device=self.device, dtype=torch.int64).unsqueeze(1)
        r = torch.tensor(r, device=self.device, dtype=torch.float).unsqueeze(1)

        c = [[x['near_crash'] + x['near_off_road']* 0.5] for x in c]
        c = torch.tensor(np.array(c), device=self.device, dtype=torch.float)
        s_next = torch.tensor(np.array(s_next), device=self.device, dtype=torch.float)
        dones = torch.tensor(dones, device=self.device, dtype=torch.float)
        s0 = torch.tensor(np.array(s0), device=self.device, dtype=torch.float)

        gamma = self.gamma
        alpha = self.alpha

        nu = self.nu_network(s)
        nu_next = self.nu_network(s_next)
        nu_0 = self.nu_network(s0)

        lamb = torch.clip(torch.exp(self.lamb), 0, 1e3) * 20
        chi = self.chi_network(s)

        e_nu_lamb = (
            r - torch.sum(c * lamb.detach(), axis=-1, keepdim=True)
            + self.gamma * nu_next
            - nu
        )

        e_nu_lamb = torch.clip(e_nu_lamb, max=10)

        w = torch.relu(self.f_prime_inv(e_nu_lamb / self.alpha))



        # nu loss [Eq 23]
        nu_loss = (
            (1 - gamma) * torch.mean(nu_0)
            - self.alpha * torch.mean(self.f(w))
            + torch.mean(w * e_nu_lamb)
        )

        # chi tau loss
        chi_0 = self.chi_network(s0)
        chi = self.chi_network(s)
        chi_next = self.chi_network(s_next)

        tau = torch.exp(self.tau) + 1e-6

        # [Eq 18]
        ell = (
            (1 - gamma) * chi_0
            + w.detach() * (
                c + gamma * chi_next - chi
            )
        )
        logits = ell / tau.detach()
        weights = torch.nn.functional.softmax(logits, dim=0) * n
        log_weights = torch.nn.functional.log_softmax(logits, dim=0) + np.log(n)
        kl_divergence = torch.mean(
            weights * log_weights - weights + 1, axis=0
        )
        cost_ub = torch.mean(weights * w.detach() * c, axis=0)
        chi_tau_loss = (
            torch.sum(torch.mean(weights * ell, axis=0))
            + torch.sum(-tau * (kl_divergence.detach() - self.cost_ub_eps))
        )

        # lambda loss [Eq 26]
        lamb_loss = torch.dot(
            lamb,
            self.c_hats - cost_ub.detach()
        )

        loss = nu_loss + lamb_loss + chi_tau_loss

        if torch.isnan(loss).any():
            print("nan")

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.chi_network.parameters():
            param.grad.data.clamp_(-1, 1)
        for param in self.nu_network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # policy loss

        p = self.policy_network(s).gather(1, a).flatten()
        policy_loss = -torch.mean(w.detach() * torch.log(p))

        if torch.isnan(policy_loss).any():
            print("nan")

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        for param in self.policy_network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.policy_optimizer.step()

        return loss.item()


    def save(self, path):
        torch.save(self.policy_network.state_dict(), path)

    def get_norm(self):
        total = 0
        for param in self.nu_network.parameters():
            total += param.norm().item() ** 2
        return np.sqrt(total)

    def choose_action(self, state):
        with torch.no_grad():
            state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
            p = self.policy_network(state).detach().cpu().squeeze().numpy()
            return p.argmax()
            # hack for getting round rounding error
            #i = np.argmax(p)
            #p[i] = 1 - np.sum([x for j, x in enumerate(p) if i != j])
            #return np.random.choice(5, p=p)


    def load(self, path):
        self.policy_network.load_state_dict(
            torch.load(path, map_location=self.device)
        )




