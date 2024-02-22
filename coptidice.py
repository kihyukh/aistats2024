import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from Q_network import MLP,GaussianPolicy
from replay_buffer import Dataset


class COptiDICE:
    def __init__(self, dim_states, dim_actions, config):
        super(COptiDICE, self).__init__()

        self.config = config
        self.dim_states = dim_states
        self.dim_actions = dim_actions

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device('cpu')

        self.gamma = config.get('gamma', 0.95)
        self.alpha = config.get('alpha', 1)
        self.batch_size = config.get('batch_size', 1024)
        self.f_type = config.get('f_type', 'kl')
        self.num_costs = config.get('num_costs', 1)
        self.c_hats = torch.ones(self.num_costs, device=self.device) * config.get('cost_thresholds', 1)
        self.cost_ub_eps = config.get('cost_ub_epsilon', 0)

        self.nu_network = MLP(dim_states, 1).to(self.device)
        self.chi_network = MLP(dim_states, self.num_costs).to(self.device)
        self.policy_network = GaussianPolicy(dim_states, dim_actions).to(self.device)

        self.lamb = torch.zeros(self.num_costs, device=self.device, requires_grad=True)
        self.tau = torch.zeros(self.num_costs, device=self.device, requires_grad=True)

        self.memory = Dataset()
        self.memory.load_from_files(config['dataset'])

        self.lr = config.get('lr', 0.001)
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
        batch = np.array(self.memory.sample(self.batch_size),dtype=object)
        states, actions, rewards, costs, next_states, dones, initial_states = batch[:,0], batch[:,1], batch[:,2],batch[:,3],\
                                                              batch[:,4],batch[:,5],batch[:,6]

        s = torch.from_numpy(np.stack(states)).float().to(self.device)
        a = torch.from_numpy(np.stack(actions)).float().to(self.device)
        r = torch.from_numpy(np.stack(rewards)).float().unsqueeze(1).to(self.device)
        c = torch.from_numpy(np.stack(costs)).float().unsqueeze(1).to(self.device)
        s_next = torch.from_numpy(np.stack(next_states)).float().to(self.device)
        dones = torch.from_numpy(np.stack(dones)).unsqueeze(1).to(self.device)
        s0 = torch.from_numpy(np.stack(initial_states)).float().to(self.device)

        n = len(a)

        gamma = self.gamma
        alpha = self.alpha

        nu = self.nu_network(s)
        nu_next = self.nu_network(s_next)
        nu_0 = self.nu_network(s0)

        lamb = torch.clip(torch.exp(self.lamb), 0, 1e3)

        e_nu_lamb = (
                r - torch.sum(c * lamb.detach(), axis=-1, keepdim=True)
                + self.gamma * nu_next
                - nu
        )


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

        nn.utils.clip_grad_norm_(self.chi_network.parameters(),max_norm=1.0)
        nn.utils.clip_grad_norm_(self.nu_network.parameters(),max_norm=1.0)

        self.optimizer.step()

        # policy loss [Eq 22]
        dist = self.policy_network(s)
        action = dist.sample()

        p = dist.log_prob(a)
        policy_loss = -torch.mean(w.detach() * p)

        if torch.isnan(policy_loss).any():
            print("nan")

        self.policy_optimizer.zero_grad()
        policy_loss.backward()

        nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
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
            dist = self.policy_network(state)
            action = dist.sample().detach().cpu().squeeze().numpy()
            return action


    def load(self, path):
        self.policy_network.load_state_dict(
            torch.load(path, map_location=self.device)
        )

