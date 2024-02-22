import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn
from replay_buffer import Dataset
from Q_network import MLP, Q_network
import argparse
from rwrl import make_rwrl_env, CustomEnv
import yaml


class PDCA:
    def __init__(self, dim_states, dim_actions, constraints, dataset, conf):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device('cpu')
        self.dim_states = dim_states
        self.dim_actions = dim_actions
        self.constraints = torch.tensor(constraints).to(self.device)
        self.num_constraints = len(constraints)
        self.dataset = dataset
        self.batch_size = conf['batch_size']
        self.reward_network = Q_network(input_dim=(dim_states + dim_actions), output_dim=1).to(self.device)
        self.cost_network = Q_network(input_dim=(dim_states + dim_actions), output_dim=self.num_constraints).to(
            self.device)
        self.policy = MLP(input_dim=dim_states, output_dim=dim_actions, final_activation="tanh").to(self.device)
        self.fast_lr = conf['fast_lr']
        self.slow_lr = conf['slow_lr']
        self.lambdas_lr = conf['lambdas_lr']
        self.critic_optimizer = optim.Adam(
            [
                {'params': self.reward_network.parameters()},
                {'params': self.cost_network.parameters()},
            ],
            lr=self.fast_lr,
            weight_decay=0.
        )
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.slow_lr, weight_decay=0.)
        self.gamma = conf['gamma']

        self.B = conf['B']
        if self.num_constraints > 1:
            self.dummy_lambda = None
            self.lambdas = torch.tensor([self.B / self.num_constraints] * self.num_constraints).to(self.device)
        else:
            self.dummy_lambda = torch.tensor([self.B / 2.]).to(self.device)
            self.lambdas = torch.tensor([self.B / 2.]).to(self.device)

    def func_E(self, f, R, policy, states, actions, next_states, dones):
        X = f(states, actions) - R - self.gamma * f(next_states, policy(next_states)) * dones
        sum_positive = torch.mean(torch.clamp(X, min=0))
        sum_negative = torch.mean(torch.clamp(X, max=0))
        return torch.max(sum_positive, -sum_negative)

    def func_A(self, f, policy, states, actions):
        return torch.mean(f(states, policy(states)) - f(states, actions))

    def combined_reward(self, states, actions):
        return self.reward_network(states, actions) + (
                self.constraints - self.cost_network(states, actions)) * self.lambdas

    def online_update(self, initial_states):
        with torch.no_grad():
            w = torch.mean(self.constraints - self.cost_network(initial_states, self.policy(initial_states)), axis=0)
            exp_w = torch.exp(-self.lambdas_lr * w)
            lambdas = self.lambdas * exp_w

            if self.dummy_lambda:
                self.lambdas = self.B * lambdas / (lambdas + self.dummy_lambda)
                self.dummy_lambda = self.B * self.dummy_lambda / (lambdas + self.dummy_lambda)
            else:
                self.lambdas = self.B * lambdas / torch.sum(lambdas)

    def update(self):
        batch = np.array(self.dataset.sample(self.batch_size), dtype=object)
        states, actions, rewards, costs, next_states, dones, initial_states = batch[:, 0], batch[:, 1], batch[:,
                                                                                                        2], batch[:, 3], \
                                                                              batch[:, 4], batch[:, 5], batch[:, 6]

        states = torch.from_numpy(np.stack(states)).float().to(self.device)
        actions = torch.from_numpy(np.stack(actions)).float().to(self.device)
        rewards = torch.from_numpy(np.stack(rewards)).float().unsqueeze(1).to(self.device)
        costs = torch.from_numpy(np.stack(costs)).float().unsqueeze(1).to(self.device)
        next_states = torch.from_numpy(np.stack(next_states)).float().to(self.device)
        dones = torch.from_numpy(np.stack(dones)).unsqueeze(1).to(self.device)
        initial_states = torch.from_numpy(np.stack(initial_states)).float().to(self.device)

        self.critic_optimizer.zero_grad()

        loss_reward = 2 * self.func_E(self.reward_network, rewards, self.policy, states, actions, next_states, dones) + \
                      self.func_A(self.reward_network, self.policy, states, actions)
        loss_costs = 2 * self.func_E(self.cost_network, costs, self.policy, states, actions, next_states, dones) - \
                     self.func_A(self.cost_network, self.policy, states, actions)
        loss_critic = loss_reward + loss_costs
        loss_critic.backward()
        nn.utils.clip_grad_norm_(self.reward_network.parameters(), max_norm=1.0)
        nn.utils.clip_grad_norm_(self.cost_network.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        self.policy_optimizer.zero_grad()
        loss_policy = -self.func_A(self.combined_reward, self.policy, states, actions)

        loss_policy.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.policy_optimizer.step()

        self.online_update(initial_states)


def eval(env, alg, test_epoch):
    with torch.no_grad():
        epoch_reward = 0
        epoch_cost = 0
        for e in range(test_epoch):
            state = env.reset()
            done = False
            t = 0

            while not done:
                action = alg.policy(
                    torch.tensor(state[:-env.num_constraints], device=alg.device, dtype=torch.float32).unsqueeze(0))
                next_state, reward, done, info, cost = env.log_step(action.detach().cpu().squeeze().numpy())
                epoch_reward += reward
                epoch_cost += cost
                t += 1
                state = next_state

    return epoch_reward / test_epoch, epoch_cost / test_epoch


def run(conf):
    env_name = conf['env']['env_name']
    task_name = conf['env']['task_name']
    safety_coeff = conf['env']['safety_coeff']
    env = make_rwrl_env(env_name, task_name, safety_coeff)
    env = CustomEnv(env, env_name, lamb=0)
    dim_states = env.observation_space.shape[0] - env.num_constraints
    dim_actions = env.action_space.shape[0]

    constraints = [conf['env']['constraints']]
    max_epoch = conf['alg']['max_epoch']
    dataset = Dataset(capacity=3000000)

    dataset.load_from_files(conf['alg']['dataset'])
    alg = PDCA(dim_states, dim_actions, constraints, dataset=dataset, conf=conf['alg'])
    for n in range(max_epoch):
        alg.update()

        if (n + 1) % 1000 == 0:
            res = eval(env, alg, 3)
            print("Reward: ", res[0], "Cost: ", res[1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PDCA')
    parser.add_argument(
        '--config',
        default='config/PDCA/cartpole.yaml',
        help='config file path')

    args = parser.parse_args()
    with open(args.config, 'r') as f:
        conf = yaml.safe_load(f)
    run(conf)
