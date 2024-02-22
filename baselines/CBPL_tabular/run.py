import numpy as np
from MDP_util import *
import argparse
import yaml
import sys


def generate_data(cmdp: CMDP, mu: np.ndarray, n: int):
    ret = []
    for _ in range(n):
        i = np.random.choice(np.arange(cmdp.num_states * cmdp.num_actions), p=mu.flatten())
        state = i // cmdp.num_actions
        action = i % cmdp.num_actions
        next_state = np.random.choice(np.arange(cmdp.num_states), p=cmdp.transition[state, action, :])
        ret.append((state, action, next_state, cmdp.reward[state, action], cmdp.costs[0, state, action]))
    return ret


import argparse
parser = argparse.ArgumentParser(
    description='Tabular CBPL')
parser.add_argument('--n', type=int)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--output_file')
parser.add_argument(
    '--config',
    default='config/test.yaml',
    help='config file path')

args = parser.parse_args()

if args.output_file:
    output_file = open(args.output_file, 'w')
else:
    output_file = sys.stdout

def FQE(policy, dataset, cmdp: CMDP, reward):
    Q = np.zeros((cmdp.num_states, cmdp.num_actions))
    while True:
        Y = np.zeros((cmdp.num_states, cmdp.num_actions))
        N = np.zeros((cmdp.num_states, cmdp.num_actions))
        for state, action, next_state, r, c in dataset:
            Y[state, action] += (
                reward[state, action] +
                cmdp.gamma * Q[next_state, policy[next_state]]
            )
            N[state, action] += 1
        Q_next = Y / np.maximum(1, N)
        diff = np.max(np.abs(Q - Q_next))
        Q = Q_next
        if diff < 1e-3:
            break
    return Q[cmdp.initial_state, policy[cmdp.initial_state]]


def FQI(dataset, cmdp: CMDP, reward):
    Q = np.zeros((cmdp.num_states, cmdp.num_actions))
    while True:
        Y = np.zeros((cmdp.num_states, cmdp.num_actions))
        N = np.zeros((cmdp.num_states, cmdp.num_actions))
        for state, action, next_state, r, c in dataset:
            Y[state, action] += (
                reward[state, action] +
                cmdp.gamma * np.max(Q[next_state, :])
            )
            N[state, action] += 1
        Q_next = Y / np.maximum(1, N)
        diff = np.max(np.abs(Q - Q_next))
        Q = Q_next
        if diff < 1e-3:
            break
    return np.argmax(Q, axis=1)


def main(conf):
    constraints = np.array(conf['env']['constraints'])
    num_states = conf['env']['num_states']
    num_actions = conf['env']['num_actions']
    lambda_bound = conf['alg']['lambda_bound']
    eta = conf['alg']['online_algorithm_lr']
    initial_states = [0]
    max_iteration = conf['alg']['max_iteration']
    converge_threshold = conf['alg']['converge_threshold']

    np.random.seed(args.seed)
    cmdp = generate_toy_cmdp(num_states, num_actions, conf['env']['cmdp_gamma'], 2)
    tau = cmdp.cost_thresholds[0] / (1 - cmdp.gamma)

    # data distribution
    pi_star = solve_cmdp(cmdp)
    pi_uniform = np.ones((num_states, num_actions)) / num_actions

    pi_mu = policy_mixture(pi_star, pi_uniform, alpha=0.5)
    d_mu = compute_occupancy_measure(cmdp, pi_mu)
    d_pi_star = compute_occupancy_measure(cmdp, pi_star)
    w_pi_star = d_pi_star / d_mu
    concentrability = np.max(w_pi_star)

    np.random.seed(args.seed)
    # offline dataset
    dataset = generate_data(cmdp, d_mu, args.n)
    pi = solve_cmdp(cmdp)
    v_r, q_r, v_c, q_c = policy_evaluation(cmdp, pi_star)
    v_opt = v_r[cmdp.initial_state]
    c_opt = v_c[0][cmdp.initial_state]

    tau = cmdp.cost_thresholds[0]
    lamb = [lambda_bound / 2]
    lamb0 = [lambda_bound / 2]
    Jrs = []
    Jcs = []
    for i in range(max_iteration):
        pi = FQI(dataset, cmdp, cmdp.reward - lamb[-1] * cmdp.costs[0])
        Jrs.append(FQE(pi, dataset, cmdp, cmdp.reward))
        Jcs.append(FQE(pi, dataset, cmdp, cmdp.costs[0]))

        lamb_mean = np.mean(lamb)
        pi_tilde = FQI(dataset, cmdp, cmdp.reward - lamb_mean * cmdp.costs[0])
        Jr_tilde = FQE(pi_tilde, dataset, cmdp, cmdp.reward)
        Jc_tilde = FQE(pi_tilde, dataset, cmdp, cmdp.costs[0])

        Jr_mean = np.mean(Jrs)
        Jc_mean = np.mean(Jcs)
        L_min = min(
            Jr_mean,
            Jr_mean + lambda_bound * (tau - Jc_mean)
        )
        L_max = Jr_tilde + lamb_mean * (tau - Jc_tilde)
        diff = L_max - L_min

        z = tau - Jcs[-1]
        lamb_new = lamb[-1] * np.exp(-eta * z)
        denom = lamb_new + lamb0[-1]
        lamb.append(lamb_new / denom * lambda_bound)
        lamb0.append(lamb0[-1] / denom * lambda_bound)

        pi_flat = np.zeros((cmdp.num_states, cmdp.num_actions))
        for k in np.arange(cmdp.num_states):
            pi_flat[k, pi[k]] = 1
        v_r, _, v_c, _ = policy_evaluation(cmdp, pi_flat)
        print(i + 1, v_r[cmdp.initial_state] / v_opt, v_c[0, cmdp.initial_state] / c_opt, diff, file=output_file)
        if L_max - L_min < 0.01:
            break


if __name__ == '__main__':
    with open(args.config, 'r') as f:
        conf = yaml.safe_load(f)
    main(conf)
