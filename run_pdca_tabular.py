from MDP_util import (
    generate_toy_cmdp,
    solve_cmdp,
    policy_evaluation,
    ope_practical,
    compute_occupancy_measure,
    critic,
    policy_mixture,
    MDP,
    CMDP,
)
import numpy as np
import sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=100)
parser.add_argument('--C', type=float, default=5)
parser.add_argument('--B', type=float, default=2)
parser.add_argument('--K', type=int, default=2000)
parser.add_argument('--eta', type=float, default=2)
parser.add_argument('--eta_lambda', type=float, default=2)
parser.add_argument('--tighten', type=float, default=0)
parser.add_argument('--k_tighten', type=float, default=0)
parser.add_argument('--seed', type=int)
parser.add_argument('--seed_cmdp', type=int, default=1)
parser.add_argument('--output_file')
args = parser.parse_args()

def pi_player(pi: np.ndarray, h: np.ndarray, eta: float):
    pi_new = np.clip(np.multiply(pi, np.exp(eta * h)), 1e-6, 1e6)
    return pi_new / np.sum(pi_new, axis=1, keepdims=True)


def generate_data(cmdp: CMDP, mu: np.ndarray, n: int):
    ret = np.zeros((cmdp.num_states, cmdp.num_actions, cmdp.num_states))
    for _ in range(n):
        i = np.random.choice(np.arange(cmdp.num_states * cmdp.num_actions), p=mu.flatten())
        state = i // cmdp.num_actions
        action = i % cmdp.num_actions
        next_state = np.random.choice(np.arange(cmdp.num_states), p=cmdp.transition[state, action, :])
        ret[state, action, next_state] += 1
    return ret


if __name__ == '__main__':
    num_states = 10
    num_next_states = 2
    num_actions = 5
    gamma = 0.8
    seed = args.seed
    K = args.K
    n = args.n
    if args.output_file:
        output_file = open(args.output_file, 'w')
    else:
        output_file = sys.stdout

    eta = args.eta
    lambda_eta = args.eta_lambda
    C = args.C
    B = args.B

    if args.seed_cmdp is not None:
        np.random.seed(args.seed_cmdp)
    cmdp = generate_toy_cmdp(num_states, num_actions, gamma, num_next_states)
    tau = cmdp.cost_thresholds[0] / (1 - cmdp.gamma)
    tau = tau - args.tighten / np.sqrt(n) - args.k_tighten * 2 / np.sqrt(K)

    # data distribution
    pi_star = solve_cmdp(cmdp)
    pi_uniform = np.ones((num_states, num_actions)) / num_actions

    pi_mu = policy_mixture(pi_star, pi_uniform, alpha=0.5)
    d_mu = compute_occupancy_measure(cmdp, pi_mu)

    if args.seed is not None:
        np.random.seed(args.seed)
    else:
        np.random.seed()
    # offline dataset
    dataset = generate_data(cmdp, d_mu, n)

    # initialize
    pi_list = [None] * K
    lambda_list = [None] * K
    lambda0_list = [None] * K
    Ar_list = [None] * K
    Ac_list = [None] * K
    f_list = [None] * K
    g_list = [None] * K

    pi_list[0] = pi_uniform

    # ground truth
    v_r, q_r, v_c, q_c = policy_evaluation(cmdp, pi_star)
    v_opt = v_r[cmdp.initial_state]
    c_opt = v_c[0][cmdp.initial_state]
    v_r, _, v_c, _ = policy_evaluation(cmdp, pi_list[0])
    print(1, v_r[cmdp.initial_state] / v_opt, v_c[0, cmdp.initial_state] / c_opt, lambda_list[0], file=output_file)

    for k in range(K - 1):
        pi = pi_list[k]

        # lambda-player
        h, *_ = ope_practical(cmdp, cmdp.costs[0, :, :], pi, dataset)
        h0 = np.sum(np.multiply(h[cmdp.initial_state, :], pi[cmdp.initial_state, :]))
        l = B if tau - h0 < 0 else 0
        lambda_list[k] = l

        # critics
        f, Ar, _ = critic(cmdp, cmdp.reward, pi, dataset, sign=1, C=args.C)
        g, Ac, _ = critic(cmdp, cmdp.costs[0, :, :], pi,dataset,sign=-1, C=args.C)
        v_r, v_c = v_r / (1 - cmdp.gamma), v_c / (1 - cmdp.gamma)
        f_list[k] = f
        g_list[k] = g
        f0 = np.sum(np.multiply(f[cmdp.initial_state, :], pi[cmdp.initial_state, :]))
        g0 = np.sum(np.multiply(g[cmdp.initial_state, :], pi[cmdp.initial_state, :]))

        # pi-player
        z = (f + l * (tau - g)) * (1 - cmdp.gamma) / (2 * B + 1)
        pi_list[k + 1] = pi_player(pi, z, eta)

        v_r, _, v_c, _ = policy_evaluation(cmdp, pi_list[k + 1])
        print(k + 2, v_r[cmdp.initial_state] / v_opt, v_c[0, cmdp.initial_state] / c_opt, lambda_list[k], file=output_file)
