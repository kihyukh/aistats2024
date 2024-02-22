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
    compute_mle_cmdp,
)
import MDP_util as util
import coptidice_tabular as offline_cmdp
import numpy as np
import sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=100)
parser.add_argument('--cmdp_seed', type=int)
parser.add_argument('--seed', type=int)
parser.add_argument('--output_file')
args = parser.parse_args()

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
    n = args.n
    if args.output_file:
        output_file = open(args.output_file, 'w')
    else:
        output_file = sys.stdout

    if args.cmdp_seed is not None:
        np.random.seed(args.cmdp_seed)
    cmdp = generate_toy_cmdp(num_states, num_actions, gamma, num_next_states)
    tau = cmdp.cost_thresholds[0] / (1 - cmdp.gamma)

    # data distribution
    pi_star = solve_cmdp(cmdp)
    pi_uniform = np.ones((num_states, num_actions)) / num_actions

    pi_mu = policy_mixture(pi_star, pi_uniform, alpha=0.5)
    d_mu = compute_occupancy_measure(cmdp, pi_mu)
    d_pi_star = compute_occupancy_measure(cmdp, pi_star)
    w_pi_star = d_pi_star / d_mu
    concentrability = np.max(w_pi_star)

    if args.seed is not None:
        np.random.seed(args.seed)
    else:
        np.random.seed()
    # offline dataset
    dataset = generate_data(cmdp, d_mu, n)
    pi_mu_hat = np.ones((cmdp.num_states, cmdp.num_actions)) / cmdp.num_actions
    dataset_sa = np.sum(dataset, axis=2)
    dataset_s = np.sum(dataset, axis=(1, 2))
    for s in range(cmdp.num_states):
        if dataset_s[s] == 0:
            continue
        for a in range(cmdp.num_actions):
            pi_mu_hat[s, a] = dataset_sa[s, a] / dataset_s[s]

    # initialize
    v_r, q_r, v_c, q_c = policy_evaluation(cmdp, pi_star)
    v_opt = v_r[cmdp.initial_state]

    # MLE CMDP
    mle_cmdp, _ = compute_mle_cmdp(
        cmdp.num_states,
        cmdp.num_actions,
        1,
        cmdp.reward,
        cmdp.costs,
        cmdp.cost_thresholds,
        cmdp.gamma,
        dataset)

    alpha = 0.1
    # Vanilla ConstrainedOptiDICE
    pi = offline_cmdp.constrained_optidice(mle_cmdp, pi_mu_hat, alpha)
    v_r, _, v_c, _ = util.policy_evaluation(cmdp, pi)
    cdice_r = v_r[0] / v_opt
    cdice_c = v_c[0][0] / cmdp.cost_thresholds[0]
    print(cdice_r, cdice_c, file=output_file)
