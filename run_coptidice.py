import yaml
import argparse
from coptidice import COptiDICE
import torch
from rwrl import make_rwrl_env, CustomEnv


def eval(env, alg, test_epoch):
    with torch.no_grad():
        epoch_reward = 0
        epoch_cost = 0
        for e in range(test_epoch):
            state = env.reset()
            done = False
            t = 0

            while not done:
                action = alg.choose_action(torch.tensor(state[:-env.num_constraints]))
                next_state, reward, done, info, cost = env.log_step(action)

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

    max_epoch = conf['alg']['max_epoch']
    alg = COptiDICE(dim_states, dim_actions, conf['alg'])

    for n in range(max_epoch):
        alg.update()
        if (n + 1) % 1000 == 0:
            res = eval(env, alg, 3)
            print("Reward: ", res[0], "Cost: ", res[1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='coptidice')
    parser.add_argument(
        '--config',
        default='config/coptidice/test.yaml',
        help='config file path')

    args = parser.parse_args()
    with open(args.config, 'r') as f:
        conf = yaml.safe_load(f)
    run(conf)
