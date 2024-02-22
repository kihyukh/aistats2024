import torch
import os
from stable_baselines3 import SAC
from stable_baselines3.common.noise import NormalActionNoise
import pickle
import numpy as np
from tqdm import tqdm
import argparse
import yaml
from rwrl import make_rwrl_env, CustomEnv

parser = argparse.ArgumentParser(
    description='generating offline dataset with SAC')
parser.add_argument(
    '--config',
    default='config/online_alg/rwrl.yaml',
    help='config file path')

args = parser.parse_args()
with open(args.config, 'r') as f:
    conf = yaml.safe_load(f)

if (torch.cuda.is_available()):
    device = torch.device("cuda")
    print("Device : " + str(torch.cuda.get_device_name(device)))
else:
    device = torch.device('cpu')

env_name = conf['env']['env_name']
task_name = conf['env']['task_name']
safety_coeff = conf['env']['safety_coeff']
lambda_ = conf['env']['lambda_']

env = make_rwrl_env(env_name, task_name, safety_coeff)

env = CustomEnv(env, env_name, lambda_)

max_training_timesteps = int(
    conf['alg']['max_training_timesteps'])
trajectory_episodes = conf['dataset']['episodes']

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
model = SAC("MlpPolicy",env,verbose=1,use_sde=True)
model.learn(total_timesteps=max_training_timesteps)

trajectories = []

with torch.no_grad():
    for e in tqdm(range(trajectory_episodes)):
        state = env.reset()
        initial_state = state[:-env.num_constraints]
        done = False
        trajectory = {"epoch": [], "t": [], "state": [], "action": [], "reward": [], "cost": [], "next_state": [],
                      "done": [], "initial_state": []}
        t = 0
        ep_reward = 0
        ep_cost = 0

        while not done:
            action, _states = model.predict(state, deterministic=False)
            noise = np.random.normal(0,0.15,action.shape)
            action = np.clip(action + noise, -1,1)

            next_state, reward, done, info, cost = env.log_step(action)

            trajectory['epoch'].append(e)
            trajectory['t'].append(t)
            trajectory['state'].append(state[:-env.num_constraints])
            trajectory['action'].append(action)
            trajectory['reward'].append(reward)
            trajectory['cost'].append(cost)
            trajectory['next_state'].append(next_state[:-env.num_constraints])
            trajectory['done'].append(done)
            trajectory['initial_state'].append(initial_state)

            t += 1
            state = next_state

            ep_reward += reward
            ep_cost += cost

        trajectories.append(trajectory)
        print("Reward: {}, Average cost: {}".format(round(ep_reward, 2), round(ep_cost / t, 2)))

directory = os.path.join('./results/datasets/', env_name)
if not os.path.exists(directory):
    os.makedirs(directory)

with open(os.path.join(directory, "trajectories_{}_{}.pkl".format(lambda_, trajectory_episodes)), "wb") as f:
    pickle.dump(trajectories, f)

print("save trajectories path: {}".format(directory))

env.close()
