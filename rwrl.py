import realworldrl_suite.environments as rwrl
import dm2gym.envs.dm_suite_env as dm2gym

PER_DOMAIN_HARDEST_CONSTRAINT = {
    'cartpole': 'slider_pos_constraint',
    'humanoid': 'joint_angle_constraint',
    'quadruped': 'joint_angle_constraint',
    'walker': 'joint_velocity_constraint'}


def make_rwrl_env(env_name, task_name, safety_spec):
    safety_spec_dict = {
        'enable': True,
        'binary': True,
        'observations': True,
        'safety_coeff': safety_spec,
    }
    env = rwrl.load(
        domain_name=env_name,
        task_name=task_name,
        safety_spec=safety_spec_dict,
        environment_kwargs=dict(log_safety_vars=False, flat_observation=True))
    return env



class GymEnv(dm2gym.DMSuiteEnv):

    def __init__(self, env):
        self.env = env
        self.constraints_list = list(env._task.constraints)
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': round(1. / self.env.control_timestep())
        }
        self.observation_space = dm2gym.convert_dm_control_to_gym_space(
            self.env.observation_spec())
        self.action_space = dm2gym.convert_dm_control_to_gym_space(
            self.env.action_spec())
        self.viewer = None


class CustomEnv(GymEnv):
    def __init__(self, env, env_name, lamb):
        super().__init__(env)
        self.num_constraints = len(self.constraints_list)
        self.constraints_index = self.constraints_list.index(PER_DOMAIN_HARDEST_CONSTRAINT[env_name])
        self.lamb = lamb

    def step(self, action):
        obs, reward, done, info = super().step(action)
        constraints = obs[-self.num_constraints:]
        cost = constraints[self.constraints_index]
        reward -= self.lamb * cost
        return obs, reward, done, info

    def log_step(self, action):
        obs, reward, done, info = super().step(action)
        constraints = obs[-self.num_constraints:]
        cost = constraints[self.constraints_index]
        return obs, reward, done, info, cost

