from omni.isaac.gym.vec_env import VecEnvBase
env = VecEnvBase(headless=False)

from cartpole_task import CartpoleRLTask
task = CartpoleRLTask(name="Cartpole", env=env, device="cuda", num_envs=1)
env.set_task(task, backend="torch", sim_params={"sim_device": "cuda:0"})

from cartpole_skrl_wrapper import CartpoleSKRLWrapper
env = CartpoleSKRLWrapper(env)

import torch
import torch.nn as nn

from skrl.models.torch import GaussianMixin, Model

class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 64),
                                 nn.Tanh(),
                                 nn.Linear(64, 64),
                                 nn.Tanh(),
                                 nn.Linear(64, self.num_actions))
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return torch.tanh(self.net(inputs["states"])), self.log_std_parameter, {}


# path = "runs/torch/Cartpole/obs_gpu_100k_1env_PPO/checkpoints/best_agent.pt"
path = "runs/torch/Cartpole/100k_256envs_PPO/checkpoints/best_agent.pt"
agent = torch.load(path)

policy = agent["policy"]

model = Policy(env.observation_space, env.action_space, task.device, clip_actions=True)
model.load_state_dict(policy)
model = model.to(device=task.device)

env._world.reset()
obs, _ = env.reset()
while env._simulation_app.is_running():
    obs = obs.to(device=task.device)
    states = {"states": obs}
    _, _, action = model(states)
    obs, rewards, terminated, truncated, info = env.step(action["mean_actions"])

env.close()
