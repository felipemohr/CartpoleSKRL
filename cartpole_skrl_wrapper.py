from typing import Any, Tuple

import gymnasium as gym
import numpy as np
import torch

from skrl.envs.wrappers.torch import Wrapper

class CartpoleSKRLWrapper(Wrapper):
    def __init__(self, env: Any) -> None:
        super().__init__(env)

    @property
    def state_space(self) -> gym.Space:
        return self._env.observation_space 
    
    @property
    def observation_space(self) -> gym.Space:
        return self._env.observation_space
    
    @property
    def action_space(self) -> gym.Space:
        return self._env.action_space

    def _tensor_to_action(self, actions: torch.Tensor) -> torch.Tensor: # -> np.array:
        # space = self.action_space
        # return np.array(actions.cpu().numpy(), dtype=space.dtype).flatten()
        return actions.flatten()

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        observation, reward, terminated, truncated, info = self._env.step(self._tensor_to_action(actions))

        observation = observation.to(device=self.device).view(self.num_envs, -1)
        reward = reward.to(device=self.device).view(self.num_envs, -1)
        terminated = terminated.to(device=self.device).view(self.num_envs, -1)
        truncated = truncated.to(device=self.device).view(self.num_envs, -1)

        return observation, reward, terminated, truncated, info

    def reset(self) -> Tuple[torch.Tensor, Any]:
        observation, info = self._env.reset()
        observation = observation.to(device=self.device)
        return observation, info

    def render(self, *args, **kwargs) -> None:
        return self._env.render(*args, **kwargs)

    def close(self) -> None:
        self._env.close()
