from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.gym.tasks.rl_task import RLTaskInterface
from omni.isaac.cloner import GridCloner
from pxr import UsdGeom
from gymnasium import spaces
import omni.usd
import numpy as np
import torch
import math

class CartpoleRLTask(RLTaskInterface):
    def __init__(self, name, env, device="cpu", num_envs=1, offset=None) -> None:
        
        super().__init__(name=name, env=env, offset=offset)

        self._device = device
        self._num_envs = num_envs
        self._num_actions = 1
        self._num_observations = 4
        self._num_states = 4
        self._num_agents = 1
        self._env_spacing = 5.0

        self.action_space = spaces.Box(
            np.ones(self._num_actions, dtype=np.float32) * -1.0, np.ones(self._num_actions, dtype=np.float32) * 1.0
        )
        self.observation_space = spaces.Box(
            np.ones(self._num_observations, dtype=np.float32) * -np.Inf,
            np.ones(self._num_observations, dtype=np.float32) * np.Inf,
        )
        self.state_space = spaces.Box(
            np.ones(self._num_states, dtype=np.float32) * -np.Inf,
            np.ones(self._num_states, dtype=np.float32) * np.Inf,
        )

        self._cartpole_position = torch.tensor([0.0, 0.0, 2.0])
        self._reset_dist = 3.0
        self._max_push_effort = 400.0
        self._max_episode_length = 500
        
        self.cleanup()

    def cleanup(self) -> None:
        self.obs_buf = torch.zeros((self._num_envs, self._num_observations), dtype=torch.float)
        self.states_buf = torch.zeros((self._num_envs, self._num_states), dtype=torch.float)
        self.rew_buf = torch.zeros(self._num_envs, device=self._device, dtype=torch.float)
        self.reset_buf = torch.ones(self._num_envs, device=self._device, dtype=torch.long)
        self.progress_buf = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)
        self.extras = {}

    def set_up_scene(
        self, scene, replicate_physics=True, collision_filter_global_paths=[], filter_collisions=True, copy_from_source=False
    ) -> None:
        super().set_up_scene(scene)

        assets_root_path = get_assets_root_path()
        usd_path = assets_root_path + "/Isaac/Robots/Cartpole/cartpole.usd"

        self._cloner = GridCloner(self._env_spacing)
        self._cloner.define_base_env("/World/envs")

        stage = omni.usd.get_context().get_stage()
        UsdGeom.Xform.Define(stage, "/World/envs/env_0")

        self._ground_plane_path = "/World/defaultGroundPlane"
        collision_filter_global_paths.append(self._ground_plane_path)
        scene.add_default_ground_plane(prim_path=self._ground_plane_path)

        prim_paths = self._cloner.generate_paths("/World/envs/env", self._num_envs)
        self._env_pos = self._cloner.clone(
            source_prim_path="/World/envs/env_0", prim_paths=prim_paths, replicate_physics=replicate_physics, copy_from_source=copy_from_source
        )
        self._env_pos = torch.tensor(np.array(self._env_pos), device=self._device, dtype=torch.float)

        if filter_collisions:
            self._cloner.filter_collisions(
                self._env.world.get_physics_context().prim_path,
                "/World/collisions",
                prim_paths,
                collision_filter_global_paths,
            )

        create_prim(prim_path="/World/envs/env_0/Cartpole", prim_type="Xform", position=self._cartpole_position)
        add_reference_to_stage(usd_path, "/World/envs/env_0/Cartpole")

        self._cartpoles = ArticulationView(
            prim_paths_expr="/World/envs/.*/Cartpole", name="cartpole_view", reset_xform_properties=False
        )
        scene.add(self._cartpoles)

        # TODO: Add camera

    def pre_physics_step(self, actions) -> None:
        if not self._env.world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions, device=self.device)
        actions = actions.to(self._device)

        forces = torch.zeros((self._cartpoles.count, self._cartpoles.num_dof), dtype=torch.float32, device=self._device)
        forces[:, self._cart_dof_idx] = self._max_push_effort * actions[:]

        indices = torch.arange(self._cartpoles.count, dtype=torch.int32, device=self._device)
        self._cartpoles.set_joint_efforts(forces, indices=indices)

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)

        # randomize DOF positions
        dof_pos = torch.zeros((num_resets, self._cartpoles.num_dof), device=self._device)
        dof_pos[:, self._cart_dof_idx] = 1.0 * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
        dof_pos[:, self._pole_dof_idx] = 0.125 * math.pi * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))

        # randomize DOF velocities
        dof_vel = torch.zeros((num_resets, self._cartpoles.num_dof), device=self._device)
        dof_vel[:, self._cart_dof_idx] = 0.5 * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
        dof_vel[:, self._pole_dof_idx] = 0.25 * math.pi * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))

        # apply resets
        indices = env_ids.to(dtype=torch.int32)
        self._cartpoles.set_joint_positions(dof_pos, indices=indices)
        self._cartpoles.set_joint_velocities(dof_vel, indices=indices)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
    
    def reset(self):
        self.reset_buf = torch.ones_like(self.reset_buf)
        return self.get_observations()

    def post_reset(self):
        self._cart_dof_idx = self._cartpoles.get_dof_index("cartJoint")
        self._pole_dof_idx = self._cartpoles.get_dof_index("poleJoint")
        # randomize all envs
        indices = torch.arange(self._cartpoles.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def post_physics_step(self):
        self.progress_buf[:] += 1

        if self._env.world.is_playing():
            self.get_observations()
            self.get_states()
            self.calculate_metrics()
            self.is_done()
            self.get_extras()

        return self.obs_buf, self.rew_buf, self.reset_buf, self.reset_buf, self.extras

    def get_observations(self) -> dict:
        dof_pos = self._cartpoles.get_joint_positions(clone=False)
        dof_vel = self._cartpoles.get_joint_velocities(clone=False)

        self.cart_pos = dof_pos[:, self._cart_dof_idx]
        self.cart_vel = dof_vel[:, self._cart_dof_idx]
        self.pole_pos = dof_pos[:, self._pole_dof_idx]
        self.pole_vel = dof_vel[:, self._pole_dof_idx]

        self.obs_buf[:, 0] = self.cart_pos
        self.obs_buf[:, 1] = self.cart_vel
        self.obs_buf[:, 2] = self.pole_pos
        self.obs_buf[:, 3] = self.pole_vel

        return self.obs_buf
    
    def get_states(self) -> dict:
        self.states_buf = self.obs_buf
        return self.states_buf

    def calculate_metrics(self):
        reward = 1.0 - self.pole_pos * self.pole_pos - 0.01 * torch.abs(self.cart_vel) - 0.005 * torch.abs(self.pole_vel)
        reward = torch.where(torch.abs(self.cart_pos) > self._reset_dist, torch.ones_like(reward) * -2.0, reward)
        reward = torch.where(torch.abs(self.pole_pos) > math.pi / 2, torch.ones_like(reward) * -2.0, reward)

        self.rew_buf[:] = reward
        return self.rew_buf

    def is_done(self):
        resets = torch.where(torch.abs(self.cart_pos) > self._reset_dist, 1, 0)
        resets = torch.where(torch.abs(self.pole_pos) > math.pi / 2, 1, resets)
        resets = torch.where(self.progress_buf >= self._max_episode_length, 1, resets)
        self.reset_buf[:] = resets
        # TODO: Truncated and Terminated
        return self.reset_buf
    