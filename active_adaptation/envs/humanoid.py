from math import inf
import torch
from typing import Sequence, TYPE_CHECKING

from isaaclab.utils.math import quat_error_magnitude, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.sensors import ContactSensor, RayCaster, Imu
    from isaaclab.sensors import Camera, TiledCamera
# from isaaclab.sensors import ContactSensor, RayCaster
# from isaaclab.actuators import DCMotor
# from isaaclab.assets import Articulation
# from isaaclab.utils.math import yaw_quat
# from isaaclab.utils.warp import raycast_mesh
import active_adaptation
from active_adaptation.utils.helpers import batchify
from active_adaptation.utils.math import quat_rotate, quat_rotate_inverse

quat_rotate = batchify(quat_rotate)
quat_rotate_inverse = batchify(quat_rotate_inverse)
from tensordict.tensordict import TensorDictBase, TensorDict

from active_adaptation.envs.locomotion import SimpleEnv
import active_adaptation.envs.mdp as mdp

ADAPTIVE_SIGMA = {
    "sigma": {
        "tracking_anchor_pos": 0.16,
        "tracking_anchor_quat": 0.16,
        "tracking_qpos": 0.16,
        "tracking_kp_pos": 0.36,
        "tracking_kp_quat": 0.36,
        "tracking_kp_lin_vel": 1.0,
        "tracking_kp_ang_vel": 3.14,
    },
    "params": {
        "alpha": 1e-3
    }
}

class Humanoid(SimpleEnv):

    def __init__(self, cfg):
        super().__init__(cfg)
        # self.max_episode_length = torch.ones(self.num_envs, dtype=torch.long, device=self.device) * self.command_manager.num_frames
        self.start_frames = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.end_frames = torch.ones(self.num_envs, dtype=torch.long, device=self.device) * self.command_manager.num_frames
        self._init_adaptive_sigma()

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        if tensordict is not None:
            env_mask = tensordict.get("_reset").reshape(self.num_envs)
            env_ids = env_mask.nonzero().squeeze(-1)
            self.episode_count += env_ids.numel()
        else:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids):
            self._reset_idx(env_ids)
            self.scene.reset(env_ids)
        for callback in self._reset_callbacks:
            callback(env_ids)
        tensordict = TensorDict({}, self.num_envs, device=self.device)
        tensordict.update(self.observation_spec.zero())
        # self._compute_observation(tensordict)
        return tensordict
    
    def _reset_idx(self, env_ids: torch.Tensor):
        self.command_manager._update_stats(env_ids)
        start_frames, end_frames = self.command_manager.sample_init(env_ids)
            
        self.stats[env_ids] = 0.

        self.scene.reset(env_ids)

        self.episode_length_buf[env_ids] = self.start_frames[env_ids] = start_frames
        self.max_episode_length[env_ids] = self.end_frames[env_ids] = end_frames

        # in `self._reset_callbacks`
        # self.command_manager.reset(env_ids=env_ids)
        # self.action_manager.reset(env_ids=env_ids)

    def _compute_reward(self) -> TensorDictBase:
        rew_dict = super()._compute_reward()
        self.stats["episode_len"][:] = (self.episode_length_buf - self.start_frames).unsqueeze(1)
        self.stats["success"][:] = (self.episode_length_buf >= self.end_frames).unsqueeze(1).float()
        self.stats["episode_len_ratio"][:] = ((self.episode_length_buf - self.start_frames).float() / (self.end_frames - self.start_frames).float()).unsqueeze(1)
        return rew_dict

    # Observations of reference motion
    class ref_qpos(mdp.Observation):
        def __init__(self, env, joint_names=".*"):
            super().__init__(env)
            self.robot: Articulation = self.env.scene["robot"]
            self.joint_indices, self.joint_names = self.robot.find_joints(joint_names, preserve_order=True)

        def compute(self) -> torch.Tensor:
            timestep = self.env.episode_length_buf
            max_timestep = self.env.max_episode_length - 1
            timestep = torch.clamp(timestep, max=max_timestep)
            ref_qpos = self.env.command_manager.joint_pos[timestep]
            ref_qpos = ref_qpos[:, self.joint_indices]
            return ref_qpos.reshape(self.num_envs, -1)
        
    class ref_kp_pos_gap(mdp.Observation):
        def __init__(self, env):
            super().__init__(env)
            self.robot: Articulation = self.env.scene["robot"]
            self.keypoint_body_index = self.env.command_manager.keypoint_body_index

            self.ref_kp_pos = self.env.command_manager.body_pos_w[:, self.keypoint_body_index]    # (num_frames, num_keypoints, 3)
            self.ref_kp_quat = self.env.command_manager.body_quat_w[:, self.keypoint_body_index]

        def compute(self):
            timestep = self.env.episode_length_buf
            max_timestep = self.env.max_episode_length - 1
            timestep = torch.clamp(timestep, max=max_timestep)
            ref_kp_pos = self.ref_kp_pos[timestep]       # (num_envs, num_keypoints, 3)
            ref_kp_pos.add_(self.env.scene.env_origins[:, None])
            ref_kp_quat = self.ref_kp_quat[timestep]

            body_kp_pos = self.robot.data.body_pos_w[:, self.keypoint_body_index]
            body_kp_quat = self.robot.data.body_quat_w[:, self.keypoint_body_index]

            pos, _ = subtract_frame_transforms(body_kp_pos, body_kp_quat, ref_kp_pos, ref_kp_quat)
            return pos.reshape(self.num_envs, -1)

        def debug_draw(self):
            if active_adaptation._BACKEND == "isaac":
                timestep = self.env.episode_length_buf
                ref_kp_pos = self.ref_kp_pos[timestep]
                ref_kp_pos.add_(self.env.scene.env_origins[:, None])

                body_pos_global = self.robot.data.body_pos_w[:, self.keypoint_body_index]

                for i in range(ref_kp_pos.shape[1]):
                    self.env.debug_draw.point(ref_kp_pos[:, i], color=(1., 0., 0., 1.), size = 20)
                    self.env.debug_draw.point(body_pos_global[:, i], color=(0., 1., 0., 1.), size = 20)
    
    class ref_kp_quat(mdp.Observation):
        def __init__(self, env):
            super().__init__(env)
            self.robot: Articulation = self.env.scene["robot"]
            self.keypoint_body_index = self.env.command_manager.keypoint_body_index

            self.ref_kp_pos = self.env.command_manager.body_pos_w[:, self.keypoint_body_index]    # (num_frames, num_keypoints, 3)
            self.ref_kp_quat = self.env.command_manager.body_quat_w[:, self.keypoint_body_index]

        def compute(self):
            timestep = self.env.episode_length_buf
            max_timestep = self.env.max_episode_length - 1
            timestep = torch.clamp(timestep, max=max_timestep)
            ref_kp_pos = self.ref_kp_pos[timestep]       # (num_envs, num_keypoints, 3)
            ref_kp_pos.add_(self.env.scene.env_origins[:, None])
            ref_kp_quat = self.ref_kp_quat[timestep]

            body_kp_pos = self.robot.data.body_pos_w[:, self.keypoint_body_index]
            body_kp_quat = self.robot.data.body_quat_w[:, self.keypoint_body_index]

            _, quat = subtract_frame_transforms(body_kp_pos, body_kp_quat, ref_kp_pos, ref_kp_quat)
            return quat.reshape(self.num_envs, -1)


    def _init_adaptive_sigma(self):
        self._adaptive_sigma = {k: torch.tensor(v, device=self.device) for k, v in ADAPTIVE_SIGMA["sigma"].items()}
        self._error_ema = {k: torch.tensor(v, device=self.device) for k, v in self._adaptive_sigma.items()}
        self._alpha = ADAPTIVE_SIGMA["params"]["alpha"]

    def _update_adaptive_sigma(self, error, term):
        self._error_ema[term] = self._error_ema[term] * (1 - self._alpha) + error * self._alpha
        self._adaptive_sigma[term] = torch.minimum(self._adaptive_sigma[term], self._error_ema[term])
    
    # Motion Tracking Reward
    class tracking_anchor_pos(mdp.Reward):
        def __init__(self, env, weight: float, enabled: bool = True):
            super().__init__(env, weight, enabled)
            self.robot: Articulation = self.env.scene["robot"]
            self.anchor_body_index = self.env.command_manager.anchor_body_index

        def compute(self) -> torch.Tensor:
            timestep = (self.env.episode_length_buf-1)
            ref_anchor_pos_w = self.env.command_manager.body_pos_w[timestep][:, self.anchor_body_index]
            ref_anchor_pos_w.add_(self.env.scene.env_origins)
            anchor_pos_w = self.robot.data.body_pos_w[:, self.anchor_body_index]
            error = (anchor_pos_w - ref_anchor_pos_w).square().sum(-1, True)
            # reward = torch.exp(- error / self.sigma)
            reward = torch.exp(- error / self.env._adaptive_sigma["tracking_anchor_pos"])
            self.env._update_adaptive_sigma(error.mean(), "tracking_anchor_pos")
            return reward
        
    class tracking_anchor_quat(mdp.Reward):
        def __init__(self, env, weight: float, enabled: bool = True):
            super().__init__(env, weight, enabled)
            self.robot: Articulation = self.env.scene["robot"]
            self.anchor_body_index = self.env.command_manager.anchor_body_index

        def compute(self) -> torch.Tensor:
            timestep = (self.env.episode_length_buf-1)
            ref_anchor_quat_w = self.env.command_manager.body_quat_w[timestep][:, self.anchor_body_index]
            anchor_quat_w = self.robot.data.body_quat_w[:, self.anchor_body_index]
            error = (quat_error_magnitude(anchor_quat_w, ref_anchor_quat_w) ** 2).unsqueeze(-1)
            # reward = torch.exp(- error / self.sigma)
            reward = torch.exp(- error / self.env._adaptive_sigma["tracking_anchor_quat"])
            self.env._update_adaptive_sigma(error.mean(), "tracking_anchor_quat")
            return reward
        
    class tracking_qpos(mdp.Reward):
        def __init__(self, env, weight: float, enabled: bool = True, joint_names: str = ".*"):
            super().__init__(env, weight, enabled)
            self.robot: Articulation = self.env.scene["robot"]
            self.joint_indices, self.joint_names = self.robot.find_joints(joint_names, preserve_order=True)

        def compute(self) -> torch.Tensor:
            timestep = (self.env.episode_length_buf-1)
            ref_qpos = self.env.command_manager.joint_pos[timestep][:, self.joint_indices]
            qpos = self.robot.data.joint_pos[:, self.joint_indices]
            error = (qpos - ref_qpos).square().mean(-1, True)
            # reward = torch.exp(- error / self.sigma)
            reward = torch.exp(- error / self.env._adaptive_sigma["tracking_qpos"])
            self.env._update_adaptive_sigma(error.mean(), "tracking_qpos")
            return reward
        
    class tracking_kp_pos(mdp.Reward):
        def __init__(self, env, weight: float, enabled: bool = True):
            super().__init__(env, weight, enabled)
            self.robot: Articulation = self.env.scene["robot"]
            self.keypoint_body_index = self.env.command_manager.keypoint_body_index

        def compute(self) -> torch.Tensor:
            timestep = (self.env.episode_length_buf-1)
            ref_keypoints = self.env.command_manager.body_pos_w[timestep][:, self.keypoint_body_index]
            ref_keypoints.add_(self.env.scene.env_origins[:, None])

            body_pos_global = self.robot.data.body_pos_w[:, self.keypoint_body_index]

            error = (ref_keypoints - body_pos_global).square().sum(-1).mean(-1, True)
            # reward = torch.exp(- error / self.sigma)
            reward = torch.exp(- error / self.env._adaptive_sigma["tracking_kp_pos"])
            self.env._update_adaptive_sigma(error.mean(), "tracking_kp_pos")
            return reward
        
    class tracking_kp_quat(mdp.Reward):
        def __init__(self, env, weight: float, enabled: bool = True):
            super().__init__(env, weight, enabled)
            self.robot: Articulation = self.env.scene["robot"]
            self.keypoint_body_index = self.env.command_manager.keypoint_body_index

        def compute(self) -> torch.Tensor:
            timestep = (self.env.episode_length_buf-1)
            ref_keypoints = self.env.command_manager.body_quat_w[timestep][:, self.keypoint_body_index]

            body_quat_w = self.robot.data.body_quat_w[:, self.keypoint_body_index]
            error = (quat_error_magnitude(body_quat_w, ref_keypoints) ** 2).mean(-1, True)

            reward = torch.exp(- error / self.env._adaptive_sigma["tracking_kp_quat"])
            self.env._update_adaptive_sigma(error.mean(), "tracking_kp_quat")
            return reward

    class tracking_kp_lin_vel(mdp.Reward):
        def __init__(self, env, weight: float, enabled: bool = True):
            super().__init__(env, weight, enabled)
            self.robot: Articulation = self.env.scene["robot"]
            self.keypoint_body_index = self.env.command_manager.keypoint_body_index

        def compute(self) -> torch.Tensor:
            timestep = (self.env.episode_length_buf-1)
            ref_lin_vel = self.env.command_manager.body_lin_vel_w[timestep][:, self.keypoint_body_index]
            body_lin_vel_w = self.robot.data.body_lin_vel_w[:, self.keypoint_body_index]
            error = (body_lin_vel_w - ref_lin_vel).square().sum(-1).mean(-1, True)
            reward = torch.exp(- error / self.env._adaptive_sigma["tracking_kp_lin_vel"])
            self.env._update_adaptive_sigma(error.mean(), "tracking_kp_lin_vel")
            return reward

    class tracking_kp_ang_vel(mdp.Reward):
        def __init__(self, env, weight: float, enabled: bool = True):
            super().__init__(env, weight, enabled)
            self.robot: Articulation = self.env.scene["robot"]
            self.keypoint_body_index = self.env.command_manager.keypoint_body_index

        def compute(self) -> torch.Tensor:
            timestep = (self.env.episode_length_buf-1)
            ref_ang_vel = self.env.command_manager.body_ang_vel_w[timestep][:, self.keypoint_body_index]
            body_ang_vel_w = self.robot.data.body_ang_vel_w[:, self.keypoint_body_index]
            error = (body_ang_vel_w - ref_ang_vel).square().sum(-1).mean(-1, True)
            reward = torch.exp(- error / self.env._adaptive_sigma["tracking_kp_ang_vel"])
            self.env._update_adaptive_sigma(error.mean(), "tracking_kp_ang_vel")
            return reward

    class mean_kp_error(mdp.Reward):
        def __init__(self, env, weight: float, enabled: bool = True):
            super().__init__(env, weight, enabled)
            self.robot: Articulation = self.env.scene["robot"]
            self.keypoint_body_index = self.env.command_manager.keypoint_body_index
            
        def compute(self) -> torch.Tensor:
            timestep = (self.env.episode_length_buf-1)
            ref_keypoints = self.env.command_manager.body_pos_w[timestep][:, self.keypoint_body_index]
            ref_keypoints.add_(self.env.scene.env_origins[:, None])
            body_pos_global = self.robot.data.body_pos_w[:, self.keypoint_body_index]
            error = (ref_keypoints - body_pos_global).norm(dim=-1)
            return error.mean(-1, True)

    # Early Termination Conditions
    class dummy(mdp.Termination):
        def __init__(self, env):
            super().__init__(env)
            self.device = self.env.device

        def compute(self, termination: torch.Tensor) -> torch.Tensor:
            return torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.bool)

    class root_deviation(mdp.Termination):
        def __init__(self, env, max_distance: float):
            super().__init__(env)
            self.device = self.env.device
            self.max_distance = torch.tensor(max_distance, device=self.env.device)
            self.robot: Articulation = self.env.scene["robot"]

        def compute(self, termination: torch.Tensor) -> torch.Tensor:
            timestep = (self.env.episode_length_buf - 1)
            ref_root_translation = self.env.command_manager.root_pos_w[timestep]
            ref_root_translation.add_(self.env.scene.env_origins)
            root_pos_w = self.robot.data.root_pos_w
            deviation = (root_pos_w - ref_root_translation).norm(dim=1, keepdim=True)
            return deviation > self.max_distance
        
    class root_rot_deviation(mdp.Termination):
        def __init__(self, env, threshold: float):
            super().__init__(env)
            self.device = self.env.device
            self.threshold = threshold
            self.robot: Articulation = self.env.scene["robot"]

        def compute(self, termination: torch.Tensor) -> torch.Tensor:
            timestep = (self.env.episode_length_buf - 1)
            ref_quat_w = self.env.command_manager.root_quat_w[timestep]
            root_quat_w = self.robot.data.root_quat_w

            ref_projected_gravity_b = quat_rotate_inverse(ref_quat_w, self.robot.data.GRAVITY_VEC_W)
            projected_gravity_b = quat_rotate_inverse(root_quat_w, self.robot.data.GRAVITY_VEC_W)

            diff = (projected_gravity_b[:, 2] - ref_projected_gravity_b[:, 2]).abs().unsqueeze(-1)
            return diff > self.threshold
        
    class track_kp_error(mdp.Termination):
        def __init__(self, env, threshold: float, body_names: str = ".*"):
            super().__init__(env)
            self.device = self.env.device
            self.threshold = threshold
            self.robot: Articulation = self.env.scene["robot"]
            self.body_indices = [self.robot.body_names.index(name) for name in body_names]

        def compute(self, termination: torch.Tensor) -> torch.Tensor:
            timestep = (self.env.episode_length_buf - 1)
            ref_keypoints = self.env.command_manager.body_pos_w[timestep][:, self.body_indices]
            ref_keypoints.add_(self.env.scene.env_origins[:, None])

            body_pos_global = self.robot.data.body_pos_w[:, self.body_indices]

            # diff = (ref_keypoints - body_pos_global).norm(dim=-1)    # (num_envs, num_bodies)
            # return diff.mean(-1, True) > self.threshold
            diff = (ref_keypoints[:, :, 2] - body_pos_global[:, :, 2]).abs()
            return (diff > self.threshold).any(dim=-1, keepdim=True)

def dot(a: torch.Tensor, b: torch.Tensor):
    return (a * b).sum(-1, True)