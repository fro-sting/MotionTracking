import torch
import einops
import warnings
from typing import Dict, Literal, Tuple, Union, TYPE_CHECKING, List
from tensordict import TensorDictBase
import isaaclab.utils.string as string_utils
from active_adaptation.utils.math import (
    # quat_mul,
    # quat_conjugate,
    # axis_angle_from_quat,
    # quat_inv,
    quat_rotate_inverse,
    quat_rotate,
    yaw_quat,
)
import active_adaptation.utils.symmetry as symmetry_utils

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from active_adaptation.envs.base import _Env

TENSORLIKE = Union[torch.Tensor, TensorDictBase]

class ActionManager:

    action_dim: int

    def __init__(self, env):
        self.env: _Env = env
        self.asset: Articulation = self.env.scene["robot"]
        self.action_buf: torch.Tensor

    def reset(self, env_ids: torch.Tensor):
        pass

    def debug_draw(self):
        pass

    def apply_action(self, tensordict: TENSORLIKE, substep: int):
        pass

    @property
    def num_envs(self):
        return self.env.num_envs

    @property
    def device(self):
        return self.env.device

class JointPosition(ActionManager):
    def __init__(
        self,
        env,
        action_scaling: Dict[str, float] = 0.5,
        max_delay: int = 2,  # delay in simulation steps
        alpha: Union[float, Tuple[float, float]] = (0.5, 1.0),
        **kwargs,
    ):
        super().__init__(env)
        if len(kwargs) > 0: # warn user that kwargs are deprecated
            warnings.warn(f"JointPosition kwargs are deprecated: {kwargs}")
        self.joint_ids, self.joint_names, self.action_scaling = (
            string_utils.resolve_matching_names_values(
                dict(action_scaling), self.asset.joint_names
            )
        )
        self.joint_ids = torch.tensor(self.joint_ids, device=self.device)

        self.action_scaling = torch.tensor(self.action_scaling, device=self.device)
        self.max_delay = max_delay
        self.action_dim = len(self.joint_ids)

        import omegaconf

        if isinstance(alpha, float):
            self.alpha_range = (alpha, alpha)
        elif isinstance(alpha, omegaconf.listconfig.ListConfig):
            self.alpha_range = tuple(alpha)
        else:
            raise ValueError(f"Invalid alpha type: {type(alpha)}")

        self.default_joint_pos = self.asset.data.default_joint_pos.clone()
        self.offset = torch.zeros_like(self.default_joint_pos)
        self.decimation = int(self.env.step_dt / self.env.physics_dt)

        with torch.device(self.device):
            self.action_buf = torch.zeros(self.num_envs, self.action_dim, 4, device=self.device) # TODO: permute to (num_envs, 4, action_dim)
            self.action_queue = torch.zeros(self.num_envs, self.max_delay + self.decimation, self.action_dim)
            self.applied_action = torch.zeros(self.num_envs, self.action_dim)
            self.alpha = torch.ones(self.num_envs, 1)
            self.delay = torch.zeros(self.num_envs, 1, dtype=int)

    def resolve(self, spec):
        return string_utils.resolve_matching_names_values(dict(spec), self.asset.joint_names)

    def symmetry_transforms(self):
        transform = symmetry_utils.joint_space_symmetry(self.asset, self.joint_names)
        return transform

    def reset(self, env_ids: torch.Tensor):
        self.delay[env_ids] = torch.randint(0, self.max_delay + 1, (len(env_ids), 1), device=self.device)
        self.action_buf[env_ids] = 0
        self.applied_action[env_ids] = 0

        default_joint_pos = self.asset.data.default_joint_pos[env_ids]
        self.default_joint_pos[env_ids] = default_joint_pos + self.offset[env_ids]

        alpha = torch.empty(len(env_ids), 1, device=self.device)
        alpha.uniform_(self.alpha_range[0], self.alpha_range[1])
        self.alpha[env_ids] = alpha

    def apply_action(self, action: TENSORLIKE, substep: int):
        if substep == 0:
            if isinstance(action, TensorDictBase):
                action = action["action"]
            self.action_buf = self.action_buf.roll(1, dims=-1)
            self.action_buf[:, :, 0] = action
            self.action_queue = torch.where(
                (torch.arange(self.action_queue.shape[1], device=self.device) < self.delay).reshape(self.num_envs, self.action_queue.shape[1], 1),
                self.action_queue,
                action.unsqueeze(1)
            )
        # deplay model: each substep, the first action in queue is consumed
        self.applied_action.lerp_(self.action_queue[:, 0], self.alpha)
        self.action_queue = self.action_queue.roll(-1, dims=1)

        pos_target = self.default_joint_pos.clone()
        pos_target[:, self.joint_ids] += self.applied_action * self.action_scaling
        self.asset.set_joint_position_target(pos_target)

def clamp_norm(x: torch.Tensor, max_norm: float):
    norm = x.norm(dim=-1, keepdim=True)
    return x * (max_norm / norm.clamp(min=1e-6)).clamp(max=1.0)
