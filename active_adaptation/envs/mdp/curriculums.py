from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TYPE_CHECKING, Optional

import torch

from active_adaptation.envs.mdp.base import Curriculum
from active_adaptation.utils.math import quat_rotate_inverse


if TYPE_CHECKING:
    from isaaclab.assets import Articulation


@dataclass
class _AssistanceSchedule:
    initial: float
    final: float
    decay_episodes: int

    def value(self, episode_count: int) -> float:
        if self.decay_episodes <= 0:
            return self.final
        progress = min(max(episode_count, 0) / float(self.decay_episodes), 1.0)
        return (1.0 - progress) * self.initial + progress * self.final


class external_force(Curriculum):
    """Apply assistance forces that decay as training progresses."""

    def __init__(
        self,
        env,
        body_names: Optional[Sequence[str]] = None,
        kp: float | Sequence[float] = 250.0,
        kd: float | Sequence[float] = 40.0,
        max_force: Optional[float] = 1500.0,
        initial_strength: float = 1.0,
        final_strength: float = 0.0,
        decay_episodes: int = 250_000,
        train_only: bool = True,
        enabled: bool = True,
    ):
        super().__init__(env, enabled=enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.train_only = train_only

        if body_names is None:
            if hasattr(self.command_manager, "keypoint_body_index"):
                indices = list(self.command_manager.keypoint_body_index)
            else:
                raise ValueError(
                    "body_names must be provided when the command manager does not expose keypoint indices."
                )
            self.body_indices = torch.as_tensor(indices, device=self.device, dtype=torch.long)
            self.body_names = [self.asset.body_names[i] for i in indices]
        else:
            ids, names = self.asset.find_bodies(body_names, preserve_order=True)
            self.body_indices = torch.as_tensor(ids, device=self.device, dtype=torch.long)
            self.body_names = names

        if self.body_indices.numel() == 0:
            raise ValueError("external_force curriculum requires at least one body to target")

        self.kp = self._prepare_gain(kp)
        self.kd = self._prepare_gain(kd)
        self.max_force = None if max_force is None else torch.tensor(max_force, device=self.device)
        self.schedule = _AssistanceSchedule(initial_strength, final_strength, decay_episodes)
        self._strength = torch.tensor(initial_strength, device=self.device)
        self._last_force_norm = torch.tensor(0.0, device=self.device)

    def _prepare_gain(self, gain: float | Sequence[float]) -> torch.Tensor:
        tensor = torch.as_tensor(gain, device=self.device, dtype=torch.float32)
        if tensor.ndim == 0:
            tensor = tensor.expand(self.body_indices.numel()).clone()
        elif tensor.numel() != self.body_indices.numel():
            raise ValueError(
                f"Expected {self.body_indices.numel()} gain values, got {tensor.numel()}"
            )
        return tensor.view(1, -1, 1)

    def startup(self):
        self._update_strength()

    def reset(self, env_ids: torch.Tensor):
        del env_ids  # unused
        self._last_force_norm.zero_()

    def pre_step(self, substep: int):
        if not self.enabled:
            return
        if self.train_only and not getattr(self.env, "training", True):
            return
        if self._strength.item() <= 0.0:
            return

        timestep = self._current_timestep()
        ref_pos = self.command_manager.body_pos_w.index_select(0, timestep)
        ref_vel = self.command_manager.body_lin_vel_w.index_select(0, timestep)
        ref_pos = ref_pos[:, self.body_indices] + self.env.scene.env_origins[:, None]
        ref_vel = ref_vel[:, self.body_indices]

        body_pos = self.asset.data.body_pos_w[:, self.body_indices]
        body_vel = self.asset.data.body_lin_vel_w[:, self.body_indices]

        pos_error = ref_pos - body_pos
        vel_error = ref_vel - body_vel

        force_w = (self.kp * pos_error + self.kd * vel_error) * self._strength
        if self.max_force is not None:
            norm = force_w.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            scale = torch.minimum(self.max_force / norm, torch.ones_like(norm))
            force_w = force_w * scale

        body_quat = self.asset.data.body_quat_w[:, self.body_indices]
        force_b = quat_rotate_inverse(body_quat, force_w)
        self.asset._external_force_b[:, self.body_indices] += force_b
        self.asset.has_external_wrench = True

        self._last_force_norm = force_w.norm(dim=-1).mean()

    def update(self):
        self._update_strength()
        self.env.extra["curriculum/external_force_scale"] = float(self._strength.item())
        self.env.extra["curriculum/external_force_norm"] = float(self._last_force_norm.item())

    def _update_strength(self):
        strength = self.schedule.value(int(self.env.episode_count))
        self._strength.fill_(strength)

    def _current_timestep(self) -> torch.Tensor:
        max_timestep = torch.maximum(self.env.max_episode_length - 1, torch.zeros_like(self.env.max_episode_length))
        timestep = torch.minimum(self.env.episode_length_buf, max_timestep)
        return timestep.to(torch.long)

    def debug_draw(self):
        pass