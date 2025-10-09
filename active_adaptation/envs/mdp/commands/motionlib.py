from ast import List
from math import pi
import torch
import torch.distributions as D
import torch.nn.functional as F
from typing import Sequence, List, TYPE_CHECKING, Union, Optional, Tuple

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.sensors import ContactSensor, RayCaster, Imu
    from isaaclab.sensors import Camera, TiledCamera
 
import active_adaptation
from active_adaptation.envs.mdp.observations.motion import joint_vel
from active_adaptation.utils.math import quat_rotate, quat_rotate_inverse, MultiUniform
from active_adaptation.utils.helpers import batchify
from active_adaptation.envs.mdp.base import Command

import joblib
import os
import importlib.util
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from tqdm import tqdm
import numpy as np
from scipy.interpolate import interp1d

spec = importlib.util.find_spec("active_adaptation")
package_path = spec.origin

quat_rotate_inverse = batchify(quat_rotate_inverse)

CURRENT_MOTION = 0

class MotionLib(Command):
    def __init__(
            self, 
            env,
            motion_clip_dir: str,
            dataset: Union[List[str], str],
            occlusion: str,
            anchor_body: Optional[str] = None,
            keypoint_body: Optional[List[str]] = None,
            mode: str = "train",
            eval_id: Optional[Union[int, tuple[int, int]]] = None,
            teleop: bool = False,
        ):
        super().__init__(env, teleop=teleop)
        self.robot: Articulation = env.scene["robot"]
        self.env_origin = self.env.scene.env_origins
        self.anchor_body_index = self.robot.body_names.index(anchor_body)
        self.keypoint_body_index = [self.robot.body_names.index(body) for body in keypoint_body]

        package_dir = os.path.dirname(package_path)
        occlusion_path = os.path.join(package_dir, "..", motion_clip_dir, "..", occlusion)
        occlusion_keys = list(joblib.load(occlusion_path).keys())

        # Support both single string (backward compatibility) and list of strings
        if isinstance(dataset, str):
            dataset = [dataset]
        
        # Load and merge all datasets
        data = {}
        for dataset_name in dataset:
            motion_clip = os.path.join(package_dir, "..", motion_clip_dir, dataset_name) + ".pkl"
            dataset_data = joblib.load(motion_clip)
            dataset_data = {k.replace("_stageii", "_poses"): v for k, v in dataset_data.items()}
            dataset_data = {k: v for k, v in dataset_data.items() if k not in occlusion_keys}
            
            data.update(dataset_data)
        
        print(f"Loaded motion clips from {len(dataset)} dataset(s)")

        self._per_env_fixed = False
        self._motion_for_env: Optional[torch.Tensor] = None

        if eval_id is not None:
            data_keys = list(data.keys())
            if isinstance(eval_id, int):
                # Single motion by index
                idx = int(eval_id)
                assert 0 <= idx < len(data_keys), (
                f"eval_id {idx} out of range for {len(data_keys)} motions"
                )
                data = {data_keys[idx]: data[data_keys[idx]]}
            else:
                start, end = eval_id
                assert 0 <= start < end <= len(data_keys), (
                f"Invalid eval_id slice {eval_id}; total motions: {len(data_keys)}"
                )
                keep_keys = data_keys[start:end]
                num_eval_motion = end - start
                assert self.num_envs == num_eval_motion, (
                f"num_envs ({self.num_envs}) must equal slice length ({num_eval_motion})"
                )
                data = {k: data[k] for k in keep_keys}
                # per-env fixed mapping: env i -> motion i
                self._per_env_fixed = True
                self._motion_for_env = torch.arange(num_eval_motion, device=self.device)

        self.load_data(data)
        assert len(self.robot.body_names) == self.body_pos_w.shape[1]
        assert len(self.robot.joint_names) == self.joint_pos.shape[1]

        if self._per_env_fixed:
            self.num_frames = int(self.motion_length.max())
        else:
            self.num_frames = int(self.joint_pos.shape[0])
        print(f"Loaded {len(data)} motion clips with {self.num_frames} frames.")

        BASELINE_MASS = 0.02
        self.min_weight = BASELINE_MASS / self.num_motions
        self.alpha0, self.beta0 = 1.0, 1.0
        self.trials = torch.zeros(self.num_motions, device=self.device)
        self.failures = torch.zeros(self.num_motions, device=self.device)
        self.curr_motion_id = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)

        self.mode = mode
        if mode == "play":
            from pynput import keyboard
            def on_press(key):
                global CURRENT_MOTION
                try:
                    if key.char == "n":
                        CURRENT_MOTION += 1
                        CURRENT_MOTION %= self.num_motions
                        print(f"\nSwitching to motion {CURRENT_MOTION}")
                except AttributeError:
                    pass
            self.listener = keyboard.Listener(on_press=on_press)
            self.listener.start()
        
    #     if active_adaptation._BACKEND == "mujoco":
    #         self.marker = self.env.scene.create_sphere_marker(0.05, (0, 1, 0, 1))
            
    # def debug_draw(self):
    #     if active_adaptation._BACKEND == "mujoco":
    #         self.marker.geom.pos = self.robot.data.body_pos_w[0, 10]

    @torch.no_grad()
    def _sampling_probs(self) -> torch.Tensor:
        denom = (self.trials + self.alpha0 + self.beta0).clamp_min(1e-6)
        p_fail = (self.failures + self.alpha0) / denom

        w = p_fail.clamp_min(self.min_weight)
        return w / w.sum()

    def _pick_motion_ids(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Return a (len(env_ids),) tensor of motion ids for these envs, honoring mode & slicing."""
        # Deterministic mapping: env i -> motion i
        if self._per_env_fixed:
            assert self._motion_for_env is not None
            return self._motion_for_env[env_ids]

        # In eval/play without slicing, use CURRENT_MOTION for all selected envs
        if self.mode in ("play", "eval"):
            motion_id = int(CURRENT_MOTION % max(self.num_motions, 1))
            return torch.full((env_ids.shape[0],), motion_id, dtype=torch.long, device=self.device)

        # Default: train — weighted by failure-biased bandit
        probs = self._sampling_probs().to(self.device)
        return D.Categorical(probs).sample((env_ids.shape[0],))
    
    def _choose_start_frames(self, motion_ids: torch.Tensor) -> torch.Tensor:
        start_frames = self.start_frames[motion_ids]

        # Bias towards earlier bins to diversify starting phases
        if self.mode == "train" and not self._per_env_fixed:
            motion_length = self.motion_length[motion_ids]
            bin_size = 100
            max_bins = ((motion_length - 1) // bin_size).clamp_min(0)
            cap = torch.div(max_bins, 3, rounding_mode='floor') # floor(max_bins/3)
            r = torch.rand_like(max_bins, dtype=torch.float32)
            bin_ids = torch.floor(r * (cap.to(torch.float32) + 1.0)).to(torch.long)
            start_frames += bin_ids * bin_size
        return start_frames
    
    def sample_init(self, env_ids: torch.Tensor) -> torch.Tensor:
        motion_ids = self._pick_motion_ids(env_ids)
        self.curr_motion_id[env_ids] = motion_ids
        
        start_frames = self._choose_start_frames(motion_ids)
        end_frames = self.end_frames[motion_ids]
        
        init_root_state = self.init_root_state[env_ids]     # (num_envs, 3 + 4 + 6) root position, root orientation, root linear velocity and root angular velocity
        init_root_state[:, :3] = self.root_pos_w[start_frames].to(self.device) + self.env_origin[env_ids]
        init_root_state[:, :3] += torch.tensor([0, 0, 0.01], device=self.device)
        init_root_state[:, 3:7] = self.root_quat_w[start_frames].to(self.device)
        init_root_state[:, 7:10] = self.root_lin_vel_w[start_frames].to(self.device)
        init_root_state[:, 10:] = self.root_ang_vel_w[start_frames].to(self.device)
        self.robot.write_root_state_to_sim(
                init_root_state, 
                env_ids=env_ids
            )

        joint_pos = self.joint_pos[start_frames].to(self.device)
        joint_vel = self.joint_vel[start_frames].to(self.device)
        self.robot.write_joint_state_to_sim(
            joint_pos,
            joint_vel,
            joint_ids = slice(None),
            env_ids=env_ids
        )
        
        return start_frames, end_frames
    
    def reset(self, env_ids: torch.Tensor):
        pass

    def _update_stats(self, env_ids: torch.Tensor):
        mids = self.curr_motion_id[env_ids]
        valid = mids >= 0
        if valid.any():
            success = (self.env.stats["success"][env_ids].squeeze(-1) > 0.5)
            failed = (~success).to(self.trials.dtype)

            ones = torch.ones_like(failed, dtype=self.trials.dtype)

            self.trials.index_add_(0, mids[valid], ones[valid])
            self.failures.index_add_(0, mids[valid], failed[valid])

        self.curr_motion_id[env_ids] = -1
        
    def load_data(self, data):
        self.motion_length = []
        self.joint_pos = []
        self.joint_vel = []
        self.body_pos_w = []
        self.body_quat_w = []
        self.body_lin_vel_w = []
        self.body_ang_vel_w = []

        pbar = tqdm(data.items())
        for k, motion in pbar:
            pbar.set_description(f"Loading {k}: ")
            joint_pos = torch.from_numpy(motion["joint_pos"])
            joint_vel = torch.from_numpy(motion["joint_vel"])
            body_pos_w = torch.from_numpy(motion["body_pos_w"])
            body_quat_w = torch.from_numpy(motion["body_quat_w"])
            body_lin_vel_w = torch.from_numpy(motion["body_lin_vel_w"])
            body_ang_vel_w = torch.from_numpy(motion["body_ang_vel_w"])

            self.motion_length.append(joint_pos.shape[0])
            self.joint_pos.append(joint_pos)
            self.joint_vel.append(joint_vel)
            self.body_pos_w.append(body_pos_w)
            self.body_quat_w.append(body_quat_w)
            self.body_lin_vel_w.append(body_lin_vel_w)
            self.body_ang_vel_w.append(body_ang_vel_w)

        self.motion_length = torch.tensor(self.motion_length)
        self.joint_pos = torch.cat(self.joint_pos, dim=0).float().to(self.device)
        self.joint_vel = torch.cat(self.joint_vel, dim=0).float().to(self.device)
        self.body_pos_w = torch.cat(self.body_pos_w, dim=0).float().to(self.device)
        self.body_quat_w = torch.cat(self.body_quat_w, dim=0).float().to(self.device)
        self.body_lin_vel_w = torch.cat(self.body_lin_vel_w, dim=0).float().to(self.device)
        self.body_ang_vel_w = torch.cat(self.body_ang_vel_w, dim=0).float().to(self.device)

        self.root_pos_w = self.body_pos_w[:, 0]
        self.root_quat_w = self.body_quat_w[:, 0]
        self.root_lin_vel_w = self.body_lin_vel_w[:, 0]
        self.root_ang_vel_w = self.body_ang_vel_w[:, 0]

        self.num_motions = len(data)
        self.num_frames = self.joint_pos.shape[0]

        self.start_frames = torch.cat([torch.zeros(1), self.motion_length.cumsum(dim=0)[:-1]]).long().to(self.device)
        self.end_frames = self.motion_length.cumsum(dim=0).long().to(self.device)
        self.motion_length = self.motion_length.to(self.device)

    # # for sanity check
    # def update(self):
    #     self.frames = self.env.episode_length_buf.cpu()
    #     env_ids = torch.arange(self.num_envs, device=self.device)
        
    #     root_state = self.robot.data.root_state_w.clone()
    #     root_state[:, :3] = self.root_translations[self.frames].to(self.device) + self.env_origin + torch.tensor([0, 0, 1.], device=self.device)
    #     root_state[:, 3:7] = self.root_orientation[self.frames].to(self.device)
    #     self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)

    #     qpos = self.qpos[self.frames].to(self.device)
    #     self.robot.write_joint_state_to_sim(
    #         qpos,
    #         self.robot.data.default_joint_vel,
    #         env_ids=env_ids
    #     )
    #     return

class MotionLibG1(MotionLib):
    
    def __init__(
            self, 
            env,
            motion_clip_dir: str,
            dataset: List[str],
            occlusion: str,
            anchor_body: str = "torso_link",
            keypoint_body: List[str] = [
                                        "pelvis",
                                        "left_hip_pitch_link", "right_hip_pitch_link", 
                                        "left_knee_link", "right_knee_link", 
                                        "left_ankle_roll_link", "right_ankle_roll_link", 
                                        "left_shoulder_roll_link", "right_shoulder_roll_link", 
                                        "left_elbow_link", "right_elbow_link", 
                                        "left_wrist_yaw_link", "right_wrist_yaw_link"
                                        ],
            mode: str = "train",
            eval_id: Optional[Union[int, List[int]]] = None,
            teleop: bool = False,
        ):
        super().__init__(
            env,
            motion_clip_dir,
            dataset,
            occlusion,
            anchor_body,
            keypoint_body,
            mode,
            eval_id,
            teleop,
        )