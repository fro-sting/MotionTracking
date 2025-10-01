from ast import List
from math import pi
import torch
import torch.distributions as D
import torch.nn.functional as F
from typing import Sequence, List, TYPE_CHECKING

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
            dataset: str,
            occlusion: str,
            anchor_body: str = None,
            keypoint_body: List[str] = None,
            mode: str = "train",
            eval_id: int = None,
            teleop: bool = False,
        ):
        super().__init__(env, teleop=teleop)
        self.robot: Articulation = env.scene["robot"]
        package_dir = os.path.dirname(package_path)

        occlusion_path = os.path.join(package_dir, "..", motion_clip_dir, "..", occlusion)
        occlusion_keys = list(joblib.load(occlusion_path).keys())

        motion_clip = os.path.join(package_dir, "..", motion_clip_dir, dataset) + ".pkl"

        data = joblib.load(motion_clip)
        data = {k.replace("_stageii", "_poses"): v for k, v in data.items()}
        data = {k: v for k, v in data.items() if k not in occlusion_keys}

        if eval_id is not None:
            data_keys = list(data.keys())
            data = {data_keys[eval_id]: data[data_keys[eval_id]]}
        
        self.env_origin = self.env.scene.env_origins
        self.anchor_body_index = self.robot.body_names.index(anchor_body)
        self.keypoint_body_index = [self.robot.body_names.index(body) for body in keypoint_body]

        self.load_data(data)
        assert len(self.robot.body_names) == self.body_pos_w.shape[1]
        assert len(self.robot.joint_names) == self.joint_pos.shape[1]
        print(f"Loaded {len(data)} motion clips with {self.num_frames} frames.")

        self.min_weight = 3e-3
        self.alpha0, self.beta0 = 1.0, 1.0
        self.trials = torch.zeros(self.num_motions)
        self.failures = torch.zeros(self.num_motions)
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
    
    def sample_init(self, env_ids: torch.Tensor) -> torch.Tensor:
        if self.mode in ["play", "eval"]:
            motion_ids = torch.ones(env_ids.shape[0], dtype=torch.long) * CURRENT_MOTION
        else:
            probs = self._sampling_probs()
            motion_ids = D.Categorical(probs).sample((env_ids.shape[0],))

        self.curr_motion_id[env_ids] = motion_ids.to(self.device)
        
        start_frames = self.start_frames[motion_ids]
        end_frames = self.end_frames[motion_ids]
        motion_length = self.motion_length[motion_ids]

        if self.mode == "train":
            r = torch.rand(motion_length.shape) * 0.5
            offsets = (r * motion_length.float()).floor().long()
            start_frames += offsets

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
        
        return start_frames.to(self.device), end_frames.to(self.device)
    
    def reset(self, env_ids: torch.Tensor):
        pass

    def _update_stats(self, env_ids: torch.Tensor):
        mids = self.curr_motion_id[env_ids].cpu()
        valid = mids >= 0
        if valid.any():
            success = (self.env.stats["success"][env_ids].squeeze(-1) > 0.5)
            failed = (~success).to(self.trials.dtype).cpu()

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
        self.joint_pos = torch.cat(self.joint_pos, dim=0).float()
        self.joint_vel = torch.cat(self.joint_vel, dim=0).float()
        self.body_pos_w = torch.cat(self.body_pos_w, dim=0).float()
        self.body_quat_w = torch.cat(self.body_quat_w, dim=0).float()
        self.body_lin_vel_w = torch.cat(self.body_lin_vel_w, dim=0).float()
        self.body_ang_vel_w = torch.cat(self.body_ang_vel_w, dim=0).float()

        self.root_pos_w = self.body_pos_w[:, 0]
        self.root_quat_w = self.body_quat_w[:, 0]
        self.root_lin_vel_w = self.body_lin_vel_w[:, 0]
        self.root_ang_vel_w = self.body_ang_vel_w[:, 0]

        self.num_motions = len(data)
        self.num_frames = self.joint_pos.shape[0]

        self.start_frames = torch.cat([torch.zeros(1), self.motion_length.cumsum(dim=0)[:-1]]).long()
        self.end_frames = self.motion_length.cumsum(dim=0).long()

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

def contact_from_positions(kp_global, left_foot_idx, right_foot_idx, v_thresh=0.01, h_thresh=0.01):
    r'''
    Args:
        kp_global: (N, 24, 3)  torch tensor in global coordinate system
        left_foot_idx: int
        right_foot_idx: int
        v_thresh: float
        h_thresh: float
    '''
    feet_l = kp_global[:, left_foot_idx, :]
    feet_l_vel = (torch.diff(feet_l, dim=0) ** 2).sum(dim=-1)
    feet_l_still = feet_l_vel < v_thresh

    feet_r = kp_global[:, right_foot_idx, :]
    feet_r_vel = (torch.diff(feet_r, dim=0) ** 2).sum(dim=-1)
    feet_r_still = feet_r_vel < v_thresh

    feet_l_still = feet_l_still.unsqueeze(-1)
    feet_r_still = feet_r_still.unsqueeze(-1)
    feet_still = torch.cat([feet_l_still, feet_r_still], dim=-1)    # (N-1, 2）
    feet_still = torch.cat([feet_still[:1], feet_still], dim=0)     # (N, 2)
    return feet_still

class MotionLibG1(MotionLib):
    
    def __init__(
            self, 
            env,
            motion_clip_dir: str,
            dataset: str,
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
            eval_id: int = None,
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