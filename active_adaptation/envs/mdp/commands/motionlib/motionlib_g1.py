from math import pi
import torch
import torch.distributions as D
import torch.nn.functional as F
from typing import Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.sensors import ContactSensor, RayCaster, Imu
    from isaaclab.sensors import Camera, TiledCamera
 
import active_adaptation
from active_adaptation.envs.mdp.observations.motion import joint_vel
from active_adaptation.utils.math import quat_rotate, quat_rotate_inverse, MultiUniform
from active_adaptation.utils.helpers import batchify

import joblib
import os
import importlib.util
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from tqdm import tqdm
import numpy as np
from scipy.interpolate import interp1d

from .motionlib import MotionLib

spec = importlib.util.find_spec("active_adaptation")
package_path = spec.origin

quat_rotate_inverse = batchify(quat_rotate_inverse)

CURRENT_MOTION = 0

class MotionLibG1(MotionLib):
    source_fps: int = 30
    target_fps: int = 50
    isaacsim_joints = [
                "left_hip_pitch_joint", "right_hip_pitch_joint",
                "waist_yaw_joint", 
                # "waist_roll_joint", "waist_pitch_joint",
                "left_hip_roll_joint", "right_hip_roll_joint",
                "left_hip_yaw_joint", "right_hip_yaw_joint",
                "left_knee_joint", "right_knee_joint",
                "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
                "left_ankle_pitch_joint", "right_ankle_pitch_joint",
                "left_shoulder_roll_joint", "right_shoulder_roll_joint",
                "left_ankle_roll_joint", "right_ankle_roll_joint",
                "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
                "left_elbow_joint", "right_elbow_joint",
                "left_wrist_roll_joint", "right_wrist_roll_joint",
                # "left_wrist_pitch_joint", "right_wrist_pitch_joint",
                # "left_wrist_yaw_joint", "right_wrist_yaw_joint"
            ]
    mujoco_joints = [                       # g1 23 dof version
                "left_hip_pitch_joint",
                "left_hip_roll_joint",
                "left_hip_yaw_joint",
                "left_knee_joint",
                "left_ankle_pitch_joint",
                "left_ankle_roll_joint",
                "right_hip_pitch_joint",
                "right_hip_roll_joint",
                "right_hip_yaw_joint",
                "right_knee_joint",
                "right_ankle_pitch_joint",
                "right_ankle_roll_joint",
                "waist_yaw_joint",
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
                "left_wrist_roll_joint",
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_joint",
                "right_wrist_roll_joint"
            ]
    
    def __init__(
            self, 
            env,
            motion_clip_dir: str,
            dataset: str,
            occlusion: str,
            joint_matches: list,
            mode: str = "train",
            eval_id: int = None,
            teleop: bool = False,
        ):
        super().__init__(
            env,
            motion_clip_dir,
            dataset,
            occlusion,
            joint_matches,
            mode,
            eval_id,
            teleop,
        )
    
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
        init_root_state[:, :3] = self.root_translations[start_frames].to(self.device) + self.env_origin[env_ids]
        init_root_state[:, :3] += torch.tensor([0, 0, 0.01], device=self.device)
        init_root_state[:, 3:7] = self.root_orientation[start_frames].to(self.device)

        qpos = self.qpos[start_frames].to(self.device)
        n_tgt = self.robot.data.default_joint_pos.shape[1]
        pad = n_tgt - qpos.shape[1]
        qpos = F.pad(qpos, (0, pad))
        self.robot.write_joint_state_to_sim(
            qpos,
            self.robot.data.default_joint_vel[env_ids],
            joint_ids = slice(None),
            env_ids=env_ids
        )
        
        return init_root_state, start_frames.to(self.device), end_frames.to(self.device)

    def load_data(self, data):
        self.motion_length = []
        self.phase = []
        self.root_translations = []
        self.root_orientation = []
        self.root_linear = []
        self.qpos = []
        self.kp_global = []
        self.kp_local = []
        self.contact = []

        mujoco_to_isaac_idx = self.mujoco_to_isaac()
        smpl_idx = [SMPL_BONE_ORDER_NAMES.index(j[1]) for j in self.joint_matches]

        pbar = tqdm(data.items())
        for k, motion in pbar:
            pbar.set_description(f"Loading {k}: ")
            interpolated_root_trans = self.interpolate(motion, "root_trans_offset", self.source_fps, self.target_fps)
            interpolated_root_rot = self.interpolate(motion, "root_rot", self.source_fps, self.target_fps)
            interpolated_qpos = self.interpolate(motion, "dof", self.source_fps, self.target_fps)[:, :23]
            interpolated_kp_global = self.interpolate(motion, "smpl_joints", self.source_fps, self.target_fps)
            interpolated_kp_local = convert2local(interpolated_kp_global, interpolated_root_rot)

            contact = contact_from_positions(interpolated_kp_global, 
                                            left_foot_idx=SMPL_BONE_ORDER_NAMES.index("L_Ankle"),
                                            right_foot_idx=SMPL_BONE_ORDER_NAMES.index("R_Ankle"),
                                            v_thresh=getattr(self, "v_thresh", 0.01),
                                            h_thresh=None)

            interpolated_kp_global = interpolated_kp_global[:, smpl_idx, :]
            interpolated_kp_local = interpolated_kp_local[:, smpl_idx, :]

            self.motion_length.append(interpolated_root_trans.shape[0])
            self.phase.append(torch.linspace(0, 1, interpolated_root_trans.shape[0]))
            self.root_translations.append(interpolated_root_trans)
            self.root_linear.append(torch.diff(interpolated_root_trans,
                                               dim=0,
                                               append=torch.zeros(1, 3)) * self.target_fps)
            self.root_orientation.append(interpolated_root_rot[:, [3, 0, 1, 2]])
            self.qpos.append(interpolated_qpos[:, mujoco_to_isaac_idx])
            self.kp_global.append(interpolated_kp_global)
            self.kp_local.append(interpolated_kp_local)
            self.contact.append(contact)
            breakpoint()

        self.motion_length = torch.tensor(self.motion_length)
        self.phase = torch.cat(self.phase, dim=0).float()
        self.root_translations = torch.cat(self.root_translations, dim=0).float()
        self.root_orientation = torch.cat(self.root_orientation, dim=0).float()
        self.root_linear = torch.cat(self.root_linear, dim=0).float()
        self.qpos = torch.cat(self.qpos, dim=0).float()
        self.kp_global = torch.cat(self.kp_global, dim=0).float()
        self.kp_local = torch.cat(self.kp_local, dim=0).float()
        self.contact = torch.cat(self.contact, dim=0).float()

        self.num_motions = len(data)
        self.num_frames = self.root_translations.shape[0]

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

from matplotlib import pyplot as plt
import matplotlib.animation as animation

def animate_3d(joints, orientation):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    vis = 1

    def update(num, data, line):
        ax.clear()
        ax.scatter(data[num][:, 0], data[num][:, 1], data[num][:, 2], c='y', marker='o')
        ax.scatter(data[num][vis, 0], data[num][vis, 1], data[num][vis, 2], c='r', marker='*', s=50)

        # unit_vector = np.array([1, 0, 0])
        # unit_vector = R.apply(R.from_quat(orientation[num]), unit_vector)
        # ax.quiver(0, 0, 0, unit_vector[0], unit_vector[1], unit_vector[2], color='r', length=0.5)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_title(f"Frame {num}")

        ax.quiver(0, 0, 0, 1, 0, 0, color='r', length=0.1)
        ax.quiver(0, 0, 0, 0, 1, 0, color='g', length=0.1)
        ax.quiver(0, 0, 0, 0, 0, 1, color='b', length=0.1)
        return line,

    ani = animation.FuncAnimation(fig, update, frames=joints.shape[0], fargs=(joints, None), interval=50)
    plt.show()
            
SMPL_BONE_ORDER_NAMES = [
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "Torso",
    "L_Knee",
    "R_Knee",
    "Spine",
    "L_Ankle",
    "R_Ankle",
    "Chest",
    "L_Toe",
    "R_Toe",
    "Neck",
    "L_Thorax",
    "R_Thorax",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    "L_Hand",
    "R_Hand",
]

# g1 29 dof version
# mujoco_joints = [
#     "left_hip_pitch_joint",
#     "left_hip_roll_joint",
#     "left_hip_yaw_joint",
#     "left_knee_joint",
#     "left_ankle_pitch_joint",
#     "left_ankle_roll_joint",

#     "right_hip_pitch_joint",
#     "right_hip_roll_joint",
#     "right_hip_yaw_joint",
#     "right_knee_joint",
#     "right_ankle_pitch_joint",
#     "right_ankle_roll_joint",

#     "waist_yaw_joint",
#     "waist_roll_joint",
#     "waist_pitch_joint",

#     "left_shoulder_pitch_joint",
#     "left_shoulder_roll_joint",
#     "left_shoulder_yaw_joint",
#     "left_elbow_joint",
#     "left_wrist_roll_joint",
#     "left_wrist_pitch_joint",
#     "left_wrist_yaw_joint",

#     "right_shoulder_pitch_joint",
#     "right_shoulder_roll_joint",
#     "right_shoulder_yaw_joint",
#     "right_elbow_joint",
#     "right_wrist_roll_joint",
#     "right_wrist_pitch_joint",
#     "right_wrist_yaw_joint"
# ]