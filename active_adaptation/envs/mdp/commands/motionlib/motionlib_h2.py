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

class MotionLibH2(MotionLib):
    source_fps: int = 30
    target_fps: int = 50
    isaacsim_joints = [
                "left_hip_yaw_joint", "right_hip_yaw_joint", 
                "torso_joint", 
                "left_hip_pitch_joint", "right_hip_pitch_joint", 
                "left_shoulder_pitch_joint", "right_shoulder_pitch_joint", 
                "left_hip_roll_joint", "right_hip_roll_joint", 
                "left_shoulder_roll_joint", "right_shoulder_roll_joint", 
                "left_knee_joint", "right_knee_joint", 
                "left_shoulder_yaw_joint", "right_shoulder_yaw_joint", 
                "left_ankle_pitch_joint", "right_ankle_pitch_joint", 
                "left_elbow_joint", "right_elbow_joint", 
                "left_ankle_roll_joint", "right_ankle_roll_joint", 
                "left_wrist_roll_joint", "right_wrist_roll_joint", 
                "left_wrist_pitch_joint", "right_wrist_pitch_joint", 
                "left_wrist_yaw_joint", "right_wrist_yaw_joint"
            ]
    mujoco_joints = [
                "left_hip_yaw_joint",
                "left_hip_pitch_joint",
                "left_hip_roll_joint",
                "left_knee_joint",
                "left_ankle_pitch_joint",
                "left_ankle_roll_joint",
                "right_hip_yaw_joint",
                "right_hip_pitch_joint",
                "right_hip_roll_joint",
                "right_knee_joint",
                "right_ankle_pitch_joint",
                "right_ankle_roll_joint",
                "torso_joint",
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
                "left_wrist_roll_joint",
                "left_wrist_pitch_joint",
                "left_wrist_yaw_joint",
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_joint",
                "right_wrist_roll_joint",
                "right_wrist_pitch_joint",
                "right_wrist_yaw_joint"
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