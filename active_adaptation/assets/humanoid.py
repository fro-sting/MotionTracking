import os
import copy
import isaaclab.sim as sim_utils
import torch
from isaaclab_assets import H1_CFG
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg, IdealPDActuatorCfg
from isaaclab.assets import Articulation
from active_adaptation.envs.actuator import HybridActuatorCfg
import active_adaptation.utils.symmetry as symmetry_utils

from .base import ArticulationCfg


ASSET_PATH = os.path.dirname(__file__)

G1_29DOF_CFG = ArticulationCfg( # no wrist pitch and yaw
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        urdf_path=f"{ASSET_PATH}/unitree_description/urdf/g1/main.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, 
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.78),
        joint_pos={
            ".*_hip_pitch_joint": -0.1,
            ".*_knee_joint": 0.6,
            ".*_ankle_pitch_joint": -0.2,
            ".*_elbow_joint": 1.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": ImplicitActuatorCfg(
            joint_names_expr=".*",
            effort_limit_sim={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 139.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
                ".*waist_yaw_joint": 88.0,
                ".*waist_roll_joint": 35.0,
                ".*waist_pitch_joint": 35.0,
                ".*_ankle_pitch_joint": 35.0,
                ".*_ankle_roll_joint": 35.0,
                ".*_shoulder_pitch_joint": 25.0,
                ".*_shoulder_roll_joint": 25.0,
                ".*_shoulder_yaw_joint": 25.0,
                ".*_elbow_joint": 25.0,
                ".*_wrist_yaw_joint": 5.0,
                ".*_wrist_roll_joint": 25.0,
                ".*_wrist_pitch_joint": 5.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
                ".*waist_yaw_joint": 32.0,
                ".*waist_roll_joint": 30.0,
                ".*waist_pitch_joint": 30.0,
                ".*_ankle_pitch_joint": 30.0,
                ".*_ankle_roll_joint": 30.0,
                ".*_shoulder_pitch_joint": 37.0,
                ".*_shoulder_roll_joint": 37.0,
                ".*_shoulder_yaw_joint": 37.0,
                ".*_elbow_joint": 37.0,
                ".*_wrist_yaw_joint": 22.0,
                ".*_wrist_roll_joint": 37.0,
                ".*_wrist_pitch_joint": 22.0,
            },
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 200.0,
                ".*waist.*": 200.0,
                ".*_ankle_pitch_joint": 20.0,
                ".*_ankle_roll_joint": 20.0,
                ".*_shoulder_pitch_joint": 100.0,
                ".*_shoulder_roll_joint": 100.0,
                ".*_shoulder_yaw_joint": 50.0,
                ".*_elbow_joint": 50.0,
                ".*_wrist_yaw_joint": 40.0,
                ".*_wrist_roll_joint": 40.0,
                ".*_wrist_pitch_joint": 40.0,
            },
            damping={
                ".*_hip_yaw_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_pitch_joint": 5.0,
                ".*_knee_joint": 5.0,
                ".*waist.*": 5.0,
                ".*_ankle_pitch_joint": 2.0,
                ".*_ankle_roll_joint": 2.0,
                ".*_shoulder_pitch_joint": 2.0,
                ".*_shoulder_roll_joint": 2.0,
                ".*_shoulder_yaw_joint": 2.0,
                ".*_elbow_joint": 2.0,
                ".*_wrist_yaw_joint": 2.0,
                ".*_wrist_roll_joint": 2.0,
                ".*_wrist_pitch_joint": 2.0,
            },
            armature=0.01,
            friction=0.01,
        ),
    }
)