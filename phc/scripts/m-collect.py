import os
import time
from typing import List, Dict
import hydra
from omegaconf import DictConfig
import joblib
from tqdm import tqdm

import jax
import jax.numpy as jnp
import numpy as np
import mujoco
import mujoco.viewer

from loco_mujoco.environments import UnitreeH1v2
from loco_mujoco.trajectory import Trajectory, TrajectoryInfo, TrajectoryModel, TrajectoryData, TrajectoryHandler

FPS = 30.0
DATASET_DIR = "./datasets"
# DATASET_NAME = "DefaultDatasets"
DATASET_NAME = "Lafan1"

TRACKED_BODY_NAMES = [  "left_hip_pitch_link", "left_knee_link", "left_ankle_roll_link",
                        "right_hip_pitch_link", "right_knee_link", "right_ankle_roll_link",
                        "left_shoulder_pitch_link", "left_elbow_link", "left_rubber_hand",
                        "right_shoulder_pitch_link", "right_elbow_link", "right_rubber_hand"]

SMPL_BONE_ORDER_NAMES = [
    "Pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "Torso",
    "left_knee_link",
    "right_knee_link",
    "Spine",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "Chest",
    "L_Toe",
    "R_Toe",
    "Neck",
    "L_Thorax",
    "R_Thorax",
    "Head",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_elbow_link",
    "right_elbow_link",
    "L_Wrist",
    "R_Wrist",
    "left_rubber_hand",
    "right_rubber_hand",
]

def list_traj_files(path: str, exts=(".npz", ".npy", ".pkl")) -> List[str]:
    files = []
    for f in os.listdir(path):
        if f.endswith(exts):
            files.append(os.path.join(path, f))
    return sorted(files)

def map_qpos_into_model(mj_data: mujoco.MjData, q: np.ndarray) -> None:
    """
    Map source qpos (shape: 3 + 4 + 21) into target model layout (3 + 4 + 27),
    matching the slicing scheme you used in the viewer script.
    """
    # root
    mj_data.qpos[:3] = q[:3]
    mj_data.qpos[3:7] = q[3:7]

    # legs (12 DoF)
    mj_data.qpos[7:19] = q[7:19]

    # waist
    mj_data.qpos[19:20] = q[19:20]    # torso joint

    # left arm: source 4 DoF -> target, 7 DoF, zero extras
    mj_data.qpos[20:24] = q[20:24]
    mj_data.qpos[24:27] = 0.0

    # right arm
    mj_data.qpos[27:31] = q[24:28]
    mj_data.qpos[31:34] = 0.0

@hydra.main(version_base=None, config_path="../cfg", config_name="unitree_h2_fitting")
def main(cfg : DictConfig) -> None:
    env = UnitreeH1v2(init_state_type="DefaultInitialStateHandler")
    env_model = env.get_model()
    
    humanoid_xml = cfg.asset.assetFileName
    mj_model = mujoco.MjModel.from_xml_path(humanoid_xml)
    mj_data = mujoco.MjData(mj_model)
    # print(mj_data.qpos.shape)           (3 + 4 + 27, )

    body_ids = []
    for name in SMPL_BONE_ORDER_NAMES:
        bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
        # if bid == -1:
        #     raise ValueError(f"Body {name} not found in model")
        body_ids.append(bid)
    n_tracked = len(body_ids)
    
    collected: Dict[str, Dict] = {}

    traj_files = list_traj_files(os.path.join(DATASET_DIR, DATASET_NAME, "UnitreeH1v2"))
    for path in tqdm(traj_files):
        data_key = DATASET_NAME + "_" + path.split("/")[-1].split(".")[0] + "_stageii"
        
        th = TrajectoryHandler(
            model=env_model,
            traj_path=path,
            control_dt=1 / FPS
        )
        qpos_seq: np.ndarray = th.traj.data.qpos        # (T, 3 + 4 + 21)
        T = qpos_seq.shape[0]
        mapped_qpos_seq = np.zeros((T, mj_data.qpos.shape[0]))

        xpos_tracked = np.zeros((T, n_tracked, 3))
        for t in range(T):
            map_qpos_into_model(mj_data, qpos_seq[t])
            mujoco.mj_forward(mj_model, mj_data)
            
            mapped_qpos_seq[t] = mj_data.qpos.copy()
            for i, bid in enumerate(body_ids):
                if bid != -1:
                    xpos_tracked[t, i] = mj_data.xpos[bid].copy()

        # quat (w, x, y, z) --> (x, y, z, w)
        data_dump = {
            "root_trans_offset": np.array(mapped_qpos_seq[:, :3].copy()),
            "root_rot": np.array(mapped_qpos_seq[:, 3:7].copy())[:, [1, 2, 3, 0]],
            "dof": np.array(mapped_qpos_seq[:, 7:].copy()),
            "smpl_joints": xpos_tracked,
        }

        collected[data_key] = data_dump
    
    joblib.dump(collected, os.path.join(DATASET_DIR, f"{DATASET_NAME}.pkl"))
    print(f"Saved {len(collected)} trajectories to {os.path.join(DATASET_DIR, f'{DATASET_NAME}.pkl')}")

if __name__ == "__main__":
    main()