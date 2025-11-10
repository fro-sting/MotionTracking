# active-adaptation

## Features
* Automatic shape handling for observation.
* Clean and efficient single-file RL implementation.
* Easy symmetry augmentation.
* Seamless Mujoco sim2sim.

## Current Limitations
* TorchRL stores redundant information and therefore the rollout buffer consumes more GPU memory.

## Installation (Ubuntu 22.04)

0. First install `uv`:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
1. For the following steps, the recommended way to structure the (VSCode or Cursor) workspace is:
   ```bash
    ${workspaceFolder}/ # File->Open Folder here
      lab51/    # uv venv
      active-adaptation/
      IsaacLab/
        _isaac_sim/
   ```
2. Install [Isaac Sim 5.1.0](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/download.html) by downloading the latest release and unzip it to a desired location `$ISAACSIM_PATH`.
3. Install [Isaac Lab (update to 2025.11.08)](https://github.com/isaac-sim/IsaacLab) and setup a conda environment:
   ```bash
   uv venv --python 3.11 lab51
   source lab51/bin/activate
   uv pip install -U pip
   # install IsaacLab to the exisiting conda environment
   # git clone https://github.com/isaac-sim/IsaacLab.git
   git clone git@github.com:isaac-sim/IsaacLab.git # SSH recommended
   cd IsaacLab
   ln -s $ISAACSIM_PATH _isaac_sim
   ./isaaclab.sh --uv ../lab51
   ./isaaclab.sh -i none # install without additional RL libraries
   # reactivate the environment
   cd ..
   deactivate
   source lab51/bin/activate
   echo $PYTHONPATH 
   ```
   You should see the isaac-sim related dependencies are added to `$PYTHONPATH`.
4. [**Optional**] VSCode setup. This enables the Python extension for code analysis to provide auto-completiong and linting. Edit `.vscode/settings.json` on demand:
   ```json
   "python.analysis.extraPaths": [
        // Recommended
        "./IsaacLab/source/isaaclab",
        "./IsaacLab/source/isaaclab_assets",
        // Optional, modified from IsaacLab/.vscode/settings.json
        "${workspaceFolder}/IsaacLab/_isaac_sim/exts/isaacsim.replicator.behavior",
        "${workspaceFolder}/IsaacLab/_isaac_sim/exts/isaacsim.replicator.behavior.ui",
        "${workspaceFolder}/IsaacLab/_isaac_sim/exts/isaacsim.replicator.domain_randomization",
        "${workspaceFolder}/IsaacLab/_isaac_sim/exts/isaacsim.replicator.examples",
        "${workspaceFolder}/IsaacLab/_isaac_sim/exts/isaacsim.replicator.scene_blox",
        "${workspaceFolder}/IsaacLab/_isaac_sim/exts/isaacsim.replicator.synthetic_recorder",
        "${workspaceFolder}/IsaacLab/_isaac_sim/exts/isaacsim.replicator.writers",
        //... note that adding extraPaths may increase VSCode CPU usage
    ],
   ```
6. `uv pip install -U torch torchvision tensordict torchrl`
7. Install this repo:
   ```bash
   git clone git@github.com:xiaohu-art/MotionTracking.git # SSH recommended
   cd active-adaptation
   uv pip install -e . 
   # if you encounter errors about evdev, you can:
   # unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS GCC_EXEC_PREFIX
   uv pip install xxhash mujoco joblib
   ```

# Data Preparation
1. Create folder `scripts/data/g1` and put the `.pkl` file [retargeted by PHC](https://github.com/xiaohu-art/phc-retarget) in this folder.
2. Run 
    ```bash
    python process/load_pkl.py --input_file <motions>.pkl --input_fps 30 --output_fps 50 --headless
    ```
    to interpolate the motion to 50 FPS and convert the data to isaacsim format.
3. For [LAFAN1 dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset), run

    ```bash
    hf download lvhaidong/LAFAN1_Retargeting_Dataset --repo-type dataset --local-dir <path/to/lafan1_data>
    ```
    and then run

    ```bash
    python process/csv_to_pkl.py --input_dir <path/to/lafan1_data> --input_fps 30 --output_file <motions>.pkl --output_fps 50
    ```

## Basic Usage

We use Hydra for configuration management. Each task is specified by a yaml file placed under `cfg/task` or `cfg/task/{subfolder}`, for example:

```yaml
# @package task
name: motion

viewer:
  resolution: [1280, 720]
  lookat: [0., 0., 0.]
  eye: [3.0, 3.0, 3.0]

robot: 
  name: g1_29dof
terrain: plane

num_envs: 4096
max_episode_length: 1000

sim:
  step_dt: 0.02
  isaac_physics_dt: 0.005
  mujoco_physics_dt: 0.002

action:
  _target_: active_adaptation.envs.mdp.action.JointPosition
  action_scaling:
    .*hip.*: 0.5
    .*knee.*: 0.5
    .*ankle_pitch.*: 0.5
    waist_yaw_joint: 0.5
    .*shoulder.*: 0.5
    .*elbow.*: 0.5
  max_delay: 1
  alpha: [0.6, 0.8]

command:
  _target_: active_adaptation.envs.mdp.MotionLibG1
  motion_clip_dir: "scripts/data/g1"
  dataset: [pklfile1, pklfile2, ...]
  occlusion: "amass_copycat_occlusion_v3.pkl"
  mode: train
  eval_id: null
  
observation:
  robot:
    root_quat_w:
    root_angvel_b:        {noise_std: 0.05}
    projected_gravity_b:  {noise_std: 0.01}
    joint_pos:            {noise_std: 0.05}
    joint_vel:            {noise_std: 0.2}
    body_pos:             {body_names: [left_hip_pitch_link, right_hip_pitch_link, 
                                        left_knee_link, right_knee_link, 
                                        left_ankle_roll_link, right_ankle_roll_link, 
                                        left_shoulder_roll_link, right_shoulder_roll_link, 
                                        left_elbow_link, right_elbow_link, 
                                        left_wrist_yaw_link, right_wrist_yaw_link], 
                                        yaw_only: false}
    prev_actions:         {steps: 1}
  ref_motion_:
    ref_orientation:      {}
    ref_qpos:             {}
    ref_kp_pos_gap:       {}
    ref_trans_gap:        {}
  priv:
    root_height:      {}
    root_linvel_b:    {}
    body_vel:         {body_names: [left_hip_pitch_link, right_hip_pitch_link, 
                                    left_knee_link, right_knee_link, 
                                    left_ankle_roll_link, right_ankle_roll_link, 
                                    left_shoulder_roll_link, right_shoulder_roll_link, 
                                    left_elbow_link, right_elbow_link, 
                                    left_wrist_yaw_link, right_wrist_yaw_link], 
                                    yaw_only: false}
    # joint_forces:     {}

reward:
  loco:
    tracking_root_trans:    {weight: 2., enabled: true}
    tracking_root_rot:      {weight: 2., enabled: true}
    tracking_qpos:          {weight: 2., enabled: true}
    tracking_keypoints:     {weight: 3., enabled: true}

    feet_slip:          {weight: 1.0, enabled: true, body_names: .*ankle_roll_link}

    action_rate_l2:     {weight: 0.5, enabled: true}

randomization:
  push:
    body_names: ["torso_link", "pelvis"]
    force_range: [0.0, 0.1]
  perturb_body_mass:
    ".*wrist_yaw_link": [1.0, 2.0]
    "^(?!.*wrist_yaw_link).*": [0.9, 1.1]
    # .*: [0.9, 1.1]
  perturb_body_materials:
    body_names: ".*ankle_roll_link"
    static_friction_range: [0.3, 4.0]
    dynamic_friction_range: [0.3, 4.0]
    restitution_range: [0.0, 0.2]
  motor_params_implicit:
    stiffness_range:
      .*: [0.8, 1.1]
    damping_range:
      .*: [0.8, 1.1]
    armature_range:
      .*: [0.0, 0.01]
  reset_joint_states_uniform:
    pos_ranges:
      .*: [-0.1, 0.1]
    rel: true

termination:
  # dummy: {}
  root_deviation: {max_distance: 0.4}
  root_rot_deviation: {max_theta: 30}
  track_kp_error: {max_distance: 0.5, 
                  body_names: [ left_wrist_yaw_link, right_wrist_yaw_link,
                                left_ankle_roll_link, right_ankle_roll_link]}

```

Observations are grouped by keys and the observation of the same group is concatenated.

Rewards are grouped by keys and the rewards of the same group is summed up, excluding those marked with `enabled=false`. However, rewards with `enabled=false` will still be computed and logged as metrics for debugging purposes.

### Training

Examples:

```bash
python test_env.py task=G1/motion algo=ppo
# hydra command-line overrides
python test_env.py task=G1/motion algo=ppo algo.entropy_coef=0.002 total_frames=200_000_000 wandb.mode=disabled
# finetuning
python test_env.py task=G1/motion algo=ppo checkpoint_path=${local_checkpoint_path}
python test_env.py task=G1/motion algo=ppo checkpoint_path=run:${wandb_run_path}
# multi-GPU training
export OMP_NUM_THREADS=4 # a number greater than 1
torchrun --nnodes=1 --nproc-per-node=4 ...
```

### Evaluation and Visualization

Examples:

```bash
# play the policy
python play.py task=G1/motion algo=ppo task.num_envs=1 task.command.dataset=sfu checkpoint_path=${local_checkpoint_path}
# mujoco sim2sim verification, requires MJCF assets to be specified
python play_mujoco.py task=...
# export to onnx for deployment
python play.py task=... export_policy=true
# record video
python eval.py task=... task.command.eval_id=0 eval_render=true headless=false
# evaluation for success rate and mean per keypoint error
# currently 89.24% success rate and 32.12 mean per keypoint error on 23 datasets
python eval.py task=... task.command.eval_id=[0, num_envs] # modify in eval.yaml
```

## Development Guide

### Task Definition

All components, including action, command, observation, reward, termination condition are defined by subclassing the base class. The base classes have a series of callbacks that will be called at each environment step:

```python
class Observation:
    def __init__(self, env):
        self.env: _Env = env

    @property
    def num_envs(self):
        return self.env.num_envs
    
    @property
    def device(self):
        return self.env.device

    @abc.abstractmethod
    def compute(self) -> torch.Tensor:
        raise NotImplementedError
    
    def __call__(self) ->  Tuple[torch.Tensor, torch.Tensor]:
        tensor = self.compute()
        return tensor
    
    def startup(self):
        """Called once upon initialization of the environment"""
        pass
    
    def post_step(self, substep: int):
        """Called after each physics substep"""
        pass

    def update(self):
        """Called after all physics substeps are completed"""
        pass

    def reset(self, env_ids: torch.Tensor):
        """Called after episode termination"""

    def debug_draw(self):
        """Called at each step **after** simulation, if GUI is enabled"""
        pass
```

The stepping logic is defined in `active_adaptation.envs.base._Env.step`.

The data tensordict's content looks like:

```
TensorDict(
    fields={
        action: Tensor(shape=torch.Size([4096, 32, 12]), device=cuda:0, dtype=torch.float32, is_shared=True),
        done: Tensor(shape=torch.Size([4096, 32, 1]), device=cuda:0, dtype=torch.bool, is_shared=True),
        is_init: Tensor(shape=torch.Size([4096, 32, 1]), device=cuda:0, dtype=torch.bool, is_shared=True),
        next: TensorDict(
            fields={
                discount: Tensor(shape=torch.Size([4096, 32, 1]), device=cuda:0, dtype=torch.float32, is_shared=True),
                done: Tensor(shape=torch.Size([4096, 32, 1]), device=cuda:0, dtype=torch.bool, is_shared=True),
                is_init: Tensor(shape=torch.Size([4096, 32, 1]), device=cuda:0, dtype=torch.bool, is_shared=True),
                policy: Tensor(shape=torch.Size([4096, 32, 100]), device=cuda:0, dtype=torch.float32, is_shared=True),
                stats: TensorDict(
                    ...),
                step_count: Tensor(shape=torch.Size([4096, 32, 1]), device=cuda:0, dtype=torch.int64, is_shared=True),
                terminated: Tensor(shape=torch.Size([4096, 32, 1]), device=cuda:0, dtype=torch.bool, is_shared=True),
                truncated: Tensor(shape=torch.Size([4096, 32, 1]), device=cuda:0, dtype=torch.bool, is_shared=True)},
            batch_size=torch.Size([4096, 32]),
            device=cuda:0,
            is_shared=True),
        policy: Tensor(shape=torch.Size([4096, 32, 100]), device=cuda:0, dtype=torch.float32, is_shared=True),
        step_count: Tensor(shape=torch.Size([4096, 32, 1]), device=cuda:0, dtype=torch.int64, is_shared=True),
        terminated: Tensor(shape=torch.Size([4096, 32, 1]), device=cuda:0, dtype=torch.bool, is_shared=True),
        truncated: Tensor(shape=torch.Size([4096, 32, 1]), device=cuda:0, dtype=torch.bool, is_shared=True)},
    batch_size=torch.Size([4096, 32]),
    device=cuda:0,
    is_shared=True)
```