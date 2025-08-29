import jax
import jax.numpy as jnp
import mujoco

from loco_mujoco.environments import UnitreeH1v2
from loco_mujoco.trajectory import Trajectory, TrajectoryInfo, TrajectoryModel, TrajectoryData, TrajectoryHandler

# create the environment
env = UnitreeH1v2(init_state_type="DefaultInitialStateHandler")

# reset the env
key = jax.random.PRNGKey(0)
env.reset(key)

# get the model and data of the environment
model = env.get_model()
data = env.get_data()

fps = 30.0
traj_handler = TrajectoryHandler(
    model=model,
    traj_path="./datasets/DefaultDatasets/UnitreeH1v2/balance.npz",
    # traj_path="./datasets/Lafan1/UnitreeH1v2/dance1_subject1.npz",
    control_dt=1 / fps
)

qpos = traj_handler.traj.data.qpos      # (T, 28)
qvel = traj_handler.traj.data.qvel      # (T, 27)

# create a trajectory info -- this stores basic information about the trajectory
njnt = model.njnt
jnt_type = model.jnt_type
jnt_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(njnt)]


traj_info = TrajectoryInfo(jnt_names, model=TrajectoryModel(njnt, jnp.array(jnt_type)), frequency=fps)

# create a trajectory data -- this stores the actual trajectory data
traj_data = TrajectoryData(jnp.array(qpos), jnp.array(qvel), split_points=jnp.array([0, len(qpos)]))

# combine them to a trajectory
traj = Trajectory(traj_info, traj_data)

# example: save the trajectory
#traj.save("trajectory.npz")
#traj = Trajectory.load("trajectory.npz")

# add the trajectory to the environment
env.load_trajectory(traj)

# replay the trajectory
env.play_trajectory(n_steps_per_episode=len(qpos))