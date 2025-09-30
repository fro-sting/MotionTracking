import os
import json
from active_adaptation.envs.mujoco import MJArticulationCfg
import active_adaptation.utils.symmetry as symmetry_utils

ROBOTS = {}

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DESCRIPTION_DIR = os.path.join(BASE_DIR, "g1_description")  

ROBOTS["g1_29dof"] = MJArticulationCfg(
    mjcf_path=os.path.join(DESCRIPTION_DIR, "g1_29dof_rev_1_0.xml"),
    **json.load(open(os.path.join(DESCRIPTION_DIR, "g1_29dofjson"))),
)