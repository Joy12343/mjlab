"""Unitree G1 flat tracking environment configurations."""

from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import ObservationGroupCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp.commands import MotionCommandCfg
from mjlab.tasks.tracking_localobs.mdp import MotionCommandLocalCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg


def unitree_g1_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain tracking configuration."""
  cfg = make_tracking_env_cfg()

  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  cfg.scene.sensors = (self_collision_cfg,)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  assert cfg.commands is not None
  base_cmd = cfg.commands["motion"]
  assert isinstance(base_cmd, MotionCommandCfg)
  cfg.commands["motion"] = MotionCommandLocalCfg(
    motion_file=base_cmd.motion_file,
    anchor_body_name=base_cmd.anchor_body_name,
    body_names=base_cmd.body_names,
    asset_name=base_cmd.asset_name,
    resampling_time_range=base_cmd.resampling_time_range,
    pose_range=base_cmd.pose_range,
    velocity_range=base_cmd.velocity_range,
    joint_position_range=base_cmd.joint_position_range,
    adaptive_kernel_size=base_cmd.adaptive_kernel_size,
    adaptive_lambda=base_cmd.adaptive_lambda,
    adaptive_uniform_ratio=base_cmd.adaptive_uniform_ratio,
    adaptive_alpha=base_cmd.adaptive_alpha,
    sampling_mode=base_cmd.sampling_mode,
    viz=base_cmd.viz,
  )
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandLocalCfg)
  motion_cmd.anchor_body_name = "torso_link"
  motion_cmd.body_names = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
  )

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(left|right)_foot[1-7]_collision$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  )

  cfg.viewer.body_name = "torso_link"

  print("==== DEBUG: In env_cfgs.py ====")
  print("Replaced command['motion'] with:", cfg.commands["motion"])
  print("Type:", type(cfg.commands["motion"]))
  print("Class_type:", getattr(cfg.commands["motion"], "class_type", None))
  print("================================")

  # Modify observations if we don't have state estimation.
  if not has_state_estimation:
    new_policy_terms = {
      k: v
      for k, v in cfg.observations["policy"].terms.items()
      if k not in ["motion_anchor_pos_b", "base_lin_vel"]
    }
    cfg.observations["policy"] = ObservationGroupCfg(
      terms=new_policy_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )

  policy_terms = cfg.observations["policy"].terms
  policy_terms.pop("joint_pos", None)
  policy_terms.pop("joint_vel", None)

  critic_terms = cfg.observations["critic"].terms
  critic_terms.pop("joint_pos", None)
  critic_terms.pop("joint_vel", None)
  critic_terms.pop("body_pos", None)
  critic_terms.pop("body_ori", None)

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    # Disable RSI randomization.
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}

    motion_cmd.sampling_mode = "start"

  return cfg
