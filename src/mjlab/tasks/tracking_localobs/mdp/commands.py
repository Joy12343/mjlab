from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from mjlab.tasks.tracking.mdp.commands import MotionCommand
from mjlab.tasks.tracking.mdp.commands import MotionCommandCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  subtract_frame_transforms,
)

class MotionLoader:
  def __init__(
    self, motion_file: str, body_indexes: torch.Tensor, device: str = "cpu"
  ) -> None:
    data = np.load(motion_file)
    self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
    self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
    self._body_pos_w = torch.tensor(
      data["body_pos_w"], dtype=torch.float32, device=device
    )
    self._body_quat_w = torch.tensor(
      data["body_quat_w"], dtype=torch.float32, device=device
    )
    self._body_lin_vel_w = torch.tensor(
      data["body_lin_vel_w"], dtype=torch.float32, device=device
    )
    self._body_ang_vel_w = torch.tensor(
      data["body_ang_vel_w"], dtype=torch.float32, device=device
    )
    self._body_indexes = body_indexes
    self.time_step_total = self.joint_pos.shape[0]

  @property
  def body_pos_w(self) -> torch.Tensor:
    return self._body_pos_w[:, self._body_indexes]

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self._body_quat_w[:, self._body_indexes]

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self._body_lin_vel_w[:, self._body_indexes]

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self._body_ang_vel_w[:, self._body_indexes]


class MotionCommandLocal(MotionCommand):

  @property
  def command(self) -> torch.Tensor:
      num_bodies = len(self.cfg.body_names)

      # body position relative to anchor frame
      pos_b, _ = subtract_frame_transforms(
          self.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
          self.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
          self.body_pos_w,
          self.body_quat_w,
      )
      pos_b = pos_b.reshape(self.num_envs, -1)

      # transform body linear velocities into anchor frame
      rotm_T = matrix_from_quat(self.robot_anchor_quat_w).transpose(-1, -2)

      lin_vel_b = torch.einsum("bij,bmj->bmi", rotm_T, self.body_lin_vel_w)
      lin_vel_b = lin_vel_b.reshape(self.num_envs, -1)

      return torch.cat([pos_b, lin_vel_b], dim=1)

@dataclass(kw_only=True)
class MotionCommandLocalCfg(MotionCommandCfg):
    class_type: type = MotionCommandLocal
