import numpy as np

from utils.strings import resolve_matching_names_values
from utils.strings import unitree_joint_names


class CommandSender:
    def __init__(self, robot_config, policy_config):
        if robot_config.get("ROBOT_TYPE") != "g1_real":
            raise ValueError("CommandSender requires ROBOT_TYPE=g1_real")
        self.dry_run = bool(robot_config.get("_UFO_DRY_RUN", False))
        self.closed = False
        if not self.dry_run:
            self.robot = robot_config["robot"]

        # init robot and kp kd
        self._kp_level = 1.0

        self.policy_config = policy_config
        joint_kp_dict = self.policy_config["joint_kp"]
        joint_indices, joint_names, joint_kp = resolve_matching_names_values(
            joint_kp_dict,
            unitree_joint_names,
            preserve_order=True,
            strict=False,
        )
        self.joint_kp_unitree_default = np.zeros(len(unitree_joint_names))
        self.joint_kp_unitree_default[joint_indices] = joint_kp
        self.joint_kp_unitree = self.joint_kp_unitree_default.copy()

        joint_kd_dict = self.policy_config["joint_kd"]
        joint_indices, joint_names, joint_kd = resolve_matching_names_values(
            joint_kd_dict,
            unitree_joint_names,
            preserve_order=True,
            strict=False,
        )
        self.joint_kd_unitree = np.zeros(len(unitree_joint_names))
        self.joint_kd_unitree[joint_indices] = joint_kd

        joint_names_isaac = self.policy_config["isaac_joint_names"]
        self.joint_indices_unitree = [unitree_joint_names.index(name) for name in joint_names_isaac]

    @property
    def kp_level(self):
        return self._kp_level

    @kp_level.setter
    def kp_level(self, value):
        self._kp_level = value
        self.joint_kp_unitree[:] = self.joint_kp_unitree_default * self._kp_level

    def send_command(self, cmd_q, cmd_dq, cmd_tau):
        if self.closed:
            raise RuntimeError("command sender is closed")
        if self.dry_run:
            return
        cmd = self.robot.create_zero_command()

        # kp_level setter maintains joint_kp_unitree; kd remains constant.
        kp_scaled = self.joint_kp_unitree
        kd_scaled = self.joint_kd_unitree

        q_target = list(cmd.q_target)
        dq_target = list(cmd.dq_target)
        tau_ff = list(cmd.tau_ff)
        kp = list(cmd.kp)
        kd = list(cmd.kd)
        for i_policy, idx_unitree in enumerate(self.joint_indices_unitree):
            q_target[idx_unitree] = float(cmd_q[i_policy])
            dq_target[idx_unitree] = float(cmd_dq[i_policy])
            tau_ff[idx_unitree] = float(cmd_tau[i_policy])
            kp[idx_unitree] = float(kp_scaled[idx_unitree])
            kd[idx_unitree] = float(kd_scaled[idx_unitree])

        cmd.q_target = q_target
        cmd.dq_target = dq_target
        cmd.tau_ff = tau_ff
        cmd.kp = kp
        cmd.kd = kd

        self.robot.write_low_command(cmd)


    def close(self) -> None:
        """Release command transports; safe to call repeatedly."""
        if self.closed:
            return
        self.closed = True
        if hasattr(self, "robot"):
            self.robot = None
