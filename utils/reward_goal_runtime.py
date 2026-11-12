"""Explicit reward/goal current-yaw velocity contract; never heading-rotate it."""
import numpy as np
CONTRACT = 'reward_goal_yaw_command_v1'
class RewardGoalVelocityRuntime:
    def __init__(self, data):
        if str(data['context_contract']) != CONTRACT:
            raise ValueError('Wrong reward/goal context contract')
        self.commands = np.asarray(data['velocity_command_yaw'], dtype=np.float32).copy()
        n = len(data['z'])
        if self.commands.shape != (n,3) or not np.isfinite(self.commands).all():
            raise ValueError('Invalid current-yaw velocity commands')
        if not np.isin(self.commands[:,2], [0,1]).all():
            raise ValueError('Velocity valid must be 0 or 1')
        if np.any(self.commands[self.commands[:,2]==0,:2] != 0):
            raise ValueError('Invalid velocity task must have zero command')
    def command(self, root_heading, frame, active, start_key=None):
        # Same runtime interface as tracking, deliberately no coordinate rotation.
        if not active:
            return np.zeros((1,3),np.float32)
        i=int(frame)
        if not 0 <= i < len(self.commands):
            raise IndexError('Reward/goal context ended')
        return self.commands[i:i+1].copy()
