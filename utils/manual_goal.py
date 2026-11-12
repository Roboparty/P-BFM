"""Manual selection of constant goal blocks in authored reward/goal contexts."""
import json
from pathlib import Path
import numpy as np
class ManualGoalSequence:
    def __init__(self, schedule_path, context_path):
        self.entries=json.loads(Path(schedule_path).read_text())
        if not self.entries:raise ValueError('Empty goal schedule')
        self.frames=[];self.index=0
        with np.load(context_path,allow_pickle=False) as d:
            if str(d['context_contract'])!='reward_goal_yaw_command_v1':raise ValueError('Wrong manual goal contract')
            dt=float(d['dt']);last=0
            for entry in self.entries:
                start=round(float(entry['start_s'])/dt);end=round(float(entry['end_s'])/dt)
                if start!=last or end<=start or end>len(d['z']):raise ValueError('Invalid goal boundaries')
                for key in ['z','heading_relative_xy','heading_valid','velocity_command_yaw']:
                    a=d[key][start:end]
                    if not np.all(a==a[0]):raise ValueError(f'Goal segment is not constant: {key}')
                self.frames.append(start);last=end
            if last!=len(d['z']):raise ValueError('Incomplete goal schedule')
    @property
    def frame(self):return self.frames[self.index]
    def next(self):
        previous=self.index;self.index=min(self.index+1,len(self.frames)-1)
        return self.index!=previous
