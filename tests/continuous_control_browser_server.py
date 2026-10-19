"""Synthetic browser controller: records requests, never launches execution or ROS."""
import copy
import os
from pathlib import Path

import uvicorn

from forcevla_web.app import create_app
from forcevla_web.manager import Conflict, Manager
from forcevla_web.observation import DemoSource
from forcevla_web.policy import DemoPolicy
from forcevla_web.store import Store

root = Path('artifacts/continuous-control-20261015/fixture').resolve()
root.mkdir(parents=True, exist_ok=True)
config = dict(source='live', allowed_hosts=['127.0.0.1','localhost'],
              max_age_s=.75, max_skew_s=.2, max_source_age_s=1, disconnect_grace_s=10,
              data_dir=str(root), quota_mb=100, default_prompt='Synthetic control routing test',
              closed_loop=dict(enabled=True, planner_enabled=True,
                               integration_root='/home/user/forcevla-web/robot_control'))
manager = Manager(config, DemoSource(), DemoPolicy(), Store(root, 100))
app = create_app(config, manager)
original = app.state.closed_loop
base_status = original.status()


class FakeController:
    current = None
    @property
    def blocked(self):
        return self.current is not None and self.current['state'] == 'EXECUTE'
    def status(self, owner=None):
        value = copy.deepcopy(base_status)
        value.update(blocked=self.blocked, run=self.read_run('', owner) if self.current else None)
        return value
    async def start(self, owner, options):
        self.owner = owner
        self.current = dict(id='f'*32, mode=options['mode'], engine=options['engine'],
                            state='EXECUTE', cycle=1, completed_cycles=0, config=options,
                            execution=dict(phase='EXECUTE',executed_model_steps=0),
                            synthetic=True, hardware_commands_sent=0)
        return self.read_run('', owner)
    def read_run(self, run_id, owner=None):
        return dict(copy.deepcopy(self.current), is_owner=owner == self.owner)
    async def stop(self, run_id, owner):
        from forcevla_web.manager import Conflict
        if owner != self.owner: raise Conflict('Wrong owner')
        self.current.update(state='CANCELLED', reason='test_stop', stop=dict(confirmed=True))
        return self.read_run(run_id, owner)
    def runs(self, owner):
        return []
    async def close(self):
        pass


fake = FakeController()
# Routes capture the original controller instance; redirect only to the fake methods.
original.status = fake.status
original.start = fake.start
original.stop = fake.stop
original.read_run = fake.read_run
original.runs = fake.runs
original.close = fake.close
# Shared mutual-exclusion checks use the fake's blocked state.
manager.closed_loop = fake
async def reject_other_control(*args, **kwargs):
    raise Conflict('Test fixture cannot execute other hardware operations')


app.state.home.start = reject_other_control
app.state.gripper_restart.start = reject_other_control
uvicorn.run(app, host='127.0.0.1', port=int(os.environ.get('CONTROL_BROWSER_PORT','8106')), access_log=False)
