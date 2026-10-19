"""Browser parameter fixture with synthetic observations and NO execution dispatch."""
from pathlib import Path

import uvicorn

from forcevla_web.app import create_app
from forcevla_web.manager import Conflict, Manager
from forcevla_web.observation import DemoSource
from forcevla_web.policy import DemoPolicy
from forcevla_web.store import Store

root = Path('artifacts/configurable-loop-20261015/browser-fixture').resolve()
root.mkdir(parents=True, exist_ok=True)
config = dict(source='live', allowed_hosts=['127.0.0.1', 'localhost'],
              max_age_s=.75, max_skew_s=.2, max_source_age_s=1, disconnect_grace_s=10,
              data_dir=str(root), quota_mb=100, default_prompt='Synthetic parameter test',
              closed_loop=dict(enabled=True, planner_enabled=True,
                               integration_root='/home/user/forcevla-web/robot_control'))
manager = Manager(config, DemoSource(), DemoPolicy(), Store(root, 100))
app = create_app(config, manager)


async def reject_dispatch(*args, **kwargs):
    raise Conflict('Test fixture: hardware dispatch prohibited')


app.state.closed_loop.start = reject_dispatch
app.state.home.start = reject_dispatch
app.state.gripper_restart.start = reject_dispatch
uvicorn.run(app, host='127.0.0.1', port=8105, access_log=False)
