"""Synthetic HOME UI fixture; all control mutations remain inside a fake backend."""
import time
from pathlib import Path

import uvicorn

from test_home import Backend, Source
from forcevla_web.app import create_app
from forcevla_web.manager import Conflict, Manager
from forcevla_web.policy import DemoPolicy
from forcevla_web.store import Store

root=Path('artifacts/home-ui-20261015/fixture').resolve();root.mkdir(parents=True,exist_ok=True)
config=dict(source='live',allowed_hosts=['127.0.0.1','localhost'],max_age_s=.75,max_skew_s=.2,max_source_age_s=1,
            disconnect_grace_s=10,data_dir=str(root),quota_mb=100,default_prompt='Synthetic HOME browser test',
            closed_loop=dict(enabled=True,planner_enabled=True,integration_root='/home/user/forcevla-web/robot_control'))


class SlowBackend(Backend):
    def home(self):
        time.sleep(1.)
        return super().home()


backend=SlowBackend()
manager=Manager(config,Source(),DemoPolicy(),Store(root,100))
app=create_app(config,manager,home_backend=backend)


async def reject_real_dispatch(*args,**kwargs):
    raise Conflict('Test fixture cannot execute robot motion')


app.state.closed_loop.start=reject_real_dispatch
app.state.control_validation.start=reject_real_dispatch
app.state.gripper_restart.start=reject_real_dispatch
uvicorn.run(app,host='127.0.0.1',port=8107,access_log=False)
