"""Synthetic gripper restart fixture; no real control endpoint can be dispatched."""
import time
from pathlib import Path

import uvicorn

from test_gripper_restart import RestartBackend
from test_home import Backend, Source
from forcevla_web.app import create_app
from forcevla_web.manager import Conflict, Manager
from forcevla_web.policy import DemoPolicy
from forcevla_web.store import Store

root=Path('artifacts/gripper-restart-ui-20261015/fixture').resolve();root.mkdir(parents=True,exist_ok=True)
config=dict(source='live',allowed_hosts=['127.0.0.1','localhost'],max_age_s=.75,max_skew_s=.2,max_source_age_s=1,
            disconnect_grace_s=10,data_dir=str(root),quota_mb=100,default_prompt='Synthetic gripper reset test',
            closed_loop=dict(enabled=True,planner_enabled=True,integration_root='/home/user/forcevla-web/robot_control'))


class SlowRestart(RestartBackend):
    def reset_initial(self):
        time.sleep(.8)
        return super().reset_initial()


backend=SlowRestart()
manager=Manager(config,Source(),DemoPolicy(),Store(root,100))
app=create_app(config,manager,home_backend=Backend(),gripper_restart_backend=backend)


async def reject_real_dispatch(*args,**kwargs):
    raise Conflict('Test fixture cannot execute robot motion')


app.state.closed_loop.start=reject_real_dispatch
app.state.control_validation.start=reject_real_dispatch
uvicorn.run(app,host='127.0.0.1',port=8108,access_log=False)
