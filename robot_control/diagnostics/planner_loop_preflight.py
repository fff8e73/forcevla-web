"""Current observation and twenty-point QP validation, without robot commands."""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
from robot_control.planner_loop import profile,adapt,prepare_waypoints
from robot_control.closed_loop_observation import FileObservationSource,ObservationPending
from robot_control.native_qp import NativeQP
from robot_control.common import ROOT,save_json

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
out=args.output;out.mkdir(exist_ok=False)
c=profile({});source=FileObservationSource('/dev/shm/forcevla-web/observation.npz',c)
barrier=SimpleNamespace(unix=time.time(),monotonic=time.monotonic());end=time.monotonic()+15
while True:
    try:o=source.read_after(barrier,None);break
    except ObservationPending:
        if time.monotonic()>end:raise
        time.sleep(.05)
np.savez_compressed(out/'observation.npz',state=o.state,joints=o.joints,metadata=json.dumps(o.metadata),**o.images)
images=list(o.images.values());Image.fromarray(np.vstack([np.hstack(images[:2]),np.hstack(images[2:])])).save(out/'observation.png')
sys.path.insert(0,'/home/user/forcevla-web')
from forcevla_web.policy import RemotePolicy
web=json.loads(Path('/home/user/forcevla-web/config/local.json').read_text());web['rpc_timeout_s']=15
policy=RemotePolicy(web)
try:prediction=policy.predict(o.request(web['default_prompt'],0),'planner-loop-preflight-'+str(time.time_ns()))
finally:policy.close()
np.savez_compressed(out/'prediction.npz',**prediction)
a,correction=adapt(prediction['actions']);limits=json.loads((ROOT/'config/planning.json').read_text());solver=NativeQP(limits['expected_qp_parameters'],limits)
start=time.monotonic();q=prepare_waypoints(solver,a,o,out/'plan')
save_json(out/'report.json',dict(status='prepared_20_no_commands',snapshot_id=o.id,steps=len(q),prepare_s=time.monotonic()-start,gripper_correction=correction,max_gripper_targets=a[:,[6,13]].max(axis=0).tolist(),max_joint_displacement=np.max(np.abs(q-o.joints)),hardware_commands_sent=0))
print((out/'report.json').read_text())
