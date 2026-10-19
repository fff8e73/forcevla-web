"""Explicit stationary measured-target probe; --live authorizes real fixed targets."""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

from robot_control.command_bridge import ControlFault
from robot_control.common import ROOT, new_run, save_json
from robot_control.joint_source_guard import JointEnvelope
from robot_control.measured_hold import MeasuredHoldBackend
from robot_control.native_qp import NativeQP
from robot_control.ros_execution_transport import RosExecutionTransport


def snapshot(f):
    return dict(received_at=f.received_at,joints=f.joints.tolist(),velocities=f.velocities.tolist(),
                joint_target=f.joint_target.tolist() if np.isfinite(f.joint_target).all() else None,
                target_received_at=f.target_received_at,input=f.selected_input,mode=f.mode,
                gripper_target=f.gripper['applied_target'],gripper_actual=f.gripper['actual'],
                gripper_velocity=f.gripper['motor_velocities'])


def preflight(backend, duration=.6):
    rows=[]
    end=time.monotonic()+duration
    while time.monotonic()<end:
        f=backend.checked_feedback()
        if f.selected_input not in (0,2) or f.gripper.get('reserved') is not False:
            raise ControlFault('preflight_source_or_ownership')
        if np.any(np.abs(f.gripper['motor_velocities'])>backend.gripper_stop_speed):
            raise ControlFault('preflight_gripper_speed:'+str(f.gripper['motor_velocities']))
        calibration=np.asarray(f.gripper['calibration'])
        error=np.abs(np.asarray(f.gripper['actual'])-f.gripper['applied_target'])*(calibration[:,1]-calibration[:,0])
        if np.any(error>.05):raise ControlFault('preflight_gripper_tracking')
        rows.append(snapshot(f))
        time.sleep(.02)
    if np.max(np.ptp(np.array([r['joints'] for r in rows]),axis=0))>.001:
        raise ControlFault('preflight_not_stationary')
    return rows


def run_cycle(backend,transport,output):
    rows=preflight(backend)
    save_json(output/'preflight.json',rows)
    token=backend.acquire('measured_stationary_probe')
    seed=backend.last_q.copy()
    observations=[]
    start=time.monotonic()
    seq=0
    try:
        while time.monotonic()-start<4.2:
            time.sleep(.05)
            backend.heartbeat(token)
            now=time.monotonic()
            backend.write(JointEnvelope(token,seq,now,backend.seed_observed_at,seed))
            seq+=1
            f=backend.checked_feedback(owned=True)
            observations.append(snapshot(f))
    finally:
        save_json(output/'observations.json',observations)
    backend.trip('measured_hold_cycle_complete')
    deadline=time.monotonic()+3
    while backend.state=='STOPPING' and time.monotonic()<deadline:
        time.sleep(.02)
        backend.poll()
    save_json(output/'observations.json',observations)
    if backend.state!='HOLD_CONFIRMED':
        raise ControlFault('measured_hold_unconfirmed:'+str(backend.stop_report))
    backend.release(token)
    f=transport.feedback()
    if f.selected_input!=0 or f.gripper.get('reserved') is not False:
        raise ControlFault('final_release_readback_failed')
    errors=np.abs(np.array([r['joints'] for r in observations])-seed)
    return dict(status='passed_measured_joint_fixed_hold',final_state=backend.state,
                seed=seed.tolist(),target_policy='new_fixed_measured_target',
                observations=len(observations),submitted_pairs=seq,
                max_observed_joint_offset_rad=float(errors.max()),
                max_observed_joint_speed_rad_s=float(np.abs([r['velocities'] for r in observations]).max()),
                stop_report=backend.stop_report,final_feedback=snapshot(f),
                production_publications=transport.publications if transport.production else 0,
                gripper_monitor_limit_rad_s=backend.gripper_stop_speed,
                max_gripper_speed_rad_s=float(np.abs([r['gripper_velocity'] for r in observations]).max()),
                original_target_preserved=False,moving_stop_verified=False)


def read_status():
    result={}
    for name,path in [('robot','robot/status'),('home','robot/home/status'),('replay','replay/status')]:
        with urllib.request.urlopen('http://127.0.0.1:8080/api/v1/'+path,timeout=1) as response:
            result[name]=json.load(response)['data']
    if result['home'].get('busy') or result['replay'].get('is_playing') or result['replay'].get('is_paused'):
        raise ControlFault('planner_or_replay_active')
    r=result['robot']
    if not r.get('control_ready') or r.get('control_error') or r.get('current_mode')!=1 or r.get('current_input_mode') not in (0,2):
        raise ControlFault('robot_not_ready_for_position_hold')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--live',action='store_true')
    p.add_argument('--gripper-monitor-limit',type=float,default=None)
    args=p.parse_args()
    out=new_run(args.output,'measured-hold')
    report=dict(status='failed',hardware_commands_sent=0,live_requested=args.live,
                gripper_monitor_limit_override=args.gripper_monitor_limit,
                driver_claim_speed_limit_rad_s=.02)
    backend=transport=None
    try:
        report['http_preflight']=read_status()
        # Native model is loaded only for validated joint bounds, no solve or SDK.
        limits=json.loads((ROOT/'config/planning.json').read_text())
        solver=NativeQP(limits['expected_qp_parameters'],limits)
        transport=RosExecutionTransport(domain=24,namespace='/tj',allow_missing_targets=True)
        backend=MeasuredHoldBackend(transport,solver.bounds,solver.velocity,out/'execution.json',
                                    prediction_age_s=10.,stop_timeout_s=2.5,
                                    gripper_stop_speed_rad_s=.03,
                                    gripper_monitor_limit_rad_s=args.gripper_monitor_limit)
        deadline=time.monotonic()+5
        while True:
            try:
                backend.checked_feedback()
                break
            except ControlFault:
                if time.monotonic()>deadline:raise
                time.sleep(.05)
        save_json(out/'preflight.json',preflight(backend))
        if not args.live:
            report['status']='preflight_passed_no_commands'
        else:
            report['http_before_acquire']=read_status()
            backend.start_watchdog()
            report.update(run_cycle(backend,transport,out))
    except Exception as exc:
        report['error']=repr(exc)
        if backend and backend.state in ('ACQUIRING','ACTIVE'):
            backend.trip('probe_exception:'+str(exc))
        if backend:
            deadline=time.monotonic()+3
            while backend.state=='STOPPING' and time.monotonic()<deadline:
                time.sleep(.02)
                backend.poll()
            report['final_state']=backend.state
            report['stop_report']=backend.stop_report
    finally:
        if transport:
            report['hardware_commands_sent']=transport.publications
            report['operation_counts']=transport.operation_counts
        if backend:backend.close()
        elif transport:transport.close()
        save_json(out/'report.json',report)
    print(json.dumps(report,ensure_ascii=False))
    return 0 if report['status'].startswith(('passed','preflight_passed')) else 2


if __name__=='__main__':
    raise SystemExit(main())
