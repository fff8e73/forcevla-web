"""Stage D finite motion tests: installed mux, synthetic plant, domain231 only."""
import json
import sys
import time
from pathlib import Path

import numpy as np

from robot_control.common import save_json
from robot_control.joint_source_guard import JointEnvelope
from robot_control.prepared_joint_plan import PreparedJointPlan
from robot_control.segment_executor import SegmentExecutor, SegmentLimits


def run_motion_scenario(backend, transport, token, fault, output, trace):
    # Finish the installed mux's 3s source-switch ramp at unchanged seed before motion.
    # Reduce GIL handoff latency among this isolated process's ROS callbacks.
    # No host scheduler/priority or running production process is changed.
    sys.setswitchinterval(.001)
    seed = backend.last_q.copy()
    seq = 0
    end = time.monotonic()+3.3
    while time.monotonic()<end:
        time.sleep(.02)
        backend.heartbeat(token)
        now=time.monotonic()
        backend.write(JointEnvelope(token,seq,now,now,seed))
        seq+=1
    # A separately sampled 100 Hz synthetic curve; never retimes native QP output.
    n=20
    u=np.arange(1,n+1)/n
    curve=.001*(10*u**3-15*u**4+6*u**5)
    q=np.tile(seed,(n,1))
    q[:,0]+=curve
    q[:,7]-=curve
    plan=PreparedJointPlan('isolated-synthetic-segment',time.monotonic(),seed,q,
                           np.tile(backend.grip_target,(n,1)),.01,
                           {'scope':'synthetic_joint_curve_not_real_model_prediction'})
    np.savez_compressed(output/'segment.npz',seed=seed,joints=q,grippers=plan.grippers,dt=plan.dt)
    executor=SegmentExecutor(backend, SegmentLimits(max_lateness_s=.004))
    # Lease independent of writes. No self-renewal inside the executor.
    backend.guard.lease_s=.06 if fault=='upstream_loss' else .5
    backend.heartbeat(token)
    executor.start(plan,token)
    start=time.monotonic()
    injected=False
    next_heartbeat=start+.02
    original_publish=transport.publish_arm
    original_feedback=transport.feedback
    while executor.state=='RUNNING':
        now=time.monotonic()
        if now < executor.next_due:
            time.sleep(executor.next_due-now)
        if executor.index>=4 and not injected:
            injected=True
            if fault=='manual_abort':executor.request_abort()
            elif fault=='deadline':time.sleep(.01)
            elif fault=='right_write':
                def failed(side,positions):
                    if side=='right':raise ConnectionError('injected_right_write_failure')
                    original_publish(side,positions)
                transport.publish_arm=failed
            elif fault=='stale_feedback':
                from dataclasses import replace
                transport.feedback=lambda:replace(original_feedback(),source_age_s=10.)
        # Test supervisor is the upstream heartbeat source, separate from tick submission.
        if fault not in ('upstream_loss','stale_feedback') and time.monotonic()>=next_heartbeat:
            backend.heartbeat(token)
            next_heartbeat=time.monotonic()+.02
        executor.tick()
    if fault=='stale_feedback':
        transport.feedback=original_feedback
    motion_end=time.monotonic()
    # The executor has discarded the queue. Poll only holds, never resume motion.
    deadline=time.monotonic()+2.
    while executor.state=='STOPPING' and time.monotonic()<deadline:
        time.sleep(.02)
        executor.poll_hold()
    result=executor.report()
    save_json(output/'segment-execution.json',result)
    save_json(output/'mux-output.json',trace)
    expected={'none':'segment_complete','manual_abort':'manual_abort',
              'deadline':'segment_deadline_missed','upstream_loss':'upstream_lease_expired',
              'right_write':'injected_right_write_failure', 'stale_feedback':'stale_joint_feedback'}[fault]
    if result['reason'] != expected:
        raise RuntimeError('unexpected_segment_reason:'+str(result['reason']))
    uncertain=fault in ('right_write','stale_feedback')
    if result['state'] != ('FAULT_UNCONFIRMED' if uncertain else 'HOLD_CONFIRMED'):
        raise RuntimeError('unexpected_hold_state:'+result['state'])
    sent=result['sent_ticks']
    if fault=='none' and sent!=n or fault!='none' and not 0<sent<n:
        raise RuntimeError('invalid_segment_consumption')
    # Confirm actual mux output differs from seed; publishing alone isn't this evidence.
    moving=[r for r in trace if start<=r['time']<=motion_end+.3]
    excursions=[max(abs(np.array(r['positions'])-seed[r['side']*7:r['side']*7+7])) for r in moving]
    max_excursion=max(excursions, default=0.)
    if max_excursion<=1e-7 or max_excursion>.0011:
        raise RuntimeError('mux_motion_not_observed_or_exceeded_budget')
    if not uncertain:
        backend.release(token)
    return dict(status='passed',final_state=backend.state,segment=result,
                motion_mux_samples=len(moving),max_mux_excursion_rad=float(max_excursion),
                input_after=transport.feedback().selected_input,
                gripper_reserved=transport.gripper_status()['reserved'],
                scope='finite synthetic joint segment; installed mux; ideal plant; no physical stop proof',
                hardware_commands_sent=0, moving_stop_verified=False)


if __name__=='__main__':
    from robot_control.diagnostics.stage_b_sandbox import main
    raise SystemExit(main(stage_d=True))
