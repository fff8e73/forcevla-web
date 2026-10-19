"""Installed MoveJ planner in isolated domain 231; synthetic feedback, no driver."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import numpy as np
from robot_control.common import ROOT, save_json, sha256
from robot_control.native_qp import NativeQP


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); args.output.mkdir(exist_ok=False)
    with np.load(args.plan,allow_pickle=False) as d:
        waypoints=d['joints'];seed=d['seed']
    limits=json.loads((ROOT/'config/planning.json').read_text())
    solver=NativeQP(limits['expected_qp_parameters'],limits)
    import rclpy
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import QoSProfile,ReliabilityPolicy
    from marvin_msgs.msg import Jointfeedback,Jointcmd
    from marvin_msgs.srv import MoveJ
    namespace='/forcevla_sandbox_'+uuid.uuid4().hex[:10]
    env=os.environ.copy();env.update(ROS_DOMAIN_ID='231',ROS_LOCALHOST_ONLY='1')
    for key in ('ROS_STATIC_PEERS','ROS_AUTOMATIC_DISCOVERY_RANGE','FASTRTPS_DEFAULT_PROFILES_FILE','FASTDDS_DEFAULT_PROFILES_FILE','CYCLONEDDS_URI'):
        env.pop(key,None)
    ctx=Context();rclpy.init(args=[],context=ctx,domain_id=231)
    node=Node('synthetic_feedback',namespace=namespace,context=ctx)
    executor=SingleThreadedExecutor(context=ctx);executor.add_node(node)
    q=seed.copy(); trace=[]; counts=[0,0]
    sensor=QoSProfile(depth=100,reliability=ReliabilityPolicy.BEST_EFFORT)
    pub=node.create_publisher(Jointfeedback,namespace+'/info/joint_feedback',1)
    def receive(side):
        def cb(msg):
            q[side*7:side*7+7]=msg.positions
            counts[side]+=1
            trace.append(dict(time=time.monotonic(),side=side,joints=q.tolist()))
        return cb
    subs=[node.create_subscription(Jointcmd,namespace+'/control/joint_cmd_plan_'+s,receive(i),sensor) for i,s in enumerate(('A','B'))]
    def tick():
        m=Jointfeedback();m.header.stamp=node.get_clock().now().to_msg();m.arm_positions=q.tolist();m.arm_velocities=[0.]*14;pub.publish(m)
    timer=node.create_timer(.005,tick)
    client=node.create_client(MoveJ,namespace+'/control/movej')
    binary='/opt/kernelmind/apex/install/marvin_teleop/lib/marvin_teleop/planner_joint'
    log=(args.output/'planner.log').open('w')
    process=subprocess.Popen([binary,'--ros-args','-r','__node:=planner_joint_node','-r','__ns:='+namespace,'--params-file',
        '/opt/kernelmind/apex/install/marvin_teleop/share/marvin_teleop/config/robot_param_m6_696.yaml'],env=env,stdout=log,stderr=subprocess.STDOUT)
    report=dict(status='failed',domain=231,namespace=namespace,planner_sha256=sha256(binary),hardware_commands_sent=0,steps=[])
    try:
        deadline=time.monotonic()+8
        while not client.service_is_ready():
            executor.spin_once(timeout_sec=.02)
            if time.monotonic()>deadline: raise RuntimeError('planner_discovery_timeout')
        warmup=time.monotonic()+.5
        while time.monotonic()<warmup:executor.spin_once(timeout_sec=.02)
        previous=seed
        for i,target in enumerate(waypoints):
            request=MoveJ.Request();request.joint_values=target.tolist()
            start=time.monotonic();first=len(trace);future=client.call_async(request)
            settled=None
            while True:
                executor.spin_once(timeout_sec=.02)
                if future.done() and not future.result().success:raise RuntimeError('movej_rejected')
                if np.max(np.abs(q-target))<1e-5:
                    if settled is None:settled=time.monotonic()
                    if time.monotonic()-settled>.15:break
                else:settled=None
                if time.monotonic()-start>8:raise RuntimeError('arrival_timeout')
            points=np.array([r['joints'] for r in trace[first:]])
            if np.any(points<np.minimum(previous,target)-1e-6) or np.any(points>np.maximum(previous,target)+1e-6):
                raise RuntimeError('planner_path_outside_segment_box')
            nearest=10.
            for row in points:
                near=min(r['distance_m'] for r in solver.collisions(row));nearest=min(nearest,near)
                if near<.005:raise RuntimeError('native_planner_path_collision_margin')
            report['steps'].append(dict(step=i+1,seconds=time.monotonic()-start,samples=len(points),min_configured_distance_m=nearest,
                                        final_joint_error=float(np.max(np.abs(q-target)))))
            previous=target
            print('sandbox reached',i+1,flush=True)
        report['status']='passed_native_planner_ten_waypoints'
    finally:
        save_json(args.output/'report.json',report)
        save_json(args.output/'trajectory.json',trace)
        process.terminate()
        try:process.wait(timeout=3)
        except subprocess.TimeoutExpired:process.kill();process.wait()
        log.close();executor.shutdown();node.destroy_node();ctx.shutdown()
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
