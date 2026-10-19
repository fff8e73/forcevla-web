"""Read-only Stage B endpoint/target audit. No control publishers or service calls."""
import json
from pathlib import Path
import time
import urllib.request

import rclpy
from marvin_msgs.msg import Jointcmd, JointcmdArm, Jointfeedback
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import Int32, Int16MultiArray


def main():
    rclpy.init(domain_id=24)
    node = Node('forcevla_stage_b_readonly', enable_rosout=False, start_parameter_services=False)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    samples = {}
    counts = {}
    def callback(key):
        def receive(msg):
            samples[key] = (msg, time.monotonic())
            counts[key] = counts.get(key, 0)+1
        return receive
    sensor = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
    latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE)
    specs = [(Jointfeedback, '/tj/info/joint_feedback', 'joint_feedback', sensor),
             (Int32, '/tj/control/input_mode', 'input_mode', latched),
             (Int16MultiArray, '/tj/info/arm_state', 'arm_state', sensor),
             (Int16MultiArray, '/tj/info/robot_state', 'robot_state', sensor)]
    specs += [(JointcmdArm, '/tj/control/joint_cmd_'+s, 'target_'+s, sensor) for s in ('A','B')]
    specs += [(Jointcmd, '/tj/control/joint_cmd_plan_'+side, 'planner_'+side, sensor) for side in ('A','B')]
    subscriptions = [node.create_subscription(kind, topic, callback(key), qos) for kind,topic,key,qos in specs]
    report = {'commands_sent':0, 'http':{}, 'received':{}, 'topics':{}, 'services':[], 'started_unix': time.time(), 'sample_duration_s': 4}
    try:
        end=time.monotonic()+4
        while time.monotonic()<end:
            executor.spin_once(timeout_sec=.05)
        for topic, types in node.get_topic_names_and_types():
            if topic.startswith('/tj/') and any(x in topic for x in ('target','cmd','state','feedback','input')):
                report['topics'][topic] = dict(types=types, publishers=[dict(node=p.node_namespace+'/'+p.node_name,
                   reliability=str(p.qos_profile.reliability), durability=str(p.qos_profile.durability))
                   for p in node.get_publishers_info_by_topic(topic)])
        report['services']=[(name,types) for name,types in node.get_service_names_and_types()
                            if name.startswith('/tj/') and any(s in name.lower() for s in ('target','joint','hold','stop','state','input','command'))]
        for key,(msg,received) in samples.items():
            fields = {field:list(getattr(msg,field)) for field in ['positions','arm_positions','arm_velocities'] if hasattr(msg,field)}
            if hasattr(msg,'data'):
                fields['data']=int(msg.data) if key=='input_mode' else list(msg.data)
            if hasattr(msg,'header'):
                fields['source_age_s']=time.time()-(msg.header.stamp.sec+msg.header.stamp.nanosec/1e9)
            report['received'][key]=dict(age_s=time.monotonic()-received,count=counts[key],**fields)
        for key,path in [('robot','robot/status'),('gripper','gripper/web/status'),('replay','replay/status'),('home','robot/home/status')]:
            with urllib.request.urlopen('http://127.0.0.1:8080/api/v1/'+path,timeout=2) as response:
                data=json.load(response)
            if key=='gripper':
                data.get('data',{}).pop('session_id',None)
            report['http'][key]=data
        path=Path('/home/user/forcevla-web/data/integration-runs/stage-b-live-preflight') / ('readonly-'+time.strftime('%Y%m%d-%H%M%S')+'.json')
        report['status']='blocked_missing_joint_targets' if any(k not in samples for k in ('target_A','target_B')) else 'readonly_targets_observed_not_execution_acceptance'
        path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
        print(json.dumps({'received':report['received'],'services':report['services'],
                          'robot':report['http']['robot'], 'report':str(path)},ensure_ascii=False))
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__=='__main__':
    main()
