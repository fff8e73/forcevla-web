import copy
import threading
import time

import pytest
from fastapi.testclient import TestClient

from test_home import Backend, Source, wait
from forcevla_web.app import create_app
from forcevla_web.gripper_restart import ApexGripperRestartClient, GripperRestartController
from forcevla_web.manager import Manager
from forcevla_web.policy import DemoPolicy
from forcevla_web.store import Store


class RestartBackend(Backend):
    def __init__(self):
        super().__init__()
        self.gripper.update(available=True,healthy=True,errors=[1,1],
                            actual=[-.0211,.79144],applied_target=[.00256,1.],
                            motor_feedback_age=[.001,.001])
        self.stale=False
        self.bad_after=False
    def get(self,name):
        data=super().get(name)
        if name=='gripper':
            data.update(status_age_sec=.01,sample_time_ns=time.time_ns()-2_000_000)
            if self.stale:data['sample_time_ns']-=10_000_000_000
        return data
    def reset_initial(self):
        self.calls.append(('initial_target',[0.0,0.0]));self.entered.set()
        assert self.release.wait(5)
        if self.fail:raise TimeoutError(self.fail)
        if self.bad_after:self.gripper['healthy']=False;self.gripper['errors']=[0,0]
        self.gripper['applied_target']=[0.0,0.0]
        return {'status':'initial_targets_confirmed','target':[0.0,0.0],'goal_readback_confirmed':True,'hardware_commands_sent':2,'physical_arrival_verified':False}


@pytest.fixture
def system(tmp_path):
    config=dict(source='live',allowed_hosts=['testserver'],max_age_s=.75,max_skew_s=.2,
                max_source_age_s=1,disconnect_grace_s=10,data_dir=str(tmp_path),quota_mb=100,
                closed_loop=dict(enabled=True,integration_root='/home/user/forcevla-web/robot_control'))
    source=Source();manager=Manager(config,source,DemoPolicy(),Store(tmp_path,100))
    backend=RestartBackend();home_backend=Backend()
    app=create_app(config,manager,home_backend=home_backend,gripper_restart_backend=backend)
    manager.gripper_restart.verify_timeout=.3
    with TestClient(app) as client:
        session=client.get('/api/v1/session').json()
        client.headers.update({'Origin':'http://testserver','X-CSRF-Token':session['csrf_token']})
        wait(lambda:manager.snapshot is not None)
        try:yield client,manager,backend,source,config,home_backend
        finally:backend.release.set()


def start(client,command='restart-test-command'):
    return client.post('/api/v1/gripper-restart/start',json={'command_id':command})


def final(client):
    result=client.get('/api/v1/gripper-restart/status').json()
    return result if result['run'] and result['run']['state'] in ('COMPLETED','FAILED','UNCERTAIN') else None


def test_child_sends_explicit_initial_goals_without_restart_http(monkeypatch):
    from types import SimpleNamespace
    import json
    backend=ApexGripperRestartClient();calls=[]
    def process(args,**kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0,stderr='',stdout=json.dumps(dict(
            status='initial_targets_confirmed',target=[0.0,0.0],goal_readback_confirmed=True)))
    monkeypatch.setattr('forcevla_web.gripper_restart.subprocess.run',process)
    backend.request=lambda *args,**kwargs:pytest.fail('No reset/disable/enable HTTP operation allowed')
    assert backend.reset_initial()['goal_readback_confirmed']
    assert calls[0][0].endswith('/scripts/run_gripper_initial.sh')
    assert calls[0][1:]==['--execute']
    monkeypatch.setattr('forcevla_web.gripper_restart.subprocess.run',lambda *a,**kw:SimpleNamespace(
        returncode=2,stdout='{"status":"failed","published":[true,false]}',stderr=''))
    with pytest.raises(RuntimeError,match='初始目标下发未确认'):backend.reset_initial()


def test_readonly_load_then_initial_goal_confirmed_without_arrival_gate(system):
    client,manager,backend,*_=system
    assert client.get('/api/v1/status').json()['gripper_restart']['enabled']
    assert backend.calls==[]
    assert start(client).status_code==202
    result=wait(lambda:final(client))
    assert result['run']['state']=='COMPLETED',result
    assert result['run']['after']['applied_target']==[0.0,0.0]
    assert result['run']['physical_arrival_verified'] is False
    assert result['run']['operation']=='initial_target_v1'
    assert result['run']['after']['actual']==[-.0211,.79144]
    assert backend.calls==[('initial_target',[0.0,0.0])]
    assert start(client).json()['run']['id']==result['run']['id']
    assert backend.calls==[('initial_target',[0.0,0.0])]


def test_restart_excludes_home_inference_loop_and_validation(system):
    client,manager,backend,source,config,home_backend=system
    backend.release.clear();r=start(client);assert r.status_code==202
    assert backend.entered.wait(2)
    assert start(client,'another-restart').status_code==409
    assert start(client).json()['run']['id']==r.json()['run']['id']
    for path,body in [('/api/v1/home/start',dict(command_id='home-during-restart')),
                      ('/api/v1/runs',dict(kind='single',prompt='test',command_id='infer-during-restart')),
                      ('/api/v1/closed-loop/runs',dict(mode='live',engine='movej',prompt='test',command_id='loop-during-restart')),
                      ('/api/v1/control-validation/start',dict(kind='stationary_cycle',command_id='cv-during-restart',safety_ack=True))]:
        assert client.post(path,json=body).status_code==409
    assert home_backend.calls==[]
    backend.release.set();assert wait(lambda:final(client))['run']['state']=='COMPLETED'


@pytest.mark.parametrize('condition',['home','owned','replay','teleop','moving'])
def test_busy_device_refused_before_reset(system,condition):
    client,manager,backend,source,*_=system
    if condition=='home':backend.home_status['busy']=True
    elif condition=='owned':backend.gripper['reserved']=True
    elif condition=='replay':backend.replay['is_paused']=True
    elif condition=='teleop':backend.robot['current_input_mode']=1
    else:
        source.speed=.1;wait(lambda:manager.snapshot.meta['control_feedback']['joints']['velocities'][0]==.1)
    assert start(client).status_code==202
    assert wait(lambda:final(client))['run']['state']=='FAILED'
    assert backend.calls==[]


@pytest.mark.parametrize('condition',['timeout','stale','bad_feedback'])
def test_ambiguous_reset_keeps_review_state_without_retry(system,condition):
    client,manager,backend,*_=system
    if condition=='timeout':backend.fail='request timeout'
    elif condition=='stale':backend.stale=True
    else:backend.bad_after=True
    start(client);result=wait(lambda:final(client))
    assert result['run']['state']=='UNCERTAIN'
    assert result['blocked']
    assert backend.calls==[('initial_target',[0.0,0.0])]
    assert start(client,'no-auto-retry').status_code==409
    backend.stale=False;backend.gripper.update(healthy=True,errors=[1,1])
    checked=client.post('/api/v1/gripper-restart/check',json={})
    assert checked.status_code==200,checked.text
    assert checked.json()['blocked'] is False
    assert backend.calls==[('initial_target',[0.0,0.0])]


def test_cross_origin_and_demo_rejected(system):
    client,manager,backend,*_=system
    assert client.post('/api/v1/gripper-restart/start',json={'command_id':'restart-test-command'},headers={'Origin':'http://evil'}).status_code==403
    manager.gripper_restart.enabled=False
    assert start(client).status_code==409
    assert backend.calls==[]


def test_service_restart_does_not_replay_operation(system):
    client,manager,backend,source,config,_=system
    ctl=manager.gripper_restart
    ctl.current=dict(id='a'*32,owner='owner',command_id='persisted-reset')
    ctl.update('RESTARTING','reset pending')
    restored=GripperRestartController(manager,config,backend)
    assert restored.current['state']=='INTERRUPTED_REVIEW'
    assert restored.blocked
    assert backend.calls==[]
    ctl.current=None
