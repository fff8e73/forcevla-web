import copy
import json
import threading
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from forcevla_web.app import create_app
from forcevla_web.home import ApexHomeClient, HomeController
from forcevla_web.manager import Manager
from forcevla_web.observation import DemoSource
from forcevla_web.policy import DemoPolicy
from forcevla_web.store import Store


class Source(DemoSource):
    speed = 0.
    def read(self):
        snap=super().read()
        snap.meta['control_feedback']={'joints':dict(positions=[0.]*14,
            velocities=[self.speed]*14,received_mono=time.monotonic())}
        return snap


class Backend:
    def __init__(self):
        self.calls=[]
        self.robot=dict(control_ready=True,control_error=False,arm_state=[1,1],
                        current_mode=1,current_input_mode=0,arm_state_age_sec=.01)
        self.home_status=dict(busy=False)
        self.gripper=dict(reserved=False)
        self.replay=dict(is_playing=False,is_paused=False)
        self.entered=threading.Event();self.release=threading.Event();self.release.set()
        self.fail=None
        self.stop_fails=False
    def get(self,name):
        return copy.deepcopy({'robot':self.robot,'home':self.home_status,
                              'gripper':self.gripper,'replay':self.replay}[name])
    def select(self,index):
        self.calls.append(('select',index))
        if index==0 and self.stop_fails:raise RuntimeError('stop readback unavailable')
        self.robot['current_input_mode']=index
        return {'success':True}
    def home(self):
        self.calls.append(('home',None));self.home_status={'busy':True}
        self.entered.set();assert self.release.wait(5)
        if self.fail=='uncertain':raise TimeoutError('native HTTP timeout')
        self.home_status={'busy':False,'phase':'completed','success':True}
        if self.fail:raise RuntimeError(self.fail)
        return dict(success=True,phase='completed',max_error_rad=.001)


def wait(predicate,timeout=4):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        value=predicate()
        if value:return value
        time.sleep(.02)
    raise AssertionError('timeout')


@pytest.fixture
def home_system(tmp_path):
    config=dict(source='live',allowed_hosts=['testserver'],max_age_s=.75,max_skew_s=.2,
                max_source_age_s=1,disconnect_grace_s=10,data_dir=str(tmp_path),quota_mb=100,
                closed_loop=dict(enabled=True,integration_root='/home/user/forcevla-web/robot_control'))
    backend=Backend();source=Source();manager=Manager(config,source,DemoPolicy(),Store(tmp_path,100))
    app=create_app(config,manager,home_backend=backend)
    with TestClient(app) as client:
        s=client.get('/api/v1/session').json()
        client.headers.update({'Origin':'http://testserver','X-CSRF-Token':s['csrf_token']})
        wait(lambda:manager.snapshot is not None)
        try:yield client,manager,backend,source,config
        finally:backend.release.set()


def start(client,command='home-test-command'):
    return client.post('/api/v1/home/start',json={'command_id':command})


def final(client):
    value=client.get('/api/v1/home/status').json()
    return value if value['run'] and value['run']['state'] in ('COMPLETED','FAILED','UNCERTAIN') else None


def test_home_success_once_native_confirmation_and_none(home_system):
    client,manager,backend,_,_=home_system
    assert client.get('/api/v1/status').json()['home']['enabled']
    assert backend.calls==[]
    response=start(client);assert response.status_code==202,response.text
    result=wait(lambda:final(client))
    assert result['run']['state']=='COMPLETED',result
    assert result['run']['native_result']['success'] is True
    assert result['run']['final_robot']['current_input_mode']==0
    assert backend.calls==[('select',2),('home',None),('select',0)]
    assert start(client).json()['run']['id']==result['run']['id']
    assert len(backend.calls)==3


def test_home_blocks_other_starts_and_duplicate_motion(home_system):
    client,manager,backend,_,_=home_system
    backend.release.clear();response=start(client);assert response.status_code==202
    assert backend.entered.wait(2)
    assert start(client,'another-home').status_code==409
    assert start(client).json()['run']['id']==response.json()['run']['id']
    assert client.post('/api/v1/runs',json=dict(kind='single',prompt='test',command_id='single-home-test')).status_code==409
    assert client.post('/api/v1/closed-loop/runs',json=dict(engine='movej',mode='live',prompt='test',command_id='loop-home-test')).status_code==409
    assert client.post('/api/v1/control-validation/start',json=dict(kind='stationary_cycle',command_id='stationary-home-test',safety_ack=True)).status_code==409
    backend.release.set();assert wait(lambda:final(client))['run']['state']=='COMPLETED'


@pytest.mark.parametrize('blocker',['loop','validation','inference'])
def test_home_rejects_busy_local_operation(home_system,blocker):
    client,manager,backend,_,_=home_system
    if blocker=='loop':manager.closed_loop=SimpleNamespace(blocked=True)
    elif blocker=='validation':manager.control_validation=SimpleNamespace(blocked=True)
    else:manager.run=dict(state='RUNNING')
    try:assert start(client).status_code==409;assert backend.calls==[]
    finally:
        if blocker=='inference':manager.run=None


@pytest.mark.parametrize('fault',['teleop','reserved','replay','native_busy','moving','stale','mode'])
def test_home_preflight_no_writes(home_system,fault):
    client,manager,backend,source,_=home_system
    if fault=='teleop':backend.robot['current_input_mode']=1
    elif fault=='reserved':backend.gripper['reserved']=True
    elif fault=='replay':backend.replay['is_playing']=True
    elif fault=='native_busy':backend.home_status['busy']=True
    elif fault=='moving':
        source.speed=.03
        wait(lambda:manager.snapshot.meta['control_feedback']['joints']['velocities'][0]==.03)
    elif fault=='stale':backend.robot['arm_state_age_sec']=2
    else:backend.robot['current_mode']=0;backend.robot['arm_state']=[0,0]
    assert start(client).status_code==202
    assert wait(lambda:final(client))['run']['state']=='FAILED'
    assert backend.calls==[]


def test_home_failure_disconnects_planner_and_reports_failure(home_system):
    client,_,backend,_,_=home_system;backend.fail='native HOME arrival timeout'
    start(client);result=wait(lambda:final(client))
    assert result['run']['state']=='FAILED'
    assert 'arrival timeout' in result['run']['message']
    assert backend.calls==[('select',2),('home',None),('select',0)]


def test_ambiguous_native_result_requires_readonly_reconciliation(home_system):
    client,manager,backend,_,_=home_system;backend.fail='uncertain'
    start(client);result=wait(lambda:final(client))
    assert result['run']['state']=='UNCERTAIN' and result['blocked']
    assert backend.calls==[('select',2),('home',None)]
    assert client.post('/api/v1/home/check',json={}).status_code==409
    backend.home_status['busy']=False;backend.robot['current_input_mode']=0
    assert client.post('/api/v1/home/check',json={}).json()['blocked'] is False
    assert backend.calls==[('select',2),('home',None)]


def test_home_origin_and_demo_block(home_system):
    client,manager,backend,_,_=home_system
    assert client.post('/api/v1/home/start',json={'command_id':'home-test-command'},headers={'Origin':'http://elsewhere'}).status_code==403
    manager.home.enabled=False
    assert start(client).status_code==409
    assert backend.calls==[]


def test_client_uses_existing_home_endpoint_and_requires_success():
    client=ApexHomeClient();calls=[]
    def request(path,body=None,timeout=3):
        calls.append((path,body,timeout));return {'success':True}
    client.request=request
    assert client.home()['success']
    assert calls==[('robot/reset',{},100)]
    client.request=lambda *a,**kw:{'success':False,'message':'not reached'}
    with pytest.raises(RuntimeError,match='not reached'):client.home()
