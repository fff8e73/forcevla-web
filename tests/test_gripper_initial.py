import copy
import time

from forcevla_web.gripper_initial import TOPICS, run


class IO:
    def __init__(self):
        self.data=dict(boot_id='driver',reserved=False,healthy=True,errors=[1,1],
                       actual=[-.0211,.79168],applied_target=[.00256,1.],
                       motor_feedback_age=[.001,.001],calibration=[[0.,1.6],[0.,1.6]])
        self.selected=0;self.commands=[];self.fail_right=False
        self.takeover=False;self.apply=True;self.stale=False;self.reset_driver=False
    def discovered(self):return True
    def spin(self,seconds):time.sleep(.002)
    def input(self):return self.selected
    def status(self):
        value=copy.deepcopy(self.data);value['sample_time_ns']=time.time_ns()-2_000_000
        if self.stale:value['sample_time_ns']-=10_000_000_000
        return value
    def publish(self,index,value):
        if self.fail_right and index==1:raise OSError('right publish failed')
        self.commands.append((TOPICS[index],value))
        if self.apply:self.data['applied_target'][index]=value
        if self.takeover:self.data['reserved']=True
        if self.reset_driver:self.data['boot_id']='other-driver'


def test_no_execute_is_strictly_readonly():
    io=IO();result=run(io)
    assert result['status']=='preflight_passed_no_commands'
    assert io.commands==[]


def test_two_zero_goals_confirmed_without_waiting_for_actual_position():
    io=IO();result=run(io,execute=True)
    assert result['status']=='initial_targets_confirmed',result
    assert io.commands==[(TOPICS[0],0.),(TOPICS[1],0.)]
    assert result['hardware_commands_sent']==2
    assert result['after']['actual'][1]==.79168
    assert result['physical_arrival_verified'] is False


def test_owned_driver_and_nonidle_input_reject_before_publication():
    for key in ('owned','input','fault','stale','calibration'):
        io=IO()
        if key=='owned':io.data['reserved']=True
        if key=='input':io.selected=1
        if key=='fault':io.data['healthy']=False
        if key=='stale':io.stale=True
        if key=='calibration':io.data['calibration'][1][0]=.2
        result=run(io,execute=True)
        assert result['status']=='failed',key
        assert io.commands==[],key


def test_partial_send_not_retried_or_misreported():
    io=IO();io.fail_right=True
    result=run(io,execute=True)
    assert result['status']=='failed'
    assert result['published']==[True,False]
    assert len(io.commands)==1


def test_ownership_change_stops_second_send():
    io=IO();io.takeover=True
    result=run(io,execute=True)
    assert result['error']=='gripper_owned_do_not_override'
    assert result['published']==[True,False]


def test_missing_readback_does_not_republish():
    io=IO();io.apply=False
    result=run(io,execute=True,readback_timeout=.03)
    assert result['error']=='initial_target_readback_timeout_no_retry'
    assert len(io.commands)==2
    assert result['goal_readback_confirmed'] is False


def test_boot_change_invalidates_confirmation():
    io=IO();io.reset_driver=True
    result=run(io,execute=True)
    assert result['error']=='gripper_driver_restarted_during_command'
    assert result['goal_readback_confirmed'] is False
