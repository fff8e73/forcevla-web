"""One read-only RPC using the live web observation; export ten physical targets."""
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid

import numpy as np
from PIL import Image

WEB = Path('/home/user/forcevla-web')
sys.path.insert(0, str(WEB))
from forcevla_web.observation import FileSource, CAMERAS
from forcevla_web.policy import RemotePolicy


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    config_path=WEB/'config/local.json'
    config=json.loads(config_path.read_text())
    if config['source']!='live':raise RuntimeError('Live data required; no demo fallback')
    policy=RemotePolicy(config)
    try:
        if not policy.healthy():raise RuntimeError('Model unavailable')
        source=FileSource(config['ipc_path']);deadline=time.monotonic()+15
        while True:
            snapshot=source.read();validation=snapshot.validate(config) if snapshot else {'ready':False,'reasons':['no snapshot']}
            if validation['ready']:break
            if time.monotonic()>deadline:raise RuntimeError('No fresh live observation: '+str(validation['reasons']))
            time.sleep(.1)
        captured=datetime.now().astimezone();request_id=uuid.uuid4().hex
        request={**snapshot.images,'state':snapshot.state.copy(),'prompt':config['default_prompt'],'seed':0}
        start=time.monotonic();result=policy.predict(request,request_id);completed=datetime.now().astimezone();rpc_ms=(time.monotonic()-start)*1000
    finally:policy.close()
    actions=result['actions']
    if actions.shape!=(1,50,14) or not np.isfinite(actions).all():raise ValueError('Invalid physical outputs')
    rows=actions[0,:10].copy()
    bounded=rows.copy();bounded[:,[6,13]]=np.clip(rows[:,[6,13]],0.,1.)
    tag=captured.strftime('%Y%m%d-%H%M%S')
    out=Path('/home/user/forcevla-web/robot_control/runs')/(tag+'-current-first10');out.mkdir(parents=True,exist_ok=False)
    (out/'observation.npz').write_bytes(snapshot.pack())
    np.savez_compressed(out/'prediction_all50.npz',actions=actions,normalized_actions=result['normalized_actions'])
    np.savez_compressed(out/'first10.npz',raw_actions=rows,bounded_gripper_display=bounded)
    columns=['left_x_m','left_y_m','left_z_m','left_roll_rad','left_pitch_rad','left_yaw_rad','left_gripper_raw',
             'right_x_m','right_y_m','right_z_m','right_roll_rad','right_pitch_rad','right_yaw_rad','right_gripper_raw']
    with (out/'first10.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['step_1based','model_index','nominal_time_s',*columns,'left_gripper_0_1','right_gripper_0_1'])
        for i,row in enumerate(rows):writer.writerow([i+1,i,i/30,*map(float,row),float(bounded[i,6]),float(bounded[i,13])])
    grid=np.vstack([np.hstack([snapshot.images[CAMERAS[0]],snapshot.images[CAMERAS[1]]]),np.hstack([snapshot.images[CAMERAS[2]],snapshot.images[CAMERAS[3]]])])
    Image.fromarray(grid).save(out/'observation_preview.png')
    metadata={'status':'captured_and_inferred_readonly','captured_at':captured.isoformat(),'completed_at':completed.isoformat(),
              'request_id':request_id,'snapshot_id':snapshot.id,'prompt':request['prompt'],'seed':0,'model_address':config['model_address'],
              'rpc_ms':rpc_ms,'model_inference_ms':result['inference_time_us']/1000,'input_validation_at_request':validation,
              'source_timestamps':snapshot.meta.get('sources'),'state':snapshot.state.tolist(),
              'source':'live web collector, one new RPC, not historical prediction','hardware_commands_sent':0,
              'selected_model_indices':list(range(10)),'physical_action_shape':list(actions.shape),
              'max_first10_gripper_correction':float(np.max(abs(rows[:,[6,13]]-bounded[:,[6,13]]))),
              'config_sha256':digest(config_path),'script_sha256':digest(__file__)}
    (out/'export_script.py').write_text(Path(__file__).read_text())
    metadata['artifact_sha256']={p.name:digest(p) for p in out.iterdir() if p.is_file()}
    (out/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    doc=Path('/home/user/forcevla-control-planning')/(tag+'-CURRENT_OBSERVATION_FIRST10_ZH.md')
    rel='../forcevla-web/data/integration-runs/'+out.name+'/'
    text=f'''# 当前真机观察的前10步模型预测

采集时间：{captured.isoformat()}  
推理完成：{completed.isoformat()}  
来源：当前8096采集器的真实四路相机和状态，单独发起一次新RPC；不是旧历史预测或模拟数据。

**本记录仅展示预测，未发送给机械臂或夹爪。** 这些是模型绝对TCP/夹爪目标，不是关节角，也不是实测执行轨迹；没有在本次导出中做执行放行或运动验证。

## 1. 请求与动作含义

- 模型：forcevla-dual，服务 `{config['model_address']}`。
- 请求ID：`{request_id}`。
- 观察ID：`{snapshot.id}`。
- seed：0；用于记录本次随机采样条件，不代表所有seed都会得到相同结果。
- 任务文本：`{request['prompt']}`。
- RPC耗时：{rpc_ms:.2f} ms；服务报告模型推理耗时：{result['inference_time_us']/1000:.2f} ms。
- 完整返回50步，本文件取索引0～9，显示为第1～10步；动作标签间隔1/30秒。
- 表中相对时间采用第1步t=0，最后一行t=0.300秒；若包含最后一步保持区间，10个目标间隔约0.333秒。它不含推理等待，也不是实际调度/执行时间。
- xyz单位为米，参考系base_link；姿态为固定轴XYZ外旋的roll/pitch/yaw，单位弧度，模型工具点为left_tool/right_tool。
- 夹爪是无量纲归一化电机行程。表中同时列原始值与限制到[0,1]后的展示值，只有夹爪列变化；并未据此下发动作。
- 表格保留6位小数，精确float32结果和更高精度文本在NPZ/CSV中。

## 2. 本次实际输入状态

| 手臂 | TCP xyz（m） | TCP roll/pitch/yaw（rad） | 实际夹爪反馈 | 原始力 Fx/Fy/Fz | 原始力矩 Tx/Ty/Tz |
|---|---|---|---:|---|---|
'''
    for label,off in [('左臂',0),('右臂',13)]:
        a=snapshot.state[off:off+13]
        fmt=lambda x:', '.join(f'{float(v):.6f}' for v in x)
        text+=f'| {label} | {fmt(a[:3])} | {fmt(a[3:6])} | {float(a[6]):.6f} | {fmt(a[7:10])} | {fmt(a[10:13])} |\n'
    text+='\n力/力矩保留原消息分量，字段单位语义为N、N·m，实际反馈的负夹爪值未被裁剪；其物理坐标/标定边界与现有核查记录相同。\n'
    for label,offset in [('左臂',0),('右臂',7)]:
        text+=f'\n## {3 if offset==0 else 4}. {label}前10步绝对目标\n\n'
        text+='| 步 | t（s） | x（m） | y（m） | z（m） | roll（rad） | pitch（rad） | yaw（rad） | 夹爪原始 | 夹爪0～1 |\n'
        text+='|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
        for i,row in enumerate(rows):
            values=[f'{i+1}',f'{i/30:.6f}']+[f'{float(v):.6f}' for v in row[offset:offset+7]]+[f'{float(bounded[i,offset+6]):.6f}']
            text+='| '+' | '.join(values)+' |\n'
    text+=f'''\n## 5. 原始文件与复核

- [四路输入画面预览]({rel}observation_preview.png)
- [输入快照NPZ]({rel}observation.npz)
- [前10步CSV]({rel}first10.csv)
- [前10步原始/夹爪限幅展示NPZ]({rel}first10.npz)
- [完整50步模型输出NPZ]({rel}prediction_all50.npz)
- [时间戳、输入有效性、状态与SHA-256记录]({rel}metadata.json)

输入在发起请求前通过当前网页的新鲜度、偏差和格式检查。这不等于曝光同步或物理执行安全已验证。该文档是一次观察对应的预测记录，查看时可能已经过期，不能直接用作后续控制指令。\n'''
    doc.write_text(text)
    with np.load(out/'prediction_all50.npz',allow_pickle=False) as d:np.testing.assert_array_equal(rows,d['actions'][0,:10])
    with (out/'first10.csv').open() as f:assert len(list(csv.reader(f)))==11
    assert len(rows)==10 and np.isfinite(rows).all()
    print(json.dumps({'document':str(doc),'artifacts':str(out),'request_id':request_id,'rpc_ms':rpc_ms,'rows_per_arm':10,'hardware_commands_sent':0},ensure_ascii=False))


if __name__=='__main__':main()
