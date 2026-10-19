"""Read-only observation health sampling. Does not start inference or robot motion."""
import argparse
import collections
import http.cookiejar
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--url',default='http://127.0.0.1:8096')
p.add_argument('--duration',type=float,default=60)
p.add_argument('--output',type=Path,required=True)
args=p.parse_args()
args.output.mkdir(parents=True,exist_ok=False)
jar=http.cookiejar.CookieJar()
op=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
op.open(args.url+'/api/v1/session',timeout=3).close()
rows=[];start=time.monotonic();last_report=-1
while time.monotonic()-start<args.duration:
    with op.open(args.url+'/api/v1/status',timeout=3) as response:s=json.load(response)
    inp=s['input'];rows.append(dict(time=time.time(),elapsed_s=time.monotonic()-start,snapshot_id=s.get('snapshot_id'),**inp))
    marker=int((time.monotonic()-start)//15)
    if marker!=last_report:
        print(json.dumps(dict(elapsed_s=round(time.monotonic()-start),ready=inp['ready'],reasons=inp['reasons'],ages=inp.get('ages'),transport=inp.get('transport')),ensure_ascii=False),flush=True)
        last_report=marker
    time.sleep(.25)
ages=[max(r['ages'].values()) for r in rows if r.get('ages') and all(x is not None for x in r['ages'].values())]
drops=[r['transport']['udp_drops_total'] for r in rows if r.get('transport') and 'udp_drops_total' in r['transport']]
summary=dict(samples=len(rows),duration_s=time.monotonic()-start,ready_samples=sum(r['ready'] for r in rows),
             ready_fraction=sum(r['ready'] for r in rows)/len(rows),max_age_s=max(ages),p95_age_s=float(np.percentile(ages,95)),
             unique_snapshots=len(set(r['snapshot_id'] for r in rows)),udp_drop_increase=max(drops)-min(drops) if drops else None,
             last_transport=rows[-1].get('transport'),reason_counts=dict(collections.Counter(reason.split(':')[0] for r in rows for reason in r['reasons'])))
(args.output/'samples.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
(args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
