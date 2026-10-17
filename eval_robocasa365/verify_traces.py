"""Verify complete action replay and two independent closed-loop restarts per episode."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from trace_runner import sha_file, write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--port',type=int,default=10086)
    args=parser.parse_args()
    root=args.run.resolve()
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest['status']!='complete':
        raise ValueError('Collection must be complete before certification')
    checks=[]
    output=root/'verification'
    output.mkdir(exist_ok=True)
    for episode in manifest['episodes']:
        path=root/episode['path']
        snapshots=[s for s in episode['snapshots'] if s<episode['steps']]
        middle=min(snapshots,key=lambda s:abs(s-episode['steps']//2))
        plans=[('actions',0,0),('policy',middle,0)]
        if 7 in snapshots and middle!=7:
            plans.append(('policy',7,min(40,episode['steps']-7)))
        for mode,start,steps in plans:
            name=f"{episode['task']}_seed{episode['seed']}_{mode}_{start:06d}"
            report=output/(name+'.json')
            command=[sys.executable,str(Path(__file__).with_name('trace_runner.py')),'replay',
                     '--episode',str(path),'--mode',mode,'--start-step',str(start),
                     '--steps',str(steps),'--port',str(args.port),'--report',str(report)]
            if mode=='actions' and episode['seed'] in (7,157):
                command+=['--video',str(output/(name+'.mp4'))]
            with (output/(name+'.log')).open('w') as log:
                process=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,
                                       env={**os.environ,'MUJOCO_GL':'egl','PYTHONHASHSEED':'0'})
            result=json.loads(report.read_text()) if report.exists() else {'passed':False}
            check={'name':name,'exit_code':process.returncode,'report':str(report.relative_to(root)),
                   'passed':process.returncode==0 and result.get('passed',False)}
            checks.append(check)
            certificate={'schema':1,'verified_at_utc':datetime.now(timezone.utc).isoformat(),
                         'manifest_sha256':sha_file(root/'manifest.json'),
                         'episode_inventories':{e['path']:sha_file(root/e['path']/'files.sha256.json')
                                                for e in manifest['episodes']},
                         'checks':checks,'complete':False,'passed':False}
            write_json(root/'verification.json',certificate)
            print(json.dumps(check),flush=True)
            if not check['passed']:
                raise RuntimeError(f'Replay failed; inspect {report}')
    certificate.update(complete=True,passed=all(c['passed'] for c in checks))
    write_json(root/'verification.json',certificate)
    print(f'PASS: {len(checks)} replay checks across {len(manifest["episodes"])} episodes',flush=True)


if __name__=='__main__':
    main()
