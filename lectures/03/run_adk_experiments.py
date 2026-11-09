"""Run ADK CLI against the same exercise questions, each in a fresh session.

GEMINI_API_KEY must be configured. From the repository root:
    uv run python -m lectures.03.run_adk_experiments
"""
import json
import os
from pathlib import Path
import subprocess
import time
import importlib

experiment=importlib.import_module('lectures.03.run_experiments')
OUT=Path(__file__).with_name('results')


def main():
    OUT.mkdir(exist_ok=True)
    env=os.environ.copy()
    env['PYTHONPATH']=str(experiment.ROOT)
    for name in ['comparison','levels','quiz']:
        path=OUT/f'adk_{name}.jsonl'
        if path.exists() and '"finishReason": "STOP"' in path.read_text():
            continue
        command=['uv','run','--with','google-adk==2.14.0','adk','run',
            '--jsonl','--in_memory','--timeout','90s','adk_agent/glossary_agent',experiment.QUESTIONS[name]]
        start=time.perf_counter()
        run=subprocess.run(command,env=env,text=True,capture_output=True,timeout=120)
        # ADK prints two logging notices before its JSONL events.
        events=[json.loads(line) for line in run.stdout.splitlines() if line.startswith('{')]
        path.write_text(''.join(json.dumps(event,ensure_ascii=False)+'\n' for event in events))
        (OUT/f'adk_{name}_run.json').write_text(json.dumps({'command':command,
            'exit_code':run.returncode,'elapsed_seconds':round(time.perf_counter()-start,3),
            'stderr':run.stderr},indent=2)+'\n')
        print(name,'exit',run.returncode,flush=True)
        if run.returncode:
            break
        time.sleep(10)


if __name__=='__main__':
    main()
