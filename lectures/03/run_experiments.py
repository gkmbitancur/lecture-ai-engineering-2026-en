"""Run bounded real-model comparisons; preserve full outputs and messages.

Run from the repository root with GEMINI_API_KEY configured:
    uv run python -m lectures.03.run_experiments
Existing successful records are kept on resume. No learning-log tool is offered.
"""
import ast
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time

from google import genai
from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.messages import ToolCallPart, ModelMessagesTypeAdapter
from pydantic_ai.usage import UsageLimits, RunUsage

from app import tools
from app.agent import INSTRUCTIONS, build_model, print_trace
from app.config import MODEL, get_api_key

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / 'results'
BASE_COMMIT = '8d5b90c'
source = subprocess.check_output(['git','show',f'{BASE_COMMIT}:app/agent.py'], text=True)
node = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign) and
    any(isinstance(t,ast.Name) and t.id == 'INSTRUCTIONS' for t in n.targets))
ORIGINAL_INSTRUCTIONS = ast.literal_eval(node.value)
READ_TOOLS = [tools.search_glossary, tools.get_lecture_summary, tools.make_quiz]
QUESTIONS = {
    'levels': 'What are the 3 homework levels in the AIE2026 course?',
    'tag': 'What is a lecNN-start tag in the AIE2026 course?',
    'comparison': 'What is the difference between LoRA and QLoRA?',
    'quiz': 'Give me one review question from Lecture 02',
    'overview': 'What will we learn in Lecture 03?',
}


def main():
    OUT.mkdir(exist_ok=True)
    meta = {'model': MODEL, 'temperature': 0, 'request_limit': 6, 'base_commit': BASE_COMMIT,
        'instructions_before': ORIGINAL_INSTRUCTIONS, 'instructions_after': INSTRUCTIONS,
        'tools': [f.__name__ for f in READ_TOOLS],
        'versions': {n: importlib.metadata.version(n) for n in ['pydantic-ai-slim','google-genai']},
        'data_sha256': {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [ROOT/'data/glossary.json', ROOT/'data/lectures/03.md']}}
    (OUT/'experiment_config.json').write_text(json.dumps(meta,indent=2)+'\n')
    jobs = [('bare_tag', 'bare', 'tag'), ('bare_levels', 'bare', 'levels')]
    jobs += [(f'exercise_{q}', 'after', q) for q in ['tag','levels','comparison','quiz']]
    jobs += [(f'basic_{i}_{mode}', mode, 'overview') for i in range(1,4) for mode in ['before','after']]
    for label, mode, q in jobs:
        path = OUT/f'{label}.json'
        if path.exists() and not json.loads(path.read_text()).get('error'):
            continue
        if path.exists():
            attempts = OUT / 'attempts'
            attempts.mkdir(exist_ok=True)
            path.rename(attempts / f'{label}-{time.time_ns()}.json')
        record = {'label':label,'condition':mode,'question':QUESTIONS[q], 'model':MODEL,
            'started_at':datetime.now(timezone.utc).isoformat()}
        start=time.perf_counter()
        captured=[]
        usage=RunUsage()
        try:
            if mode == 'bare':
                with genai.Client(api_key=get_api_key()) as client:
                    response = client.models.generate_content(
                        model=MODEL, contents=QUESTIONS[q], config={'temperature':0})
                record.update(answer=response.text, requests=1, tool_calls=[],
                    raw=response.model_dump(mode='json'))
            else:
                target=Agent(instructions=ORIGINAL_INSTRUCTIONS if mode=='before' else INSTRUCTIONS,
                    tools=READ_TOOLS)
                with capture_run_messages() as captured:
                    result=target.run_sync(QUESTIONS[q],model=build_model(), model_settings={'temperature':0},
                        usage=usage, usage_limits=UsageLimits(request_limit=6,tool_calls_limit=10))
                record.update(answer=result.output, requests=result.usage.requests,
                    usage=asdict(result.usage), messages=json.loads(result.all_messages_json()),
                    tool_calls=[{'name':p.tool_name,'args':p.args_as_dict()} for m in result.all_messages()
                        for p in m.parts if isinstance(p,ToolCallPart)])
                print_trace(result)
            record['error']=None
        except Exception as exc:
            record['error']=f'{type(exc).__name__}: {exc}'
            if mode != 'bare':
                record.update(requests=usage.requests, usage=asdict(usage),
                    messages=json.loads(ModelMessagesTypeAdapter.dump_json(captured)),
                    tool_calls=[{'name':p.tool_name,'args':p.args_as_dict()} for m in captured
                        for p in m.parts if isinstance(p,ToolCallPart)])
        record['elapsed_seconds']=round(time.perf_counter()-start,3)
        # SDK costs use Decimal; preserve their exact value as a JSON string.
        path.write_text(json.dumps(record,ensure_ascii=False,indent=2,default=str)+'\n')
        print(label, 'ERROR' if record['error'] else 'OK', record.get('requests'), flush=True)
        if record['error']:
            print(record['error'],flush=True)
        time.sleep(30 if record['error'] else 10)


if __name__=='__main__':
    main()
