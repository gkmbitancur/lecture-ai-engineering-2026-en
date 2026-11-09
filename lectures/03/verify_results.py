"""Recompute recorded claims without API calls: uv run python -m lectures.03.verify_results."""
import hashlib
import json
from pathlib import Path

from app import tools
from app.agent import INSTRUCTIONS


def main():
    out=Path(__file__).with_name('results')
    root=Path(__file__).resolve().parents[2]
    config=json.loads((out/'experiment_config.json').read_text())
    assert config['instructions_after']==INSTRUCTIONS
    for relative,digest in config['data_sha256'].items():
        assert hashlib.sha256((root/relative).read_bytes()).hexdigest()==digest
    summary={'pydantic':[],'adk':[]}
    files=sorted(out.glob('basic_*.json'))+sorted(out.glob('exercise_*.json'))+sorted(out.glob('bare_*.json'))
    assert len(files)==12
    for path in files:
        data=json.loads(path.read_text())
        calls=[]
        if 'messages' in data:
            responses=[m for m in data['messages'] if m['kind']=='response']
            assert len(responses)==data['requests'],path.name
            for message in data['messages']:
                for part in message['parts']:
                    if part['part_kind']=='tool-call':
                        args=part['args']
                        if isinstance(args,str):args=json.loads(args)
                        calls.append({'name':part['tool_name'],'args':args})
                    if part['part_kind']=='tool-return' and part['tool_name']=='make_quiz':
                        quiz=json.loads(part['content'])
                        source=next(t for t in tools.load_glossary() if t['term']==quiz['term'])
                        assert quiz['answer']==source['definition'] and quiz['lecture']==source['lecture']
            assert calls==data['tool_calls'],path.name
            assert sum(m['usage']['input_tokens'] for m in responses)==data['usage']['input_tokens']
            assert sum(m['usage']['output_tokens'] for m in responses)==data['usage']['output_tokens']
        else:
            assert data['requests']==1 and data['answer']
            parts=data['raw']['candidates'][0]['content']['parts']
            assert ''.join(p.get('text','') for p in parts)==data['answer']
        summary['pydantic'].append({'record':path.name,'requests':data['requests'],
            'tool_calls':data['tool_calls'],'completed':not bool(data['error']),
            'input_tokens':data.get('usage',{}).get('input_tokens'),
            'output_tokens':data.get('usage',{}).get('output_tokens')})
    for path in sorted(out.glob('adk_*.jsonl')):
        events=[json.loads(line) for line in path.read_text().splitlines()]
        calls=[p['functionCall'] for e in events for p in e.get('content',{}).get('parts',[]) if 'functionCall' in p]
        responses=[e for e in events if e.get('content',{}).get('role')=='model']
        answers=[p['text'] for e in responses for p in e['content']['parts'] if 'text' in p]
        assert answers and responses[-1].get('finishReason')=='STOP',path.name
        summary['adk'].append({'record':path.name,'requests':len(responses),
            'tool_calls':[{'name':c['name'],'args':c['args']} for c in calls],
            'answer':answers[-1],
            'input_tokens':sum(e['usageMetadata']['promptTokenCount'] for e in responses),
            'output_tokens':sum(e['usageMetadata']['candidatesTokenCount'] for e in responses)})
    assert len(summary['adk'])==3
    approval=json.loads((out/'approval_verification.json').read_text())
    for run in approval:
        assert run['files_before_approval']==0
        assert run['files_after_decision']==int(run['approved'])
        if run['approved']:
            saved=run['saved_records'][0]
            assert all(saved[k]==v for k,v in run['requested_arguments'].items())
    (out/'verified_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('Verified 12 Pydantic/bare records, 3 ADK event traces, 2 approval outcomes, data hashes, request/token counts and quiz source fidelity.')
    print('Successful controlled answers:',sum(r['completed'] for r in summary['pydantic']),'/ 12')
    print('Recorded controlled LLM requests:',sum(r['requests'] for r in summary['pydantic'])+sum(r['requests'] for r in summary['adk']))


if __name__=='__main__':
    main()
