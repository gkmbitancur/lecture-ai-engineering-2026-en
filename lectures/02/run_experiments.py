#!/usr/bin/env python3
"""Run one isolated Lecture 02 experiment in the official pinned Colab environment.

Invoke each scenario in a fresh Python process. Installs nothing. Outputs are
created only by real inference/training; failures remain explicitly incomplete.
"""
import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREPARED = HERE / 'prepared'
_USED = False


def claim_process():
    global _USED
    if _USED:
        raise RuntimeError('Start a fresh Python process for every scenario; adapters may remain in memory.')
    _USED = True


def notebook_reference():
    cells = [''.join(c.get('source', [])) for c in json.loads((ROOT/'notebooks/lora_format_tuning.ipynb').read_text())['cells'] if c['cell_type']=='code']
    ns = {'json':json}
    functions = {'make_example','split_templates','build_messages','check_format','check_content','to_chat'}
    for source in cells:
        if any(line.lstrip().startswith('%') for line in source.splitlines()):
            continue
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {'CONFIG','SYSTEM'} for t in node.targets):
                ns[node.targets[0].id] = ast.literal_eval(node.value)
            elif isinstance(node, ast.FunctionDef) and node.name in functions:
                exec(compile(ast.Module(body=[node], type_ignores=[]),'<notebook-function>','exec'),ns)
        if source.startswith('FAQ = '):
            # Preserve actual course splitting, but skip display and output writes.
            nodes = [n for n in tree.body if not isinstance(n,ast.Expr) and not (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_' for t in n.targets))]
            exec(compile(ast.Module(body=nodes,type_ignores=[]),'<notebook-data>','exec'),ns)
    assert all(k in ns for k in ['CONFIG','SYSTEM','train','valid','test','gold','to_chat'])
    return ns


def scenario_data(scenario, reference):
    if scenario == 'advanced':
        spec = json.loads((PREPARED/'advanced_eval.json').read_text())
        records = [json.loads(line) for line in (PREPARED/'glossary_sft.jsonl').read_text().splitlines()]
        assert len(records)==116 and all(set(r)=={'messages'} for r in records)
        return {'train':records,'valid':None,'test':spec['examples'],'specification':spec}
    data = json.loads((PREPARED/'implementation_data.json').read_text()) if scenario=='implementation' else reference
    return {key:data[key] for key in ['train','valid','test','gold']}


def configuration(scenario, reference):
    config = json.loads(json.dumps(reference['CONFIG']))
    if scenario=='basic': config['num_train_epochs']=10
    return config


def messages_for(example, advanced, reference):
    if advanced:
        return [{'role':'user','content':example['question']}]
    return reference['build_messages'](example)


def training_records(data, advanced, reference):
    return data['train'] if advanced else [reference['to_chat'](e) for e in data['train']]



def knowledge_inputs(scenario, data, reference):
    questions=([e['question'] for e in data['test'][-3:]] + [data['train'][0]['question']] if scenario=='implementation' else ["When is the final assignment deadline for this course?","What is the instructor's name?","What is the topic of the next lecture?","When is the assignment deadline?"])
    faq=data['test'][0]['faq'] if scenario=='implementation' else reference['FAQ_TEXT']
    grounded={e['question'] for split in ['train','valid','test'] for e in data[split] if not e['needs_review']}
    return [{'question':q,'faq':faq,'in_faq':q in grounded} for q in questions]


def normalize(text):
    return re.sub(r'[^a-z0-9]+',' ',text.casefold()).strip()


STOP = set('a an the is are was were be to of in on and or for with how what which why when does do can it its this that from as by at into more one through will would should have has their them then than using use we i they about explain describe tell me much kind each together versus could learn'.split())


def retrieve(question, entries):
    """Longest boundary-matched name/alias/id (>=3 chars), else lexical overlap.

    Course app/tools.py searches exact names, then partial names, then definition
    substrings and returns up to three entries. Its query contract is one term.
    This question-level extension uses a single longest-name hit, or intersection
    of non-stopword token sets / query tokens >=0.30 with >=3 overlapping tokens.
    Equal lexical scores resolve by glossary ID. No gold IDs enter this function.
    """
    q = normalize(question)
    spans=[]
    for entry in entries:
        names=[entry['term'],entry['id'],*entry.get('aliases',[])]
        for name in names:
            n=normalize(name)
            if len(n)>=3 and f' {n} ' in f' {q} ': spans.append((len(n),entry['id'],entry))
    if spans:
        hit=sorted(spans,key=lambda v:(-v[0],v[1]))[0]
        return {'source_id':hit[1],'raw':hit[2]['definition'],'method':'name_alias_id_span','score':None}
    query=set(q.split())-STOP
    ranked=[]
    for entry in entries:
        document=set(normalize(' '.join([entry['term'],entry['id'],*entry.get('aliases',[]),entry['definition']])).split())-STOP
        overlap=len(query & document)
        score=overlap/max(1,len(query))
        if overlap>=3 and score>=0.30: ranked.append((score,entry['id'],entry))
    if ranked:
        hit=sorted(ranked,key=lambda v:(-v[0],v[1]))[0]
        return {'source_id':hit[1],'raw':hit[2]['definition'],'method':'lexical_overlap','score':hit[0]}
    return {'source_id':None,'raw':'This term is not covered by the course glossary. A glossary-backed definition cannot be provided.','method':'no_match','score':None}


def score_faq(raw, expected, reference):
    ok, reason, obj = reference['check_format'](raw)
    loose=obj
    if not ok:
        match=re.search(r'\{.*\}',raw,re.S)
        try: loose=json.loads(match.group(0)) if match else None
        except (ValueError,TypeError): loose=None
    content=reference['check_content'](loose if isinstance(loose,dict) else None,expected)
    return {'format_ok':ok,'format_reason':reason,'content_ok':content,'both_ok':ok and content,'gold_needs_review':expected['needs_review']}


def loss_summary(trainer, train_output):
    logged=[r for r in trainer.state.log_history if 'loss' in r]
    last=logged[-1] if logged else None
    return {'final_logged_loss':last['loss'] if last else None,'final_logged_loss_step':last.get('step') if last else None,
            'final_logged_loss_definition':'Last Trainer log_history loss value: interval-averaged training loss at its logged step; not necessarily final-batch loss.',
            'mean_train_loss':train_output.metrics.get('train_loss'),'mean_train_loss_definition':'Trainer train() metrics train_loss, reported separately from the last logged interval loss.'}


def write_json(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def run(scenario, output_dir):
    claim_process()
    print(f'[{scenario}] Starting isolated experiment',flush=True)
    output_dir=Path(output_dir).resolve(); output_dir.mkdir(parents=True,exist_ok=True)
    if any(output_dir.iterdir()): raise ValueError('Output directory must be empty, so previous results are preserved.')
    reference=notebook_reference(); data=scenario_data(scenario,reference); config=configuration(scenario,reference)
    advanced=scenario=='advanced'
    manifest={'scenario':scenario,'status':'started','pid':os.getpid(),'fresh_process_required':True,'config':config,'seed':config['seed'],'python':platform.python_version(),
              'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'notebooks/lora_format_tuning.ipynb',ROOT/'scripts/make_sft_dataset.py',ROOT/'data/glossary.json',PREPARED/'implementation_data.json',PREPARED/'advanced_eval.json',PREPARED/'glossary_sft.jsonl',PREPARED/'score_advanced.py',Path(__file__)]}}
    write_json(output_dir/'manifest.json',manifest)
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
        from peft import LoraConfig
        from trl import SFTTrainer, SFTConfig
        from datasets import Dataset
        if not torch.cuda.is_available(): raise RuntimeError('This experiment requires the official Colab GPU runtime.')
        set_seed(config['seed'])
        manifest.update(versions={name:importlib.metadata.version(name) for name in ['torch','transformers','peft','trl','accelerate','datasets']},gpu=torch.cuda.get_device_name(0),gpu_total_memory_bytes=torch.cuda.get_device_properties(0).total_memory)
        write_json(output_dir/'manifest.json',manifest)
        print(f'[{scenario}] Loading pinned base model on {manifest["gpu"]}',flush=True)
        tok=AutoTokenizer.from_pretrained(config['base_model'],revision=config['revision'])
        if tok.pad_token is None: tok.pad_token=tok.eos_token
        model=AutoModelForCausalLM.from_pretrained(config['base_model'],revision=config['revision'],dtype=torch.float32).to('cuda')
        model.config.use_cache=True; model.eval()
        def generate(example):
            prompt=tok.apply_chat_template(messages_for(example,advanced,reference),tokenize=False,add_generation_prompt=True,enable_thinking=False)
            inputs=tok(prompt,return_tensors='pt').to(model.device)
            with torch.no_grad(): output=model.generate(**inputs,max_new_tokens=config['max_new_tokens'],pad_token_id=tok.pad_token_id,**config['gen'])
            return tok.decode(output[0][inputs['input_ids'].shape[1]:],skip_special_tokens=True).strip()
        scorer=None
        if advanced:
            module=importlib.util.spec_from_file_location('advanced_scorer',PREPARED/'score_advanced.py'); scorer=importlib.util.module_from_spec(module); module.loader.exec_module(scorer)
            entries=json.loads((ROOT/'data/glossary.json').read_text())['terms']
            retrieval=[dict(id=e['id'],question=e['question'],**retrieve(e['question'],entries)) for e in data['test']]
            scored=scorer.score_all(retrieval,data['specification']); scored['retrieval']=retrieval
            scored['algorithm']=retrieve.__doc__; write_json(output_dir/'retrieval.json',scored)
        def evaluate(label):
            started=time.perf_counter(); rows=[]
            for i,ex in enumerate(data['test']):
                print(f'[{scenario}] {label}: question {i+1}/{len(data["test"])}',flush=True)
                raw=generate(ex); row={'id':ex.get('id',f'faq_{i+1:02d}'),'question':ex['question'],'raw':raw}
                if not advanced: row.update(score_faq(raw,data['gold'][ex['question']],reference))
                rows.append(row)
            if advanced:
                result=scorer.score_all(rows,data['specification'])
                for scored,row in zip(result['rows'],rows): scored['question']=row['question']
            else: result={'summary':{'n':len(rows),**{name+'_rate':sum(r[name+'_ok'] for r in rows)/len(rows) for name in ['format','content','both']}},'rows':rows}
            result['summary'].update(label=label,seconds=time.perf_counter()-started)
            write_json(output_dir/f'eval_{label}.json',result)
            return result
        knowledge=[]; probes=[]
        if not advanced:
            probes=knowledge_inputs(scenario,data,reference)
            for i,ex in enumerate(probes):
                print(f'[{scenario}] Knowledge before: {i+1}/{len(probes)}',flush=True)
                knowledge.append({'question':ex['question'],'in_faq':ex['in_faq'],'faq_provided':True,'before_raw':generate(ex)})
        before=evaluate('before')
        train_ds=Dataset.from_list(training_records(data,advanced,reference))
        valid_ds=None if advanced else Dataset.from_list([reference['to_chat'](e) for e in data['valid']])
        model.config.use_cache=False
        lora=LoraConfig(r=config['lora_r'],lora_alpha=config['lora_alpha'],lora_dropout=0.05,bias='none',task_type='CAUSAL_LM',target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'])
        sft=SFTConfig(output_dir=str(output_dir/'ckpt'),num_train_epochs=config['num_train_epochs'],per_device_train_batch_size=4,gradient_accumulation_steps=2,learning_rate=config['lr'],lr_scheduler_type='cosine',warmup_steps=config['warmup_steps'],logging_steps=10,save_steps=10,save_total_limit=2,fp16=False,max_length=config['max_seq_len'],packing=False,report_to='none',seed=config['seed'])
        trainer=SFTTrainer(model=model,args=sft,train_dataset=train_ds,eval_dataset=valid_ds,peft_config=lora,processing_class=tok)
        trainable=sum(p.numel() for p in trainer.model.parameters() if p.requires_grad); total=sum(p.numel() for p in trainer.model.parameters())
        manifest.update(train_records=len(train_ds),validation_records=0 if valid_ds is None else len(valid_ds),trainable_parameters=trainable,total_parameters=total)
        write_json(output_dir/'manifest.json',manifest)
        print(f'[{scenario}] Training {len(train_ds)} records for {config["num_train_epochs"]} epochs; {trainable} trainable parameters',flush=True)
        started=time.perf_counter(); train_output=trainer.train(); duration=time.perf_counter()-started
        print(f'[{scenario}] Training completed in {duration:.1f}s: {json.dumps(loss_summary(trainer,train_output))}',flush=True)
        model=trainer.model; model.save_pretrained(output_dir/'adapter'); tok.save_pretrained(output_dir/'adapter')
        model.config.use_cache=True; model.eval()
        after=evaluate('after')
        for i,(row,ex) in enumerate(zip(knowledge,probes)):
            print(f'[{scenario}] Knowledge after: {i+1}/{len(probes)}',flush=True)
            row['after_raw']=generate(ex)
            for stage in ['before','after']:
                ok,_,obj=reference['check_format'](row[stage+'_raw']); row[stage+'_classification']='format_error' if not ok else ('needs_review' if obj['needs_review'] else 'asserted')
        success='proxy_pass' if advanced else 'both_ok'
        regressed=[{'id':b['id'],'question':b['question']} for b,a in zip(before['rows'],after['rows']) if b[success] and not a[success]]
        result={'scenario':scenario,'config':{k:v for k,v in config.items() if k not in ['work_dir','drive_dir']},'versions':manifest['versions'],'gpu':manifest['gpu'],'seed':config['seed'],'method':'LoRA (no quantization)','executed_training_myself':True,'train_records':len(train_ds),'validation_records':0 if valid_ds is None else len(valid_ds),'before':before['summary'],'after':after['summary'],'regressed':regressed,'knowledge':knowledge,'training_seconds':duration,'trainable_parameters':trainable,'total_parameters':total,'trainable_percent':100*trainable/total,'global_step':trainer.state.global_step,'epoch':trainer.state.epoch,**loss_summary(trainer,train_output)}
        write_json(output_dir/'training_log.json',trainer.state.log_history)
        write_json(output_dir/'result.json',result)
        manifest['status']='completed'; write_json(output_dir/'manifest.json',manifest)
        print(json.dumps({'scenario':scenario,'status':'completed','before':result['before'],'after':result['after'],'regression_count':len(regressed),'training_seconds':duration,'result_file':str(output_dir/'result.json')},indent=2),flush=True)
    except Exception as exc:
        manifest.update(status='failed',error_type=type(exc).__name__,error=str(exc)); write_json(output_dir/'manifest.json',manifest)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario',required=True,choices=['baseline','basic','implementation','advanced'])
    parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args(); run(args.scenario,args.output_dir)
