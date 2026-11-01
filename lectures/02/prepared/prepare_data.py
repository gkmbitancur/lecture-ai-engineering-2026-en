"""Rebuild local homework datasets using only the Python standard library."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
GLOSSARY_PATH = ROOT / 'data/glossary.json'
glossary = json.loads(GLOSSARY_PATH.read_text())
terms = {t['id']: t for t in glossary['terms']}
FAQ = {
 'pickup': 'Collect reserved equipment at the club room on Wednesday from 18:00 to 19:00.',
 'duration': 'The standard equipment loan lasts 7 calendar days.',
 'booking': 'Reserve equipment through the club lending portal under Equipment Reservations.',
 'limit': 'Each member may borrow up to 3 items at the same time.',
 'return': 'Return equipment to the club room on Monday from 17:00 to 18:00.',
}
GROUNDED = {
 'pickup': (["When can I collect reserved equipment?", "Tell me the equipment pickup schedule.", "Which day and time is collection at the club room?", "What is the collection window for my reservation?"], ['Wednesday', '18:00', '19:00']),
 'duration': (["How long is a standard equipment loan?", "What is the normal borrowing period?", "For how many calendar days can I keep borrowed items?", "When does the standard loan period end?"], ['7', 'days']),
 'booking': (["How do I reserve equipment?", "Where can I book an item to borrow?", "Which portal section handles equipment reservations?", "Where should I enter an equipment reservation?"], ['club lending portal', 'Equipment Reservations']),
 'limit': (["How many items may I borrow at once?", "What is the simultaneous borrowing limit?", "What is the maximum number of items one member can have on loan?", "Can a member check out four items together?"], ['3', 'items']),
 'return': (["When should I bring borrowed equipment back?", "Tell me the return desk schedule.", "Which day and hours are set for returns at the club room?", "What is the return window for borrowed gear?"], ['Monday', '17:00', '18:00']),
}
EXTRA = {
 'pickup': ["Where and when do I pick up my reserved gear?", "Is Wednesday evening the equipment collection time?"],
 'duration': ["What length of time does a usual equipment loan cover?", "Does the normal loan run for seven calendar days?"],
 'booking': ["Tell me the website and section used to book club equipment.", "How should a member submit a gear reservation?"],
 'limit': ["What is one member's equipment checkout cap?", "How many borrowed pieces can be recorded against my membership at the same time?"],
 'return': ["Where and when is the equipment hand-back session?", "Are returns handled on Monday in the late afternoon?"],
}
# Positions 0,1,4,5 train; 2,6 valid; 3,7 test, matching the notebook.
UNKNOWN = [
 "May I renew an equipment loan?", "Is there a deposit for borrowing equipment?", "Can a nonmember borrow club equipment?", "What happens if equipment is returned late?",
 "Can I cancel a reservation?", "Can another member collect equipment for me?", "How far ahead can I reserve an item?", "Is there a waiting list when all items are booked?",
]
EXTRA_UNKNOWN = ["Can I borrow equipment during the winter closure?", "Can I transfer my reservation to another member?", "Does the club deliver borrowed gear to my home?", "How many reservations can I make in a month?"]
UNKNOWN_ANSWER = 'This FAQ does not contain that information. Confirmation from the club lending coordinator is needed.'
faq_text = '\n'.join('- ' + value for value in FAQ.values())
implementation = {'FAQ': FAQ, 'GROUNDED': GROUNDED, 'UNGROUNDED': UNKNOWN, 'TEST_EXTRA_GROUNDED': EXTRA, 'TEST_EXTRA_UNGROUNDED': EXTRA_UNKNOWN, 'UNGROUNDED_ANSWER': UNKNOWN_ANSWER, 'train': [], 'valid': [], 'test': [], 'gold': {}}
def add(split, question, answer, review, keywords):
 ex = {'question':question,'faq':faq_text,'target':json.dumps({'answer':answer,'needs_review':review}),'needs_review':review}
 implementation[split].append(ex)
 if split == 'test': implementation['gold'][question] = {'needs_review':review,'must_include':keywords}
for key, (questions, keywords) in GROUNDED.items():
 for q in questions[:2]: add('train',q,FAQ[key],False,keywords)
 add('valid',questions[2],FAQ[key],False,keywords)
 for q in questions[3:] + EXTRA[key]: add('test',q,FAQ[key],False,keywords)
for i,q in enumerate(UNKNOWN): add(['train','train','valid','test'][i%4],q,UNKNOWN_ANSWER,True,['Confirmation'])
for q in EXTRA_UNKNOWN: add('test',q,UNKNOWN_ANSWER,True,['Confirmation'])
(OUT/'implementation_data.json').write_text(json.dumps(implementation,indent=2)+'\n')

# Alternatives in each group are OR; all concept groups are required (AND).
# No glossary definition is rewritten: expected_answer is the source definition verbatim.
KNOWN = [
 ('term_name','lora','Describe which weights LoRA trains and how it represents the update.', [['freezes','frozen','unchanged','fixed'], ['low rank','lowrank'], ['matrices','matrix']], ['trains all weights','updates all pretrained weights']),
 ('term_name','peft','Under PEFT, what fraction or kind of parameters is trained?', [['small subset','small set','few parameters','subset of parameters','fraction of parameters'], ['parameter','parameters','weights']], ['trains every parameter','updates all parameters']),
 ('term_name','qlora','How does QLoRA combine compression and adapter training?', [['quantiz','4 bit','four bit'], ['lora','low rank']], ['full precision base weights only']),
 ('term_name','quantization','Explain the memory benefit and possible accuracy cost of quantization.', [['fewer bits','lower precision','reduce precision','reduced precision','4 bit','four bit'], ['memory'], ['accuracy','quality','precision loss']], ['guarantees identical accuracy']),
 ('term_name','sft','For supervised fine-tuning, what does each training example pair together?', [['input','question','prompt'], ['output','answer','response'], ['desired','reference','target','correct','demonstration','label']], ['no target outputs are used']),
 ('term_name','fine-tuning','When is fine-tuning appropriate, and how should fresh facts usually be supplied?', [['pretrained','pre trained','existing model'], ['format','tone','behavior','behaviour','specific task'], ['external','rag','retrieval','tools']], ['best way to guarantee current facts']),
 ('term_name','epoch','In training terminology, how much data does one epoch traverse?', [['one','single','once'], ['entire','whole','all','full'], ['training dataset','training data','dataset']], ['one epoch is one batch']),
 ('term_name','overfitting','How would overfitting affect training versus unseen examples, and how can we detect it?', [['training'], ['unseen','validation','new inputs','new data'], ['worse','poor','degrad','fail','generaliz','generalis']], ['improves all unseen inputs']),
 ('paraphrase','context-window','What limits how much conversation history plus a generated reply can fit into one model call?', [['input','history'], ['output','reply','response','generated'], ['token'], ['maximum','limit','capacity']], ['unlimited context']),
 ('paraphrase','temperature','Which setting changes how random the next-token choice is, and what happens when it is lowered?', [['random','variability','varied','diversity'], ['lower','low value','decrease','reducing'], ['stable','determin','consistent','predictable','less random']], ['lower values increase randomness']),
 ('paraphrase','hallucination','What do we call a convincing model answer that invents facts, and how might external evidence help?', [['plausible','convincing','confident','realistic'], ['incorrect','false','nonexistent','fabricat','invent'], ['external','evidence','tools','data','retrieval']], ['always factually correct']),
 ('paraphrase','rate-limit','Why might an agent receive HTTP 429 after sending many requests in a short period?', [['limit','quota','allowance'], ['request','token'], ['time','period','minute','second','window']], ['429 means success']),
 ('paraphrase','ai-agent','What kind of system repeatedly chooses tools, inspects their results, and decides the next step toward a goal?', [['model','llm'], ['loop','repeat','iterat'], ['tool'], ['goal','task','objective']], ['never uses tools']),
 ('paraphrase','tool','When a model requests a function call with arguments, which component actually executes it?', [['function','tool'], ['model','llm'], ['program','application','runtime','host','code executes','system executes']], ['the model directly executes the function']),
 ('paraphrase','structured-output','How can an application constrain a model response to fields and types so downstream code can process it?', [['schema','types','typed','type'], ['json','structured'], ['process','pars','programmatic','validat']], ['no schema or types']),
 ('paraphrase','mcp','Which open protocol lets compatible AI apps reuse external tools and data exposed by a server?', [['open','standard'], ['protocol'], ['tool','data'], ['client','compatible','application','apps','agent']], ['closed proprietary protocol']),
]
ABSENT = [
 ('Direct Preference Optimization','How does Direct Preference Optimization learn from preference pairs?'),
 ('Proximal Policy Optimization','Describe the role of the clipped objective in Proximal Policy Optimization.'),
 ('Reinforcement Learning from Human Feedback','What stages are involved in Reinforcement Learning from Human Feedback?'),
 ('Mixture of Experts','How does a Mixture of Experts model route tokens to specialists?'),
 ('Diffusion model','How does a diffusion model learn to remove noise from a sample?'),
 ('Cosine similarity','How does cosine similarity compare two embedding vectors?'),
 ('BM25','What does BM25 use to rank a document for a search query?'),
 ('Beam search','How does beam search keep multiple candidate continuations?'),
]
evaluation=[]
for i,(category,key,q,groups,forbidden) in enumerate(KNOWN,1):
 evaluation.append({'id':f'advanced_{i:02d}','category':category,'term_id':key,'term':terms[key]['term'],'question':q,'expected_source_membership':True,'expected_answer':terms[key]['definition'],'reference':{'file':'data/glossary.json','id':key,'field':'definition'},'rubric':{'all_concept_groups':groups,'forbidden_phrases':forbidden}})
for i,(term,q) in enumerate(ABSENT,17):
 evaluation.append({'id':f'advanced_{i:02d}','category':'absent','term_id':None,'term':term,'question':q,'expected_source_membership':False,'expected_answer':f'{term} is not covered by the supplied course glossary. A glossary-backed definition cannot be provided; consult an additional source.','reference':{'file':'data/glossary.json','membership_check':'No term or alias matches this term, case-insensitively.'},'rubric':{'all_concept_groups':[['glossary'],['not covered','not included','not listed','not found','not contain','does not cover','does not include','does not contain','absent','no entry','no definition','not in']], 'forbidden_phrases':['is a method','is an algorithm','is a technique','refers to','is a metric','is a model','is a search algorithm','is a ranking function']}})
advanced={'schema_version':1,'training_file':'glossary_sft.jsonl','training_records':len(terms)*2,'training_policy':'All unmodified script records used for training; no validation subset; no claim of held-out-term factual generalization.','evaluation_input':'Exactly one plain-text user message containing question. No FAQ, JSON schema, source text, extra system instruction, or expected answer. Use identical chat template and decoding before/after.','purpose':'Definition generalization on known terms and source-membership-aware abstention stress test on absent terms.','rubric':{'normalization':'casefold; replace all runs of non-ASCII letters/digits with one space; strip. Match each alternative as a substring of normalized output. Partial stems (e.g. quantiz, generaliz) are intentional.','pass_rule':'Nonempty text, at least one matched alternative per required concept group, and no forbidden phrase. Report known-content and absent-membership proxy rates separately; macro average the three categories.','limits':'Keyword score is a proxy, not semantic truth. Negation and contradictions beyond explicit forbidden phrases may fool it. Absent scoring cannot prove no unsupported definition; manually audit all absent outputs and disagreements against raw text. Do not report JSON format_rate for this plain-text task.','membership_limitation':'Training script provides only definitions of known entries, with no unknown/abstention examples or explicit glossary membership inventory. Eval supplies no glossary context. General pretrained knowledge may answer absent terms correctly in the real world but fail this source-aware abstention target. Therefore this is a difficult membership stress test, not evidence of general factual error.','manual_review':'Inspect each raw answer for contradictions and unsupported factual clauses. Preserve automated proxy separately from adjudicated judgment; never adjust keywords after viewing outputs without labeling a post-hoc analysis.'},'examples':evaluation}
(OUT/'advanced_eval.json').write_text(json.dumps(advanced,indent=2)+'\n')

# Verifications include exact source-record equality and membership, with casefold normalization.
train_records=[json.loads(s) for s in (OUT/'glossary_sft.jsonl').read_text().splitlines()]
expected_records=[{'messages':[{'role':'user','content':template.format(term=t['term'])},{'role':'assistant','content':t['definition']}]} for t in glossary['terms'] for template in ['What is {term}?','Briefly explain {term}.']]
assert train_records == expected_records
all_impl=[e['question'] for k in ['train','valid','test'] for e in implementation[k]]
assert len(set(all_impl))==42
assert [len(implementation[k]) for k in ['train','valid','test']]==[14,7,21]
assert sum(not e['needs_review'] for e in implementation['test'])==15
assert sum(e['needs_review'] for e in implementation['test'])==6
assert len(implementation['gold'])==21
for ex in [e for k in ['train','valid','test'] for e in implementation[k]]:
 target=json.loads(ex['target']); assert target['needs_review']==ex['needs_review']; assert set(target)=={'answer','needs_review'}
questions=[x['messages'][0]['content'] for x in train_records]
assert not set(questions)&{e['question'] for e in evaluation}
assert len(set(e['question'] for e in evaluation))==24
vocabulary={v.casefold() for t in glossary['terms'] for v in [t['term'],*t.get('aliases',[])]}
for e in evaluation:
 if e['term_id'] is not None: assert e['term_id'] in terms
 else: assert e['term'].casefold() not in vocabulary
for path in ['implementation_data.json','advanced_eval.json']: json.loads((OUT/path).read_text())
print(json.dumps({'implementation':{k:len(implementation[k]) for k in ['train','valid','test','gold']},'glossary_terms':len(terms),'glossary_training_records':len(train_records),'advanced_eval':len(evaluation),'checks':'passed'}))
