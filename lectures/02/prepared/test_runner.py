"""Standard-library-only checks; no models, GPU, network, or package installs."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('runner',HERE.parent/'run_experiments.py')
runner=importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference=runner.notebook_reference()

    def tearDown(self):
        runner._USED=False

    def test_basic_changes_only_epoch_count(self):
        before=runner.configuration('baseline',self.reference)
        basic=runner.configuration('basic',self.reference)
        self.assertEqual(before['num_train_epochs'],20)
        self.assertEqual(basic['num_train_epochs'],10)
        self.assertEqual({k for k in before if before[k]!=basic[k]},{'num_train_epochs'})
        self.assertEqual(runner.configuration('baseline',self.reference),before)
        advanced=runner.configuration('advanced',self.reference)
        self.assertEqual(advanced['num_train_epochs'],20)
        self.assertEqual(advanced['max_seq_len'],512)
        self.assertEqual(advanced['max_new_tokens'],96)
        self.assertEqual(advanced['gen'],{'do_sample':False})

    def test_reference_and_implementation_mapping(self):
        baseline=runner.scenario_data('baseline',self.reference)
        data=runner.scenario_data('implementation',self.reference)
        original=json.loads((HERE/'implementation_data.json').read_text())
        self.assertEqual(data,{k:original[k] for k in ['train','valid','test','gold']})
        self.assertEqual([len(data[k]) for k in ['train','valid','test']],[14,7,21])
        self.assertEqual([len(baseline[k]) for k in ['train','valid','test']],[14,7,21])
        self.assertNotEqual(data['train'],baseline['train'])
        records=runner.training_records(data,False,self.reference)
        self.assertEqual(records[0]['messages'][-1]['content'],data['train'][0]['target'])
        prompt=runner.messages_for(data['test'][0],False,self.reference)
        self.assertEqual(prompt[0]['content'],self.reference['SYSTEM'])
        self.assertIn(data['test'][0]['faq'],prompt[1]['content'])

    def test_advanced_never_leaks_gold_and_preserves_training_records(self):
        data=runner.scenario_data('advanced',self.reference)
        self.assertIs(runner.training_records(data,True,self.reference),data['train'])
        self.assertEqual(len(data['train']),116)
        self.assertIsNone(data['valid'])
        for row in data['test']:
            poisoned=dict(row,expected_answer='GOLD_SENTINEL',term_id='ID_SENTINEL',reference={'secret':'SECRET_SENTINEL'},faq='FAQ_SENTINEL')
            messages=runner.messages_for(poisoned,True,self.reference)
            self.assertEqual(messages,[{'role':'user','content':row['question']}])
            self.assertNotIn('SENTINEL',json.dumps(messages))
        for record in data['train']:
            self.assertEqual([m['role'] for m in record['messages']],['user','assistant'])
        train_questions={r['messages'][0]['content'] for r in data['train']}
        self.assertFalse(train_questions & {e['question'] for e in data['test']})

    def test_knowledge_probes_include_faq_without_targets(self):
        for scenario in ['baseline','basic','implementation']:
            data=runner.scenario_data(scenario,self.reference)
            probes=runner.knowledge_inputs(scenario,data,self.reference)
            self.assertEqual(len(probes),4)
            self.assertEqual([p['in_faq'] for p in probes],[False,False,False,True])
            expected_faq=data['test'][0]['faq']
            for probe in probes:
                self.assertEqual(probe['faq'],expected_faq)
                self.assertNotIn('target',probe)
                self.assertIn(expected_faq,runner.messages_for(probe,False,self.reference)[1]['content'])

    def test_process_guard_and_output_preservation(self):
        runner.claim_process()
        with self.assertRaisesRegex(RuntimeError,'fresh Python process'): runner.claim_process()
        runner._USED=False
        with tempfile.TemporaryDirectory() as directory:
            sentinel=Path(directory)/'preserve.txt'; sentinel.write_text('existing output')
            with self.assertRaisesRegex(ValueError,'empty'): runner.run('baseline',directory)
            self.assertEqual(sentinel.read_text(),'existing output')

    def test_faq_scoring_preserves_format_content_separation(self):
        expected={'needs_review':True,'must_include':['Confirmation']}
        correct='{"answer":"Confirmation is needed","needs_review":true}'
        self.assertTrue(runner.score_faq(correct,expected,self.reference)['both_ok'])
        wrapped=runner.score_faq('```json\n'+correct+'\n```',expected,self.reference)
        self.assertFalse(wrapped['format_ok']); self.assertTrue(wrapped['content_ok'])
        false_bool='{"answer":"Confirmation is needed","needs_review":"true"}'
        self.assertFalse(runner.score_faq(false_bool,expected,self.reference)['both_ok'])

    def test_retrieval_uses_only_question_and_source_and_is_deterministic(self):
        entries=[{'id':'z','term':'Low-rank adaptation','aliases':['LoRA'],'definition':'Small matrices train frozen model updates.'},
                 {'id':'a','term':'Other concept','aliases':[],'definition':'Small matrices train frozen model updates.'}]
        exact=runner.retrieve('How does LoRA work?',entries)
        self.assertEqual(exact['source_id'],'z')
        self.assertEqual(exact['raw'],entries[0]['definition'])
        # Tie break is stable by source ID and independent of source ordering.
        lexical=runner.retrieve('What do small matrices train?',entries)
        self.assertEqual(lexical['source_id'],'a')
        self.assertEqual(lexical,runner.retrieve('What do small matrices train?',list(reversed(entries))))
        self.assertIsNone(runner.retrieve('Explain unlisted banana routing.',entries)['source_id'])
        self.assertIsNone(runner.retrieve('r',entries)['source_id'])

    def test_final_logged_interval_loss_is_distinct_from_mean(self):
        trainer=SimpleNamespace(state=SimpleNamespace(log_history=[{'loss':2.0,'step':10},{'loss':1.0,'step':20},{'train_loss':1.5,'step':23}]))
        loss=runner.loss_summary(trainer,SimpleNamespace(metrics={'train_loss':1.5}))
        self.assertEqual(loss['final_logged_loss'],1.0)
        self.assertEqual(loss['final_logged_loss_step'],20)
        self.assertEqual(loss['mean_train_loss'],1.5)


if __name__=='__main__': unittest.main()
