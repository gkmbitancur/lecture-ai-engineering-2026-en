"""Recheck archived evidence without training, network access, or dependencies."""
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / 'results'


def read(path):
    return json.loads(path.read_text())


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    runner = module('verification_runner', HERE / 'run_experiments.py')
    scorer = module('verification_scorer', HERE / 'prepared/score_advanced.py')
    reference = runner.notebook_reference()
    implementation = read(HERE / 'prepared/implementation_data.json')
    glossary = read(ROOT / 'data/glossary.json')['terms']
    evaluation = read(HERE / 'prepared/advanced_eval.json')
    checks = {}

    runs = {name: read(RESULTS / name / 'result.json') for name in
            ('baseline', 'basic', 'implementation', 'advanced')}
    manifests = {name: read(RESULTS / name / 'manifest.json') for name in runs}
    hash_count = 0
    for name, manifest in manifests.items():
        require(manifest['status'] == 'completed', f'{name}: incomplete manifest')
        require(manifest['gpu'] == 'Tesla T4', f'{name}: unexpected GPU')
        expected_config = {k: v for k, v in manifest['config'].items()
                           if k not in ('work_dir', 'drive_dir')}
        require(runs[name]['config'] == expected_config, f'{name}: result/manifest configuration differs')
        for relative, expected in manifest['source_sha256'].items():
            path = ROOT / relative
            if relative == 'lectures/02/run_experiments.py' and name != 'advanced':
                path = HERE / 'run_experiments_v1.py'
            require(hashlib.sha256(path.read_bytes()).hexdigest() == expected,
                    f'{name}: source hash mismatch: {relative}')
            hash_count += 1
    require(len({m['pid'] for m in manifests.values()}) == 4, 'Repeated process IDs')
    changes = {k: [runs['baseline']['config'][k], runs['basic']['config'][k]]
               for k in runs['baseline']['config']
               if runs['baseline']['config'][k] != runs['basic']['config'][k]}
    require(changes == {'num_train_epochs': [20, 10]}, 'Basic changed other settings')
    require(runs['advanced']['config'] == runs['baseline']['config'],
            'Advanced changed training configuration beyond the dataset/task')
    require(runs['implementation']['config'] == runs['baseline']['config'],
            'Implementation changed training configuration beyond its data')
    base_before = read(RESULTS / 'baseline/eval_before.json')['rows']
    basic_before = read(RESULTS / 'basic/eval_before.json')['rows']
    require([(r['question'], r['raw']) for r in base_before] ==
            [(r['question'], r['raw']) for r in basic_before], 'Before outputs differ')
    checks['execution_and_controls'] = {
        'completed_runs': list(runs), 'distinct_process_ids': 4,
        'recorded_source_hashes_matched': hash_count,
        'basic_configuration_difference': changes,
        'identical_baseline_basic_before_answers': 21,
        'implementation_and_advanced_configuration_match_baseline': True,
    }

    for name, data in [('course_faq', reference), ('custom_faq', implementation)]:
        splits = [[e['question'] for e in data[split]] for split in ('train', 'valid', 'test')]
        require(list(map(len, splits)) == [14, 7, 21], f'{name}: wrong split sizes')
        questions = sum(splits, [])
        require(len(set(questions)) == 42, f'{name}: duplicate/leaked question strings')
        require(sum(e['needs_review'] for e in data['test']) == 6, f'{name}: wrong unknown count')
    generated = subprocess.run([sys.executable, str(ROOT / 'scripts/make_sft_dataset.py')],
                               cwd=ROOT, capture_output=True, text=True, check=True)
    rebuilt = [json.loads(line) for line in generated.stdout.splitlines()]
    training = [json.loads(line) for line in (HERE / 'prepared/glossary_sft.jsonl').read_text().splitlines()]
    require(rebuilt == training and len(training) == 116 and len(glossary) == 58,
            'Glossary training records differ from the course generator')
    questions = {r['messages'][0]['content'] for r in training}
    require(not questions & {e['question'] for e in evaluation['examples']},
            'Advanced evaluation question appears in training')
    source_ids = {e['id'] for e in glossary}
    names = {n.casefold() for e in glossary for n in [e['term'], e['id'], *e.get('aliases', [])]}
    category_counts = {}
    for example in evaluation['examples']:
        category_counts[example['category']] = category_counts.get(example['category'], 0) + 1
        if example['category'] == 'absent':
            require(example['term'].casefold() not in names, 'Absent term exists in glossary names')
        else:
            require(example['term_id'] in source_ids, 'Known term missing from glossary')
    require(category_counts == {'term_name': 8, 'paraphrase': 8, 'absent': 8}, 'Wrong categories')
    checks['data'] = {'faq_splits': [14, 7, 21], 'unique_questions_per_faq_dataset': 42,
                      'unknown_test_questions_per_faq_dataset': 6,
                      'glossary_entries': 58, 'exactly_regenerated_training_records': 116,
                      'advanced_category_counts': category_counts,
                      'advanced_training_evaluation_question_overlap': 0}

    faq_rows = 0
    faq_rates = {}
    for name in ('original_exercise', 'baseline', 'basic', 'implementation'):
        gold = implementation['gold'] if name == 'implementation' else reference['gold']
        phases = ('before', 'after', 'before_4bit') if name == 'original_exercise' else ('before', 'after')
        for phase in phases:
            saved = read(RESULTS / name / f'eval_{phase}.json')
            require(len(saved['rows']) == 21, 'Wrong FAQ row count')
            counts = {key: 0 for key in ('format', 'content', 'both')}
            for row in saved['rows']:
                actual = runner.score_faq(row['raw'], gold[row['question']], reference)
                for key in counts:
                    require(actual[key + '_ok'] == row[key + '_ok'], f'{name}/{phase}: saved judgment differs')
                    counts[key] += actual[key + '_ok']
                faq_rows += 1
            for key, count in counts.items():
                require(math.isclose(saved['summary'][key + '_rate'], count / 21), 'FAQ rate mismatch')
                if phase != 'before_4bit':
                    result = read(RESULTS / name / 'result.json')
                    require(result[phase][key + '_rate'] == saved['summary'][key + '_rate'],
                            f'{name}/{phase}: top-level result rate mismatch')
            faq_rates[f'{name}/{phase}'] = counts
    checks['faq_scoring'] = {'raw_responses_recomputed': faq_rows, 'count_denominator': 21,
                            'counts': faq_rates, 'all_saved_judgments_and_rates_matched': True}

    advanced_rates = {}
    for phase in ('before', 'after'):
        saved = read(RESULTS / 'advanced' / f'eval_{phase}.json')
        recalculated = scorer.score_all(saved['rows'], evaluation)
        for key, value in recalculated['summary'].items():
            require(saved['summary'][key] == value, f'Advanced {phase}: summary mismatch')
        require([r['proxy_pass'] for r in saved['rows']] ==
                [r['proxy_pass'] for r in recalculated['rows']], 'Advanced row score mismatch')
        require(runs['advanced'][phase] == saved['summary'], 'Advanced top-level result mismatch')
        advanced_rates[phase] = recalculated['summary']
    retrieval = read(RESULTS / 'advanced/retrieval.json')
    for example, row in zip(evaluation['examples'], retrieval['retrieval']):
        require(row['id'] == example['id'], 'Retrieval order/ID mismatch')
        calculated = runner.retrieve(example['question'], glossary)
        require(all(row[k] == v for k, v in calculated.items()), 'Retrieval output mismatch')
    recalculated = scorer.score_all(retrieval['retrieval'], evaluation)
    require(recalculated['summary'] == retrieval['summary'], 'Retrieval score mismatch')
    checks['advanced_scoring'] = {'model_answers_recomputed': 48, 'search_responses_reproduced': 24,
                                 'all_saved_scores_matched': True, 'model_rates': advanced_rates,
                                 'search_rates': recalculated['summary']}

    for name, result in runs.items():
        log = read(RESULTS / name / 'training_log.json')
        last_loss = [e for e in log if 'loss' in e][-1]
        summary = [e for e in log if 'train_loss' in e][-1]
        require(last_loss['loss'] == result['final_logged_loss'], f'{name}: final loss mismatch')
        require(last_loss['step'] == result['final_logged_loss_step'], f'{name}: loss step mismatch')
        require(summary['train_loss'] == result['mean_train_loss'], f'{name}: aggregate loss mismatch')
        require(result['training_seconds'] >= summary['train_runtime'], f'{name}: timing inconsistency')
        require(summary['step'] == result['global_step'] and summary['epoch'] == result['epoch'],
                f'{name}: training step/epoch mismatch')
    checks['training_logs'] = {'runs_cross_checked': 4, 'final_interval_and_aggregate_losses_match': True,
                               'wall_clock_training_times_cover_trainer_runtimes': True}

    tests = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s',
                            str(HERE / 'prepared'), '-p', 'test_runner.py', '-v'],
                           cwd=ROOT, capture_output=True, text=True)
    require(tests.returncode == 0 and 'Ran 8 tests' in tests.stderr, 'Runner tests failed')
    checks['runner_tests'] = {'passed': 8, 'output': tests.stdout + tests.stderr}
    report = {'verified_at_utc': datetime.now(timezone.utc).isoformat(),
              'method': 'Offline recomputation and consistency checks', 'status': 'passed',
              'scope': 'Recomputed archived results and checked recorded evidence. No GPU retraining in this verification pass.',
              'checks': checks,
              'limits': ['Matching the fixed scorer does not prove factual correctness; see the qualitative answer audits.',
                         'One seed and small evaluation sets do not establish robust generalization.']}
    (RESULTS / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'faq_answers': faq_rows, 'advanced_answers': 48,
                      'search_answers': 24, 'source_hashes': hash_count, 'runner_tests': 8}, indent=2))


if __name__ == '__main__':
    main()
