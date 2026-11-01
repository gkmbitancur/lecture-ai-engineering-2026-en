"""Transparent keyword proxy. CLI input: list of {id, raw} answers as JSON.

Usage: python score_advanced.py responses.json > advanced_scores.json
No model inference and no external dependencies. Does not replace manual review.
"""
import json
import re
import sys
from pathlib import Path


def normalize(text):
    return re.sub(r'[^a-z0-9]+', ' ', str(text).casefold()).strip()


def score_one(raw, example):
    text = normalize(raw)
    rubric = example['rubric']
    groups = rubric['all_concept_groups']
    matches = [[p for p in group if normalize(p) in text] for group in groups]
    forbidden = [p for p in rubric['forbidden_phrases'] if normalize(p) in text]
    return {'id':example['id'], 'category':example['category'], 'raw':raw,
            'proxy_pass':bool(text) and all(matches) and not forbidden,
            'concept_matches':matches, 'forbidden_matches':forbidden,
            'requires_manual_review':True}


def score_all(responses, specification):
    expected = {e['id']: e for e in specification['examples']}
    assert len(responses) == len(expected), 'Supply one raw output per evaluation example.'
    assert len({r['id'] for r in responses}) == len(responses), 'Duplicate output IDs.'
    assert {r['id'] for r in responses} == set(expected), 'Output IDs differ from evaluation IDs.'
    rows = [score_one(r['raw'], expected[r['id']]) for r in responses]
    rates = {category:sum(r['proxy_pass'] for r in rows if r['category']==category) / sum(r['category']==category for r in rows)
             for category in ['term_name','paraphrase','absent']}
    known = [r for r in rows if r['category'] != 'absent']
    return {'summary':{'n':len(rows), 'category_proxy_rates':rates,
                       'known_content_proxy_rate':sum(r['proxy_pass'] for r in known)/len(known),
                       'absent_membership_proxy_rate':rates['absent'],
                       'macro_proxy_rate':sum(rates.values())/len(rates)},
            'caution':'Keyword proxy only; manually audit every raw output, especially absent-term outputs.', 'rows':rows}


if __name__ == '__main__':
    spec=json.loads((Path(__file__).resolve().parent/'advanced_eval.json').read_text())
    responses=json.loads(Path(sys.argv[1]).read_text())
    print(json.dumps(score_all(responses,spec),indent=2))
