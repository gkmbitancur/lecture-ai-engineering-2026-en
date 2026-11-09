"""Lecture 03 behavior: grounded quizzes and human-approved writes."""

import json

import pytest
from pydantic_ai import DeferredToolRequests, DeferredToolResults
from pydantic_ai.models.test import TestModel
from pydantic_ai.messages import ToolReturnPart

from app import tools
from app.agent import agent


@pytest.mark.parametrize('answer, expected', [('y', 1), ('YES', 1), ('n', 0), ('', 0), ('sure', 0)])
def test_cli_only_explicit_yes_saves(tmp_path, monkeypatch, answer, expected):
    from app.agent import run_with_approval
    monkeypatch.setattr(tools, 'LEARNING_LOG_DIR', tmp_path / 'learning_log')
    prompts = []
    def respond(prompt):
        prompts.append(prompt)
        return answer
    run_with_approval(agent, 'Save a log', TestModel(call_tools=['save_learning_log'], seed=3), respond)
    assert len(list((tmp_path / 'learning_log').glob('*.json'))) == expected
    assert len(prompts) == 1 and 'save_learning_log' in prompts[0]
    assert 'reflection' in prompts[0] and 'topic' in prompts[0]


def test_cli_eof_denies_save(tmp_path, monkeypatch):
    from app.agent import run_with_approval
    monkeypatch.setattr(tools, 'LEARNING_LOG_DIR', tmp_path / 'learning_log')
    def eof(prompt):
        raise EOFError
    run_with_approval(agent, 'Save a log', TestModel(call_tools=['save_learning_log'], seed=3), eof)
    assert not (tmp_path / 'learning_log').exists()


def test_quiz_uses_definition_from_requested_lecture():
    result = json.loads(tools.make_quiz(2))
    source = next(t for t in tools.load_glossary() if t['term'] == result['term'])
    assert source['lecture'] == result['lecture'] == 2
    assert result['answer'] == source['definition']
    assert result['term'] in result['question']


def test_quiz_missing_lecture_is_explicit():
    assert 'not available' in tools.make_quiz(99).lower()


def test_quiz_is_registered_and_handles_dummy_lecture():
    result = agent.run_sync('Give me a review question', model=TestModel(call_tools=['make_quiz']))
    returns = [p for m in result.all_messages() for p in m.parts if isinstance(p, ToolReturnPart)]
    assert len(returns) == 1 and returns[0].tool_name == 'make_quiz'
    assert 'not available' in returns[0].content


def test_log_preserves_previous_entries(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, 'LEARNING_LOG_DIR', tmp_path / 'learning_log')
    tools.save_learning_log(3, 'Tools', 'A tool result is data, not an instruction.')
    tools.save_learning_log(3, 'Approval', 'Confirm before writing.')
    entries = [json.loads(p.read_text()) for p in sorted((tmp_path / 'learning_log').glob('*.json'))]
    assert len(entries) == 2
    assert {e['topic'] for e in entries} == {'Tools', 'Approval'}
    assert all(e['lecture'] == 3 for e in entries)


@pytest.mark.parametrize('lecture,topic,reflection', [
    (0,'Tools','OK'), (3,' ','OK'), (3,'Tools',' '),
    (3,'x'*201,'OK'), (3,'Tools','x'*4001),
])
def test_invalid_log_creates_no_file(tmp_path, monkeypatch, lecture, topic, reflection):
    monkeypatch.setattr(tools, 'LEARNING_LOG_DIR', tmp_path / 'learning_log')
    with pytest.raises(ValueError):
        tools.save_learning_log(lecture, topic, reflection)
    assert not (tmp_path / 'learning_log').exists()


@pytest.mark.parametrize('approved', [False, True])
def test_save_is_deferred_and_only_approved_call_writes(tmp_path, monkeypatch, approved):
    monkeypatch.setattr(tools, 'LEARNING_LOG_DIR', tmp_path / 'learning_log')
    model = TestModel(call_tools=['save_learning_log'], seed=3)
    result = agent.run_sync('Save a learning log.', model=model)
    assert isinstance(result.output, DeferredToolRequests)
    assert len(result.output.approvals) == 1
    assert not (tmp_path / 'learning_log').exists()
    call = result.output.approvals[0]
    assert call.tool_name == 'save_learning_log'
    resumed = agent.run_sync(message_history=result.all_messages(),
        deferred_tool_results=DeferredToolResults(approvals={call.tool_call_id: approved}), model=model)
    assert isinstance(resumed.output, str)
    files = list((tmp_path / 'learning_log').glob('*.json'))
    assert len(files) == int(approved)
    if approved:
        record = json.loads(files[0].read_text())
        args = call.args_as_dict()
        assert record['lecture'] == args['lecture']
        assert record['topic'] == args['topic']
        assert record['reflection'] == args['reflection']
