"""Exercise the CLI with the real Gemini SDK and an offline HTTP transport."""

import json

import httpx
import pytest
from google import genai
from google.genai import types

from app import ask


def response(text="A small adapter.", usage=None):
    payload = {
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": text}]},
            "finishReason": "STOP",
        }]
    }
    if usage is not None:
        payload["usageMetadata"] = usage
    return httpx.Response(200, json=payload)


@pytest.fixture
def gemini(monkeypatch):
    requests, replies = [], []

    def handle(request):
        requests.append(json.loads(request.content))
        return replies.pop(0)

    client = genai.Client(
        api_key="offline-test-key",
        http_options=types.HttpOptions(
            client_args={"transport": httpx.MockTransport(handle)},
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )
    monkeypatch.setenv("GEMINI_API_KEY", "offline-test-key")
    monkeypatch.setattr(ask.genai, "Client", lambda **kwargs: client)
    yield requests, replies
    client.close()


@pytest.mark.parametrize("question", [None, "What is LoRA?"])
def test_one_shot_keeps_default_and_explicit_questions(question, gemini, monkeypatch, capsys):
    requests, replies = gemini
    replies.append(response(usage={"promptTokenCount": 12, "candidatesTokenCount": 5}))
    monkeypatch.setattr("sys.argv", ["ask"] + ([question] if question else []))

    ask.main()

    expected = question or "What are the 3 homework levels in the AIE2026 course?"
    assert requests[0]["contents"] == [{"role": "user", "parts": [{"text": expected}]}]
    output = capsys.readouterr().out
    assert "A small adapter." in output
    assert "[tokens] input=12 output=5" in output


@pytest.mark.parametrize("usage, expected", [
    (None, None),
    ({"promptTokenCount": 0}, "[tokens] input=0 output=n/a"),
])
def test_token_metadata_can_be_missing_or_partial(usage, expected, gemini, capsys):
    _, replies = gemini
    replies.append(response(usage=usage))

    assert ask.ask("Hello") == "A small adapter."

    output = capsys.readouterr().out
    if expected is None:
        assert "[tokens]" not in output
    else:
        assert expected in output


def test_chat_sends_history_skips_blanks_and_excludes_failed_turn(gemini, monkeypatch, capsys):
    requests, replies = gemini
    replies.extend([
        response("LoRA trains adapters.", {"promptTokenCount": 6, "candidatesTokenCount": 4}),
        httpx.Response(429, json={"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}}),
        response("The base weights stay frozen."),
    ])
    prompts = iter(["   ", "Failed question", "What stays frozen?", "quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(prompts))
    monkeypatch.setattr("sys.argv", ["ask", "--chat", "What is LoRA?"])

    ask.main()

    assert len(requests) == 3
    assert requests[1]["contents"] == [
        {"role": "user", "parts": [{"text": "What is LoRA?"}]},
        {"role": "model", "parts": [{"text": "LoRA trains adapters."}]},
        {"role": "user", "parts": [{"text": "Failed question"}]},
    ]
    assert requests[2]["contents"] == [
        {"role": "user", "parts": [{"text": "What is LoRA?"}]},
        {"role": "model", "parts": [{"text": "LoRA trains adapters."}]},
        {"role": "user", "parts": [{"text": "What stays frozen?"}]},
    ]
    output = capsys.readouterr().out
    assert "[tokens] input=6 output=4" in output
    assert "Rate limit reached" in output
    assert "The base weights stay frozen." in output


@pytest.mark.parametrize("stop", [EOFError, KeyboardInterrupt])
def test_chat_without_initial_question_exits_cleanly(stop, gemini, monkeypatch):
    requests, _ = gemini

    def end_input(_):
        raise stop

    monkeypatch.setattr("builtins.input", end_input)
    monkeypatch.setattr("sys.argv", ["ask", "--chat"])

    ask.main()

    assert requests == []


def test_one_shot_api_failure_exits_nonzero(gemini, monkeypatch, capsys):
    _, replies = gemini
    replies.append(httpx.Response(403, json={"error": {"code": 403, "message": "denied", "status": "PERMISSION_DENIED"}}))
    monkeypatch.setattr("sys.argv", ["ask", "Hello"])

    with pytest.raises(SystemExit) as exc:
        ask.main()

    assert exc.value.code == 1
    assert "API key or model name (403)" in capsys.readouterr().out
