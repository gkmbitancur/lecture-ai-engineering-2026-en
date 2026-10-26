"""Offline benchmark checks using real SDK serialization and HTTP handling."""

import importlib
import json

import httpx
import pytest
from google import genai
from google.genai import types


def load_benchmark(monkeypatch):
    try:
        module = importlib.import_module("app.benchmark")
    except ModuleNotFoundError:
        pytest.fail("The benchmark runner has not been implemented")
    monkeypatch.setenv("GEMINI_API_KEY", "offline-test-key")
    return module


def install_transport(benchmark, monkeypatch, handler):
    real_client = genai.Client

    def client(**kwargs):
        options = kwargs["http_options"]
        assert options.retry_options.attempts == 1
        assert options.timeout == 60000
        options.client_args = {"transport": httpx.MockTransport(handler)}
        return real_client(**kwargs)

    monkeypatch.setattr(benchmark.genai, "Client", client)


def success():
    return httpx.Response(200, json={
        "candidates": [{"content": {"role": "model", "parts": [{"text": "Offline fixture"}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 20, "totalTokenCount": 35, "thoughtsTokenCount": 5},
    })


def test_benchmark_records_twelve_independent_calls_and_flushes_each(monkeypatch, tmp_path):
    benchmark = load_benchmark(monkeypatch)
    requests = []
    output = tmp_path / "run.json"
    durable_lengths = []

    def handle(request):
        assert request.url.path.endswith("/models/test-model:generateContent")
        requests.append(json.loads(request.content))
        return success()

    install_transport(benchmark, monkeypatch, handle)
    ticks = iter(value for n in range(12) for value in (100 + n, 100.25 + n))
    monkeypatch.setattr(benchmark, "perf_counter", lambda: next(ticks))

    def sleep(seconds):
        assert seconds == 2
        durable_lengths.append(len(json.loads(output.read_text())["results"]))

    monkeypatch.setattr(benchmark, "sleep", sleep)

    assert benchmark.run_benchmark(output, model="test-model", interval=2) == 0

    results = json.loads(output.read_text())["results"]
    assert len(results) == len(requests) == 12
    assert durable_lengths == list(range(1, 12))
    assert {row["question"] for row in results} == {"What is LoRA?", "What are the 3 homework levels in the AIE2026 course?"}
    assert {row["temperature"] for row in results} == {0.2, 0.9}
    assert {row["repetition"] for row in results} == {1, 2, 3}
    for row, request in zip(results, requests):
        assert request["contents"] == [{"role": "user", "parts": [{"text": row["question"]}]}]
        assert request["generationConfig"]["temperature"] == row["temperature"]
        assert request["generationConfig"]["maxOutputTokens"] == 256
        assert row["model"] == "test-model"
        assert row["elapsed_seconds"] == 0.25
        assert row["response"] == "Offline fixture"
        assert row["prompt_tokens"] == 10
        assert row["candidate_tokens"] == 20
        assert row["total_tokens"] == 35
        assert row["thought_tokens"] == 5
        assert row["timestamp"].endswith("+00:00")
        assert row["error"] is None


def test_failure_is_recorded_once_with_null_usage_and_nonzero_status(monkeypatch, tmp_path):
    benchmark = load_benchmark(monkeypatch)
    count = 0

    def handle(request):
        nonlocal count
        count += 1
        if count == 1:
            return httpx.Response(429, json={"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}})
        return success()

    install_transport(benchmark, monkeypatch, handle)
    ticks = iter(range(24))
    monkeypatch.setattr(benchmark, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(benchmark, "sleep", lambda _: None)
    output = tmp_path / "failures.json"

    assert benchmark.run_benchmark(output, model="test-model", interval=0) == 1

    results = json.loads(output.read_text())["results"]
    assert count == len(results) == 12
    failed = results[0]
    assert failed["response"] is None
    assert failed["elapsed_seconds"] == 1
    assert failed["error"]["code"] == 429
    assert "quota" in failed["error"]["message"]
    assert all(failed[key] is None for key in ("prompt_tokens", "candidate_tokens", "total_tokens", "thought_tokens"))
    assert results[1]["error"] is None


def test_completed_call_survives_interruption_and_existing_output_is_preserved(monkeypatch, tmp_path):
    benchmark = load_benchmark(monkeypatch)
    install_transport(benchmark, monkeypatch, lambda _: success())
    ticks = iter([1.0, 1.5])
    monkeypatch.setattr(benchmark, "perf_counter", lambda: next(ticks))

    def interrupt(_):
        raise KeyboardInterrupt

    monkeypatch.setattr(benchmark, "sleep", interrupt)
    output = tmp_path / "partial.json"
    with pytest.raises(KeyboardInterrupt):
        benchmark.run_benchmark(output, model="test-model", interval=1)
    saved = output.read_text()
    assert len(json.loads(saved)["results"]) == 1
    with pytest.raises(FileExistsError):
        benchmark.run_benchmark(output, model="test-model", interval=1)
    assert output.read_text() == saved
