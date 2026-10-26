"""Reproducible Lecture 01 Basic/Advanced measurements (12 independent calls).

Run with GEMINI_API_KEY configured securely:
    uv run python -m app.benchmark --output results/lec01-benchmark.json

Each completed attempt is saved before waiting for the next call. Existing output
files are never overwritten. Error durations describe failed attempts, not model
response latency; exclude failed records when comparing successful responses.
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from time import perf_counter, sleep

import httpx
from google import genai
from google.genai import errors, types

from app.config import MODEL, get_api_key

QUESTIONS = (
    "What is LoRA?",
    "What are the 3 homework levels in the AIE2026 course?",
)
TEMPERATURES = (0.2, 0.9)
REPETITIONS = 3
MAX_OUTPUT_TOKENS = 256


def save_results(output: Path, document: dict) -> None:
    """Flush a complete JSON document, then atomically replace the last save."""
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(output)


def run_benchmark(output: Path, *, model: str = MODEL, interval: float = 15) -> int:
    if not math.isfinite(interval) or interval < 0:
        raise ValueError("interval must be a finite nonnegative number")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    key = get_api_key()
    output.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": 1,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "interval_seconds": interval,
        "automatic_retries": False,
        "results": [],
    }
    # Reserve the destination before starting any network requests.
    with output.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2)
    calls = [(q, t, n) for q in QUESTIONS for t in TEMPERATURES for n in range(1, REPETITIONS + 1)]
    failed = False
    with genai.Client(
        api_key=key,
        http_options=types.HttpOptions(
            timeout=60000, retry_options=types.HttpRetryOptions(attempts=1),
        ),
    ) as client:
        for index, (question, temperature, repetition) in enumerate(calls):
            row = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": model,
                "temperature": temperature,
                "repetition": repetition,
                "question": question,
                "response": None,
                "raw_response": None,
                "prompt_tokens": None,
                "candidate_tokens": None,
                "total_tokens": None,
                "thought_tokens": None,
                "error": None,
            }
            start = perf_counter()
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=question,
                    config=types.GenerateContentConfig(
                        temperature=temperature, max_output_tokens=MAX_OUTPUT_TOKENS,
                    ),
                )
            except (errors.APIError, httpx.HTTPError) as error:
                row["elapsed_seconds"] = perf_counter() - start
                row["error"] = {
                    "type": type(error).__name__,
                    "code": getattr(error, "code", None),
                    "message": str(error),
                }
                failed = True
            else:
                row["elapsed_seconds"] = perf_counter() - start
                row["response"] = response.text
                row["raw_response"] = response.model_dump(mode="json", exclude_none=True)
                usage = response.usage_metadata
                if usage is not None:
                    row.update(
                        prompt_tokens=usage.prompt_token_count,
                        candidate_tokens=usage.candidates_token_count,
                        total_tokens=usage.total_token_count,
                        thought_tokens=usage.thoughts_token_count,
                    )
            document["results"].append(row)
            save_results(output, document)
            status = "error" if row["error"] else "ok"
            print(f"[{index + 1}/{len(calls)}] {status} {row['elapsed_seconds']:.3f}s", flush=True)
            if index < len(calls) - 1:
                sleep(interval)
    return 1 if failed else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New JSON output file")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--interval", type=float, default=15, help="Seconds between calls (default: 15)")
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval < 0:
        parser.error("--interval must be a finite nonnegative number")
    try:
        status = run_benchmark(args.output, model=args.model, interval=args.interval)
    except FileExistsError as error:
        parser.error(str(error))
    except KeyboardInterrupt:
        print("\nInterrupted; completed calls remain saved.")
        raise SystemExit(130)
    raise SystemExit(status)


if __name__ == "__main__":
    main()
