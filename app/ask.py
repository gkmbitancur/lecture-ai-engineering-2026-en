"""Lecture 01: ask a bare LLM one question or keep a conversation.

Usage:
    uv run python -m app.ask "What is LoRA?"
    uv run python -m app.ask            # Ask a course-specific question if none is supplied
    uv run python -m app.ask --chat "What is LoRA?"
    uv run python -m app.ask --chat     # Prompt for the first question

A "bare" call with no tools. Chat mode keeps history in memory until exit.
Check that, when asked about course-specific information, it either says "I don't know"
or gives a plausible general answer (which differs from the actual course).
(In Lecture 03, improve it by making an agent that can search the glossary.)
"""

import argparse

from google import genai
from google.genai import errors, types

from app.config import MODEL, api_error_message, get_api_key

# A course-specific question. The correct answer is in homework-levels in data/glossary.json
DEFAULT_QUESTION = "What are the 3 homework levels in the AIE2026 course?"


def print_token_counts(response: types.GenerateContentResponse) -> None:
    """Display reported usage without treating a missing count as zero."""
    usage = response.usage_metadata
    if usage is not None:
        input_tokens = usage.prompt_token_count
        output_tokens = usage.candidates_token_count
        print(
            f"[tokens] input={input_tokens if input_tokens is not None else 'n/a'} "
            f"output={output_tokens if output_tokens is not None else 'n/a'}"
        )


def ask(question: str) -> str:
    with genai.Client(api_key=get_api_key()) as client:
        response = client.models.generate_content(model=MODEL, contents=question)
    print_token_counts(response)
    return response.text


def chat(question: str | None = None) -> None:
    """Keep successful turns in one SDK chat session for follow-up questions."""
    print("[chat] Type quit or exit to finish (Ctrl-D/Ctrl-C also exits).")
    with genai.Client(api_key=get_api_key()) as client:
        session = client.chats.create(model=MODEL)
        try:
            while True:
                if question is None:
                    question = input("You: ")
                prompt = question.strip()
                question = None
                if prompt.lower() in {"quit", "exit"}:
                    return
                if not prompt:
                    continue
                try:
                    # The SDK records the turn only after a successful request.
                    response = session.send_message(prompt)
                except errors.APIError as e:
                    print(api_error_message(e.code, e.message))
                    continue
                print_token_counts(response)
                print(response.text)
        except (EOFError, KeyboardInterrupt):
            print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a bare LLM a question")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--chat", action="store_true", help="Keep history for follow-up questions")
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    if args.chat:
        chat(args.question)
        return
    question = args.question if args.question is not None else DEFAULT_QUESTION
    print(f"[question] {question}\n")
    try:
        print(ask(question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
