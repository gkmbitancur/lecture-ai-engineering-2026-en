"""Lecture 03: tools for the agent.

Function docstrings and type hints here become the "tool descriptions" presented to the model.
Rewriting descriptions changes whether the model uses a tool and which arguments it supplies.
"""

import json
from datetime import datetime, timezone
from functools import cache
from pathlib import Path
from uuid import uuid4

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MAX_RESULTS = 3
LEARNING_LOG_DIR = Path(__file__).resolve().parent.parent / "learning_log"


@cache
def load_glossary() -> list[dict]:
    with open(DATA_DIR / "glossary.json", encoding="utf-8") as f:
        return json.load(f)["terms"]


def _normalize(text: str) -> str:
    return text.casefold().replace(" ", "").replace("　", "")


def _names(entry: dict) -> list[str]:
    return [_normalize(entry["term"]), _normalize(entry["id"])] + [
        _normalize(a) for a in entry["aliases"]
    ]


def search_glossary(query: str) -> str:
    """Search the course glossary and return definitions and the sessions that cover the terms.

    Use before answering about AI engineering terms or words specific to this course (AIE2026).
    Put only one term to look up in query (for example, "LoRA").
    """
    q = _normalize(query)
    if not q:
        return "The search query is empty. Specify one term to look up."
    glossary = load_glossary()

    exact = [e for e in glossary if q in _names(e)]
    # Exclude short names so aliases such as "r" do not match everything
    partial = [
        e for e in glossary
        if e not in exact and any(q in n or (len(n) >= 3 and n in q) for n in _names(e))
    ]
    in_definition = [
        e for e in glossary
        if e not in exact and e not in partial and q in _normalize(e["definition"])
    ]
    hits = (exact + partial + in_definition)[:MAX_RESULTS]

    if not hits:
        return f"The term '{query}' was not found in the glossary."
    return json.dumps(
        [
            {
                "term": e["term"],
                "lecture": e["lecture"],
                "definition": e["definition"],
                "related": e["related"],
            }
            for e in hits
        ],
        ensure_ascii=False,
    )


def get_lecture_summary(lecture: int) -> str:
    """Return key points from the specified session (an integer starting at 1).

    Use when asked about session content or relationships, such as "What will we learn in Lecture 03?".
    """
    path = DATA_DIR / "lectures" / f"{lecture:02d}.md"
    if not path.exists():
        available = sorted(p.stem for p in (DATA_DIR / "lectures").glob("*.md"))
        return f"Key points for Lecture {lecture:02d} are not available yet. Available sessions: {', '.join(available)}"
    return path.read_text(encoding="utf-8")


def make_quiz(lecture: int) -> str:
    """Create one review question and its answer from a specified lecture's glossary.

    Use when a learner requests practice or a review question. The answer comes
    directly from the course glossary; choose the first matching term consistently.
    """
    entries = [entry for entry in load_glossary() if entry["lecture"] == lecture]
    if not entries:
        return f"A quiz for Lecture {lecture:02d} is not available in the glossary."
    entry = entries[0]
    return json.dumps({"lecture": lecture, "term": entry["term"],
        "question": f"What does {entry['term']} mean?", "answer": entry["definition"]}, ensure_ascii=False)


def save_learning_log(lecture: int, topic: str, reflection: str) -> str:
    """Save a learner-requested study reflection after the CLI obtains approval.

    Use only when asked to record learning. Pass the session number, a short topic,
    and the reflection to save. The harness pauses before this function executes.
    Each approved call creates a new JSON record; it never overwrites older logs.
    """
    if lecture < 1 or not topic.strip() or not reflection.strip():
        raise ValueError("A positive lecture number, topic, and reflection are required.")
    if len(topic) > 200 or len(reflection) > 4000:
        raise ValueError("Topic must be at most 200 characters and reflection at most 4000.")
    record = {"lecture": lecture, "topic": topic.strip(), "reflection": reflection.strip(),
        "created_at": datetime.now(timezone.utc).isoformat()}
    LEARNING_LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LEARNING_LOG_DIR / f"{uuid4().hex}.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return f"Saved learning record to learning_log/{path.name}"
