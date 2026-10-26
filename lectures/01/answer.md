# Lecture 01 Answers

## Exercises

- Model's answer to the course-specific question (the 3 homework levels; a summary is fine): All six live responses said they lacked specific information about AIE2026 rather than listing three levels. Two low-temperature responses simply declined; the other four also requested a syllabus, institution, or other context. The full responses are saved in [the benchmark data](../../results/lec01-benchmark.json).
- What you noticed when comparing it with the correct meaning in the glossary: `homework-levels` in `data/glossary.json` defines Basic as reviewing lecture content, Implementation as writing code, and Advanced as research and experimentation. Completing any one level completes that lecture's homework. None of the six course responses supplied this answer, but none invented three different levels either. This run demonstrated appropriate uncertainty about the requested course facts, rather than a fabricated course structure. The model's statements about what is in its training data are not independently verified.
- Token counts you displayed (input / output): `app/ask.py` displays reported prompt and candidate token counts, with absent metadata handled explicitly. In the live benchmark, every LoRA response reported **6 input / 252 candidate tokens**. The course responses reported **18 input tokens** each and **40, 43, 73, 72, 71, and 110 candidate tokens**, respectively. These are measured SDK usage values, not estimated word counts. Thought-token metadata was absent in every response and remains null.

## Homework

- Level attempted: **Basic, Implementation, and Advanced.** The 12-call live experiment is complete; the chat implementation is tested offline. This does not assert that the learner has personally reviewed the generated work.
- What you did:

  **Implementation.** Added `--chat` to `app/ask.py` using one Google Gen AI SDK chat session. It retains successful user questions and model responses for follow-up questions. "Conversation history" means messages inside this application's chat session; it does not mean the Codex transcript, browser activity, or other chats. History stays in memory until exit. Blank prompts are skipped; `quit`, `exit`, EOF, and Ctrl-C exit cleanly. API errors display the existing friendly explanation and do not enter subsequent history. One-shot mode and its default question remain compatible. No runtime dependencies were added.

  ```bash
  uv run python -m app.ask "What is LoRA?"
  uv run python -m app.ask --chat "What is LoRA?"
  # At the prompt, try: What stays frozen during training?
  ```

  **Basic and Advanced experiment.** Ran **gemini-3.5-flash-lite** on two questions ("What is LoRA?" and "What are the 3 homework levels in the AIE2026 course?"), at temperatures **0.2 and 0.9**, with **3 independent repetitions per cell: 12 calls**. Each request contained only its own question, with no conversation history or glossary. The output-token setting was 256. Automatic SDK retries were disabled; each request had a 60-second timeout and a 15-second pause before the next call. This varies temperature and question, rather than comparing different models.

  ```bash
  # Configure GEMINI_API_KEY securely in the process environment.
  uv run python -m app.benchmark --model gemini-3.5-flash-lite --interval 15 --output results/lec01-benchmark.json
  ```

  The runner saves each completed attempt as a flushed, atomically replaced JSON document and refuses an existing output file. Records include prompt, response text, full SDK response, model, temperature, repetition, UTC timestamp, duration, token metadata, and errors. API/HTTP failures retain null response/usage fields and produce a nonzero final exit status; completed records survive interruption.

- Results:

  **Live measurements.** All 12 requests succeeded, with **0 API/HTTP errors**. The run began at **2026-10-26 07:47:32 UTC** (15:47:32 in Manila) and finished at approximately **07:50:37 UTC**, including deliberate pauses. The four cells are below; token columns show median and range across their three repetitions.

  | Question | Temperature | Latency median, seconds (range) | Input tokens, median (range) | Candidate tokens, median (range) | Total tokens, median (range) | Finish reasons |
  |---|---:|---:|---:|---:|---:|---|
  | LoRA | 0.2 | 2.048 (1.608–2.220) | 6 (6–6) | 252 (252–252) | 258 (258–258) | 3 MAX_TOKENS |
  | LoRA | 0.9 | 1.744 (1.685–2.078) | 6 (6–6) | 252 (252–252) | 258 (258–258) | 3 MAX_TOKENS |
  | Course levels | 0.2 | 1.239 (1.134–1.626) | 18 (18–18) | 43 (40–73) | 61 (58–91) | 3 STOP |
  | Course levels | 0.9 | 1.189 (1.093–1.638) | 18 (18–18) | 72 (71–110) | 90 (89–128) | 3 STOP |

  Totals: **144 input tokens, 1,921 candidate tokens, and 2,065 total tokens**. The sum of measured request durations was **19.301 seconds**; total experiment wall time was about **184.4 seconds** with pauses. No thought-token count was reported, so no zero-thought-token claim is made.

  **Basic output comparison.** All six LoRA responses expanded the acronym correctly and described efficient adaptation, but **every LoRA response was truncated**: the SDK reported `MAX_TOKENS`, despite recorded candidate counts of 252 under the requested 256-token cap. Their visible content cannot be treated as complete technical explanations. The individual differences were:

  | Question/temperature | Repetition 1 | Repetition 2 | Repetition 3 |
  |---|---|---|---|
  | LoRA / 0.2 | Textbook/sticky-note analogy; cut off as the mechanism begins. | Emphasized full-fine-tuning costs; cut off before finishing its explanation. | Reached frozen base parameters and small trainable adapters before truncation. |
  | LoRA / 0.9 | Cost/example discussion and an origin attribution; cut off before the mechanism. | GPU/storage cost list; cut off at the next section. | Described frozen base parameters and a trainable "sidecar"; cut off at the next heading. |
  | Course / 0.2 | Declined because specific course information was unavailable. | Similar concise uncertainty statement with different wording. | Longer uncertainty statement and a request for institution/platform/instructor context. |
  | Course / 0.9 | Requested syllabus/institution/platform context. | Requested syllabus/course-description context with similar structure. | Longer uncertainty statement; additionally asserted a January 2025 training cutoff, which this experiment does not verify. |

  The LoRA glossary entry specifically describes frozen weights and low-rank trainable matrices. Two visible responses reached freezing/adapters, but truncation prevented a complete comparison of that mechanism across all six. Some responses also grouped Stable Diffusion with LLM examples, so their model-family wording needs care. The higher-temperature course cell had a larger median output (72 versus 43 tokens), but all six course answers expressed uncertainty. This small run does not show that higher temperature caused hallucination or improved factual accuracy.

  **Advanced interpretation.** `elapsed_seconds` measures `perf_counter` wall time around the SDK request, including network/service time and SDK response handling; it excludes the deliberate interval and file writing. It is neither time to first token nor isolated backend compute time. Shorter course responses had lower median latency than capped LoRA responses even though the course prompt used more input tokens. That is an association with question/output length, not proof that token count alone determines latency. Temperature-cell latency ranges overlap. Three repetitions and fixed execution order cannot separate temperature effects from time-varying service load. The output cap also hides natural LoRA response-length variation.

  **Verification.** Real-SDK offline transport tests cover serialized history, failed-turn exclusion, token display, CLI compatibility, clean exits, benchmark configuration, disabled retries, timeout, duration measurement, failure records, and durable output. The final complete repository suite reported **22 passed in 2.33s**. An initial sandbox run failed two existing MCP tests because their subprocess could not access the local dependency cache; allowing that access resolved the environment failure. Benchmark tests were written first and observed failing before implementation. `git diff --check` passed. The live experiment validates benchmark calls; interactive chat history was validated through offline request inspection rather than a separate live chat session.

- Observations and questions:

  Retaining chat history grows later input and can consume more context and tokens. It does not supply external facts or guarantee accuracy. In this live run, the bare model appropriately withheld the requested course facts; adding the glossary as retrieved context would address the missing source, while keeping history alone would not.

  **Free-tier rate limits.** Google's documentation describes requests per minute (RPM), input tokens per minute (TPM), and requests per day (RPD), applied per project rather than per key. Limits depend on model and project tier; RPD resets at midnight Pacific time. Exceeding an applicable limit can cause an error. [Source: Google Gemini API rate limits](https://ai.google.dev/gemini-api/docs/rate-limits), checked 2026-10-26.

  AI Studio showed this project's free-tier limits for `gemini-3.5-flash-lite` on **2026-10-26** as **15 RPM, 250,000 input TPM, and 500 RPD**. These are account/project observations, not universal or permanent quotas. This run used **12/500 = 2.4%** of that daily request allowance. From saved request-start timestamps, the busiest rolling 60-second interval contained **4 requests and 72 input tokens** (about **26.7% of RPM** and **0.0288% of input TPM**). Across the entire run there were 144 input tokens. Other project activity shares those limits, and local start-time windows are estimates of server-side quota windows. No 429 errors occurred; the pauses do not guarantee future quota compliance. Output/thought-token counts do not substitute for input TPM when assessing that limit.

  **AI assistance disclosure.** OpenAI Codex and a delegated Codex subagent assisted with implementation, testing, the live benchmark workflow, and this report. This does not claim that the learner personally reviewed or understood the generated work. GitHub Copilot and Codespaces were not used for the reported local work. The saved live responses and measurements are separate from the explicitly supplied offline test fixtures.
