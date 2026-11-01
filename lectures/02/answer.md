# Lecture 02 Answers

The original exercise and all four runner scenarios were executed. This report contains measured results and completed verification.

## Exercises

### Original notebook run and method

The original notebook was executed with `Qwen/Qwen3.5-0.8B`, revision `2fc06364715b967f1860aea9cf38778875588b17`, on a Colab T4 using LoRA without quantization. The configuration recorded seed 42, 20 epochs, learning rate 0.0002, rank 16, alpha 32, warmup 6 steps, maximum sequence length 512, maximum generated tokens 96, and greedy decoding. The source uses batch size 4 and gradient accumulation 2. Fourteen training examples produced 40 optimizer steps. Versions recorded in the result are transformers 5.18.0, peft 0.21.1, trl 1.14.2, and torch 2.11.0+cu130. `executed_training_myself` is true because training was actually executed in this workflow rather than replaced by instructor results.

The fictional course FAQ contains five facts. Training has ten grounded and four ungrounded examples; validation has five grounded and two ungrounded examples. Test has fifteen grounded and six ungrounded questions. Different question strings are assigned to each split, although the grounded facts are shared. The FAQ is supplied at inference, so improved scores do not establish that the facts were newly learned in model weights. The test measures question generalization and evidence-conditioned answer behavior on this small FAQ.

The original checker treats format and content separately. Format requires a JSON object containing only `answer` and `needs_review`, with string and Boolean values. Content requires the correct review flag and all specified answer substrings. If a JSON object can be recovered from a malformed response, content can still pass. This separation is useful, but content is a keyword proxy: spelling variants can fail, and a response can contain the keywords while still making an unsupported claim.



Prepared source snapshot SHA-256 values, also recorded in `prepared/data_notes.txt`:

- `notebooks/lora_format_tuning.ipynb`: `6f92482023aac18149265cf35e1b3ef6fa6e34a335d9dfa8f9ee5e1d3383d807`
- `scripts/make_sft_dataset.py`: `fb0c571558cf29fd6c4cc15add61128542f25f15bdd1a13b0e789bdb1b58cc41`
- `data/glossary.json`: `d13b2497fb1ac10612c88683e7e1934a200790732a0157cf5acac9e6fc5441ba`

### Measured original exercise results

| Original notebook condition | Format rate | Content rate | Both rate | Test generation time |
|---|---:|---:|---:|---:|
| Before LoRA | 20/21 = 95.24% | 8/21 = 38.10% | 8/21 = 38.10% | 56.1 s |
| After 20 epochs | 21/21 = 100.00% | 19/21 = 90.48% | 19/21 = 90.48% | 35.9 s |

There were no questions that passed both checks before training and failed afterward. The final logged loss was **0.0144890666 at step 40**. This is the last logged interval-averaged loss, not a final-example loss. The separately reported whole-run `train_loss` was **0.4477807306**. Trainer recorded **181.6573 seconds** of training runtime. Loss declined across the logged steps 10, 20, 30, and 40: approximately 1.69353, 0.06492, 0.01818, and 0.01449. A falling training loss alone does not establish reliable handling of unknown questions.

Two failures remained after training. For “Can I submit a format other than PDF?”, the output was:

```json
{"answer": "This FAQ does not contain that information. Confirmation from the course organizers is needed.", "needs_review": true}
```

This is evasion despite evidence: the FAQ specifies one PDF file, so the expected flag is false. For “Do office hours require an appointment?”, the output was:

```json
{"answer": "Questions are welcome during office hours every Tuesday from 17:00 to 18:00.", "needs_review": false}
```

The office-hours schedule does not establish an appointment policy. This is an unsupported assertion produced by substituting a related FAQ fact for the missing answer. Both outputs have valid JSON; neither has correct content under the stated task.

Training did correct several unsupported answers. Before training, “Will I receive feedback on my submitted assignment?” received “Yes, questions are welcome during office hours every Tuesday from 17:00 to 18:00.” with `needs_review:false`. After training it received the explicit confirmation response with `needs_review:true`. However, some apparent improvements also reflect the literal rubric. The pretraining response to the character-count question correctly said that no limit was mentioned and set the flag true, yet failed because it omitted the required word “Confirmation.” Automated gains should therefore be checked against raw outputs.

### Knowledge probes and their ambiguity

| Original question | Before classification | After classification | Interpretation |
|---|---|---|---|
| Final assignment deadline for this course? | asserted | asserted | Generic assignment deadline repeated; final-deadline specificity is unproven |
| Instructor's name? | needs_review | needs_review | Missing fact recognized; after-training wording became explicit |
| Topic of the next lecture? | needs_review | needs_review | Missing fact recognized before and after |
| Assignment deadline? | asserted | asserted | Grounded control answered using the FAQ |

The FAQ is included in these probe inputs. In particular, `make_example(q, "", True)` leaves the supplied FAQ intact; the empty string is the unused target answer. The first probe is labeled outside the FAQ by the notebook, but “final assignment” can be interpreted as the assignment whose deadline is listed. Both outputs gave September 30 at 23:59 Japan time. That response does not prove the model knows a distinct final-assignment deadline. It is an ambiguous probe and should not be presented as definitive hallucination evidence.

### Optional 4-bit loading observation

A separate before-training model loaded in 4-bit used **0.76 GB** versus **3.01 GB** for the unquantized model, measured with the notebook's model-footprint method. Its format rate was 21/21, content rate 1/21, and generation time 35.7 seconds. These measurements concern loading and inference only; no QLoRA training was performed. The large content decrease is an observation from this one run, not a general estimate of quantization's effect. The [saved 4-bit outputs](results/original_exercise/eval_before_4bit.json) set `needs_review:true` on all 21 questions. Nine grounded responses retain every required answer keyword but fail on the flag alone. The sole pass correctly requests confirmation about grade release; five other unknown questions receive unsupported yes/no answers despite the true flag.

## Homework

### What was verified

The following checks were completed on the executed work. The [verification record](results/verification.json) contains the results, and [verify_results.py](verify_results.py) reproduces the numerical and consistency checks without retraining.

1. **The experiments actually ran under the intended conditions.** All four saved manifests report completion on a Tesla T4 and have distinct process IDs. The Basic configuration differs from its matched baseline only in epochs, 20 versus 10; all 21 before-training answers match exactly. Implementation and Advanced retain the baseline training configuration. All 32 recorded source checksums match the archived versions, including the preserved earlier runner.
2. **Training and evaluation data were separated as described.** Each FAQ dataset has 14 training, 7 validation, and 21 test questions, with 42 unique question strings and six unknown test questions. Regenerating the glossary data with the course script produced exactly the archived 116 records from 58 entries. The 24 Advanced questions are absent from the training questions and split into eight named terms, eight paraphrases, and eight absent terms. The known IDs and absent term names were checked against the glossary. The facts themselves are shared across the FAQ splits, and the known glossary concepts appear in training; this is not held-out-fact evaluation.
3. **The numbers agree with the saved evidence.** Scores were recomputed from all 189 FAQ responses, all 48 Advanced model responses, and all 24 search responses. Every saved judgment and summary matched the frozen scoring rules. The search outputs were also reproduced from the question and glossary alone. Final logged loss, its step, aggregate training loss, and training timing were cross-checked against the four training logs. All eight runner checks passed, including checks for answer leakage, process reuse, output preservation, and separation of format from content.
4. **Raw answers were checked beyond their scores.** The [Basic audit](results/basic/comparison_audit.txt) confirms that its net one-question decrease hides two losses and one gain. The [Implementation audit](results/implementation/comparison_audit.txt) identifies a monthly-reservation question incorrectly answered with a simultaneous-item limit despite zero automated regressions. The Advanced audits cover all 48 model answers and 24 search responses. They identify both scoring mistakes and factual errors: “HTTP/2” incorrectly passes the MCP keyword test before training, while the after-training cosine answer incorrectly restricts similarity to 0–1. The [absent-term audit](results/advanced/absent-semantic-audit.json) links primary references used to check those technical claims.
5. **The conclusions reflect what the checks support.** Basic roughly halves training time while content changes from 19/21 to 18/21. Implementation improves from 6/21 to 18/21 but retains unsupported answers. Advanced improves named-term scores from 5/8 to 7/8 and worsens paraphrases from 4/8 to 2/8, leaving known-content at 9/16. These results do not support an overall Advanced improvement. The absent-term score measures glossary membership, not factual accuracy, because the model is not given the glossary inventory. The qualitative judgments supplement the fixed automated rubric; one seed and these small test sets do not establish broad generalization.

### Levels attempted and common protocol

This draft covers **Basic, Implementation, and Advanced**, although completing any one level satisfies the homework requirement. The original exercise above remains a separate experiment. Basic's causal comparison uses a newly matched runner baseline and a changed-epoch run, both started in fresh Python processes. This avoids silently comparing a seeded runner run with an original notebook whose random-state setup may differ.

The runner reads the original notebook's configuration, FAQ splitting, system prompt, chat construction, and checkers. All scenarios use the same pinned model revision, unquantized float32 weights, LoRA targets (`q/k/v/o_proj` and `gate/up/down_proj`), dropout 0.05, alpha 32, batch size 4, accumulation 2, cosine schedule, warmup 6, greedy decoding, and token limits. Seed 42 is applied before model loading and training in each runner process. These controls reduce variation but do not make GPU execution perfectly reproducible. The first three runner scenarios used the source now preserved as `run_experiments_v1.py`; their manifest hashes refer to that exact file content. Before Advanced started, its epoch override was removed so Advanced also retains the original 20 epochs, following the instruction to keep other conditions unchanged. This is the only runner change; datasets and the scoring rubric stayed fixed. The Advanced runner SHA-256 is `aae3d486aadfe1d72aa4ae91f64f202d3ba43f6c95ddb01f528923628ac9bcfa`.

Each process saves a manifest with source hashes, package versions, GPU information, seed, and configuration; raw before/after outputs; regression IDs and questions; training logs and timing; trainable and total parameter counts; and the adapter. The final logged interval loss is reported separately from Trainer's aggregate training loss. Existing nonempty output directories are rejected to preserve previous evidence. All completed sections below use inspected saved artifacts. Adapters were saved in Colab; the repository archives contain the small result files and audits rather than adapter weights.

### Basic: changing only the number of epochs

The matched baseline uses 20 epochs; Basic uses 10. No other model or training setting changes. Both runs use the original course FAQ and its identical 14/7/21 split. The purpose is to test whether half as many passes retain format compliance and content performance while reducing training time. Fewer steps may change how much of the fixed warmup schedule occupies training; that follows directly from the single epoch-count intervention.

| Matched runner condition | Before format/content | After format/content | Final logged loss | Mean train loss | Training time |
|---|---|---|---|---|---|
| [Baseline, 20 epochs](results/baseline/result.json) | 20/21 (95.24%) / 8/21 (38.10%) | 21/21 (100%) / 19/21 (90.48%) | 0.01479294, step 40 | 0.4510146 | 187.508 s |
| [Basic, 10 epochs](results/basic/result.json) | 20/21 (95.24%) / 8/21 (38.10%) | 21/21 (100%) / 18/21 (85.71%) | 0.07120965, step 20 | 0.8827261 | 94.444 s |

All 21 before-training raw outputs are identical across the matched runs; both have zero regressions relative to their own before outputs. Halving epochs reduced training time by 49.63%, preserved perfect format compliance, and reduced content by a net one question. The changed failure set matters: 10 epochs unnecessarily requests confirmation for the non-PDF and missed-deadline questions, which 20 epochs answers correctly, but correctly abstains on feedback, which 20 epochs answers with an unrelated office-hours schedule. Both fail the appointment question. For non-PDF submissions, the 20-epoch output is `{"answer": "Submit one PDF file named student_id_assignment_number.pdf.", "needs_review": false}`; the 10-epoch output instead says `{"answer": "This FAQ does not contain that information. Confirmation from the course organizers is needed.", "needs_review": true}`. These are two losses and one gain, rather than one additional failure. [All raw comparisons and manual interpretations](results/basic/comparison_audit.txt) are preserved alongside the [20-epoch](results/baseline/eval_after.json) and [10-epoch](results/basic/eval_after.json) outputs. Both models train 6,389,760 of 758,782,784 parameters (0.8421%). The original exercise remains separate: its 19/21 score has a different failure set. This one-seed, 21-question result does not establish a robust epoch optimum.

### Implementation: fictional equipment lending FAQ

The replacement topic is a fictional hiking club's equipment lending rules. Its five non-safety facts are Wednesday 18:00–19:00 pickup in the club room, a standard seven-calendar-day loan, booking through the club lending portal's Equipment Reservations section, a maximum of three items per member simultaneously, and Monday 17:00–18:00 returns in the club room. These invented administrative rules were selected to have distinct answers that can be checked without making claims about real equipment use.

The dataset preserves the notebook's JSON output and review-flag design. It has 14 training, 7 validation, and 21 test examples; test contains 15 grounded and 6 ungrounded questions. Each fact has two training forms, one validation form, and three test forms. Forty-two question strings are unique across the splits. This keeps question wording separate while supplying the same five-fact evidence at inference time.

Ungrounded questions were prepared by listing plausible adjacent policies that none of the five facts entails: renewals, deposits, nonmember access, late returns, cancellations, proxy collection, booking horizon, waiting lists, winter closure, reservation transfers, delivery, and monthly reservation limits. Ordinary assumptions about clubs were not treated as evidence. Four unknowns appear in training, two in validation, and six in test; four of those test-only questions supplement the original cyclic split. Their target explicitly requests coordinator confirmation and sets `needs_review:true`.

| Implementation, 20 epochs | Format rate | Content rate | Both rate | Training time / final loss |
|---|---|---|---|---|
| Before training | 21/21 = 100% | 6/21 = 28.57% | 6/21 = 28.57% | — |
| [After training](results/implementation/result.json) | 21/21 = 100% | 18/21 = 85.71% | 18/21 = 85.71% | 186.807 s / 0.01642460, step 40 |

Aggregate `train_loss` was 0.44956496; 40 optimizer steps completed. After training, 14/15 grounded and 4/6 unknown questions pass. For “How should a member submit a gear reservation?”, the correct booking text already appeared before training but incorrectly set `needs_review:true`; after training, `{"answer": "Reserve equipment through the club lending portal under Equipment Reservations.", "needs_review": false}` correctly couples evidence and flag. For the test-only delivery question, the before answer `{"answer": "No", "needs_review": true}` became `{"answer": "This FAQ does not contain that information. Confirmation from the club lending coordinator is needed.", "needs_review": true}`. The flag alone had concealed an unsupported denial.

The automated regression count is zero, but manual inspection finds substantive deterioration hidden by keyword failures. The Wednesday pickup yes/no question changes from a correct “Yes” lacking the required schedule keywords to a wrong Monday return schedule. Late-return and monthly-reservation questions change from reasonable source-absence statements lacking the literal “Confirmation” keyword to unrelated return-time and simultaneous-item policies with `needs_review:false`. These are the three after-training failures. The monthly probe specifically changes `needs_review`→`asserted`; the model answers a monthly booking question with “Each member may borrow up to 3 items at the same time.” Transfer and delivery probes retain review classifications but improve their answer text; the grounded pickup control remains correct. All probe inputs include the FAQ. Some other score gains reflect fuller phrasing: a correct bare “3” previously failed the required word “items.” Thus the 28.57%→85.71% change combines better flag use, better wording, and some genuine remaining errors; it does not establish new factual learning. [Raw before](results/implementation/eval_before.json), [raw after](results/implementation/eval_after.json), and the [complete manual audit](results/implementation/comparison_audit.txt) preserve all cases.

### Advanced: definitions, paraphrases, absent terms, and search

The unmodified `scripts/make_sft_dataset.py` was executed with standard Python. It produced 116 messages-format records from 58 glossary entries: “What is {term}?” and “Briefly explain {term}.” paired with each source definition. All 116 records are used directly in `Dataset.from_list`; there is no validation holdout. Training uses 20 epochs. Evaluation has 24 distinct held-out question strings: eight named-term questions, eight genuine concept paraphrases, and eight absent-term questions. Sixteen known IDs were checked against the glossary, and all eight absent names were checked against names and aliases.

The known concepts occur in training, so this evaluates new questions about trained facts, not held-out-term learning. Evaluation sends exactly one plain user question, with no FAQ prompt, JSON schema, glossary inventory, reference answer, or gold ID. The plain training/evaluation message styles match. JSON format rate is consequently inappropriate for this task.

A fixed keyword rubric accepts alternative phrasing through required concept groups and rejects specified contradictory phrases. Its known-content and absent-membership rates are reported separately, together with three category rates and their macro average. Raw text remains the primary evidence for semantic review. Substring scoring can miss a valid explanation or accept a misleading one; especially, an absence statement followed by an unsupported definition requires manual inspection.

| Advanced method | Named terms, n=8 | Paraphrases, n=8 | Absent terms, n=8 | Macro proxy rate |
|---|---|---|---|---|
| [Model before training](results/advanced/eval_before.json) | 5/8 = 62.50% | 4/8 = 50.00% | 0/8 = 0.00% | 37.50% |
| [Model after 20 epochs](results/advanced/eval_after.json) | 7/8 = 87.50% | 2/8 = 25.00% | 0/8 = 0.00% | 37.50% |
| [Deterministic glossary search](results/advanced/retrieval.json) | 8/8 = 100.00% | 4/8 = 50.00% | 7/8 = 87.50% | 79.17% |

[The qualitative review of the 16 known before-training answers](results/advanced/advanced-known-before-audit.json) identifies clear proxy artifacts. The MCP question confidently receives “HTTP/2” yet passes generic protocol/tool/client keywords. Conversely, a substantively correct quantization explanation says “reduces the precision” and fails the frozen precision alternatives. Several answers give relevant introductions but omit a requested second part, such as detecting overfitting or explaining how evidence helps; verbose saved responses visibly end mid-sentence or at a heading under the fixed 96-token generation budget. Missing continuations are not credited, and exact finish reasons/token counts were not captured. These are limitations of the measured response-and-budget setup, not grounds to alter the rubric after seeing results.

The search baseline extends the course tool's term-query approach to whole questions. It first chooses the longest boundary-matched glossary name, ID, or alias of at least three characters. Otherwise it ranks non-stopword token overlap against names, aliases, IDs, and definitions; acceptance requires at least three overlapping tokens and overlap/query-token count of at least 0.30. Ties resolve by source ID. It returns the original definition or an explicit no-match statement. Retrieval sees only the question and glossary, never gold IDs or reference answers. Source IDs, match methods, scores, and original retrieved text are saved and scored with the same proxy. The [24-question search audit](results/advanced/retrieval-audit.json) confirms all eight named-term matches, but incidental aliases cause errors: “next-token” selects Token instead of temperature, “agent” selects AI agent instead of HTTP 429/rate limit, and BM25's word “rank” selects LoRA rank. Lexical overlap also selects fine-tuning for the hallucination question. The structured-output paraphrase gets no match despite an existing entry, demonstrating that no-match is a retrieval failure signal, not proof of glossary absence.

This comparison has an important fairness limit. The absent terms are real concepts, including BM25 and beam search. Source-aware abstention is the desired benchmark behavior, but the original training script supplies no unknown examples and no explicit glossary membership inventory. Plain user questions also do not specify a glossary-only task. A pretrained model may correctly define an absent term and still fail this membership target. Search has direct source access and can explicitly report no match, although lexical failure also cannot prove a concept is absent. The absent score is therefore a membership stress test, not a general factual-accuracy measure; the combined macro score should not be used to rank general answer quality across these unequal source-access conditions.

The [completed Advanced run](results/advanced/result.json) trained 116 records for 20 epochs and 300 optimizer steps in **651.862 seconds**. Final logged interval loss was **0.07271329 at step 300**, while aggregate `train_loss` was **0.31919152**. It trained 6,389,760 of 758,782,784 parameters (0.8421%), with no validation subset. Named-term proxy performance increased by two questions while paraphrases decreased by two; known-content remained **9/16 = 56.25%** and the overall macro proxy remained **37.50%**. This is a redistribution of performance, not an overall improvement.

There are real gains in individual answers. The QLoRA response now accurately states, “A method that saves memory by quantizing base-model weights to, for example, 4 bits, then trains with LoRA.” The context-window paraphrase now receives “The model has a maximum total token count for a single call. Conversation history and the generated text must fit within that limit, making the choice of what to pass to the model important.” However, source-like wording is not reliable correctness: the passing LoRA answer factorizes “the model” rather than the weight update, and the passing PEFT answer wrongly connects choosing a parameter subset to 8-bit quantization of “original 64-bit parameters.” The failing fine-tuning answer even recommends inferring fresh facts from general knowledge instead of external tools, contradicting the source.

The recorded proxy regressions are `advanced_11` (hallucination), `advanced_12` (HTTP 429), and `advanced_16` (MCP). Their meanings differ. Hallucination loses the correct concept name but gains an actual evidence discussion, so the semantic change is mixed. HTTP 429 still correctly says, “A rate limit has been reached. The server has rejected the request and returns 429 Too Many Requests.” It fails only the explicit time/window keyword group, which illustrates a new false negative rather than clear loss of the concept. MCP was already wrong as “HTTP/2” before training; afterward it incorrectly answers “Copilot Chat” while mentioning MCP servers. The loss of its earlier false-positive pass is not a demonstrated new semantic regression. [The qualitative review after training](results/advanced/advanced-known-after-audit.json) and the [16-question raw comparison audit](results/advanced/comparison_audit.txt) preserve these distinctions. The qualitative review supplements the automated rubric, which was not changed after inspection.

The separate [absent-term semantic audit](results/advanced/absent-semantic-audit.json) inspected all eight before/after pairs using linked primary references. The before answers were assessed as one substantially correct, three partial, and four incorrect; all eight after answers contain substantive errors. These qualitative labels do not replace the frozen membership score. A concrete regression is cosine similarity: the before answer gives the correct −1 to +1 range, while the after answer claims “0 (dislike) to 1 (like).” Normalizing vectors does not remove negative values: opposite unit vectors have cosine −1 under the [normalized-dot-product definition](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.cosine_similarity.html). The after DPO answer also invents a rule that the model chooses the first answer presented, and BM25 receives an invented formula. These are identified factual errors, not errors inferred merely from glossary absence. The experiment therefore shows that more consistent glossary-style wording can coexist with unsupported explanations on these probes. This one small, single-seed test does not establish that fine-tuning generally harms factual knowledge.

### Interpretation and reproducibility

The measured original exercise shows that training can improve adherence to a format and repeated answer patterns while leaving both unsupported assertions and unnecessary abstention. Fine-tuning is appropriate when the objective is stable behavior, tone, or output structure supported by representative examples. For frequently changing facts, auditable source membership, and answers requiring a current reference, retrieval offers a more direct update path. Search quality still depends on query formulation and matching; lexical retrieval can miss paraphrases. A practical hybrid could retrieve evidence and train consistent answer behavior, but this experiment has not tested that hybrid.

Run the official notebook's pinned installation cells first in Colab; the runner installs nothing. From the course repository root, run each scenario in a fresh Python process and use a distinct empty output directory:

```bash
python lectures/02/run_experiments.py --scenario baseline --output-dir /content/lec02-results/baseline
python lectures/02/run_experiments.py --scenario basic --output-dir /content/lec02-results/basic
python lectures/02/run_experiments.py --scenario implementation --output-dir /content/lec02-results/implementation
python lectures/02/run_experiments.py --scenario advanced --output-dir /content/lec02-results/advanced
```

Dataset generation and local non-GPU checks are reproducible with:

```bash
python scripts/make_sft_dataset.py > lectures/02/prepared/glossary_sft.jsonl
python lectures/02/prepared/prepare_data.py
python -m unittest discover -s lectures/02/prepared -p test_runner.py -v
python lectures/02/verify_results.py
```

Eight runner tests passed before execution, covering the single changed Basic setting, dataset mapping, advanced prompt leakage, process isolation, output preservation, knowledge-probe evidence, format/content separation, retrieval determinism, and loss definitions. These tests verify the experimental machinery; they do not substitute for model execution. Original numerical evidence is archived in [`results/original_exercise/result.json`](results/original_exercise/result.json), [`extra_metrics.json`](results/original_exercise/extra_metrics.json), and the corresponding [before](results/original_exercise/eval_before.json), [after](results/original_exercise/eval_after.json), and [4-bit](results/original_exercise/eval_before_4bit.json) raw evaluation files. The completed runner artifacts are archived in [baseline](results/baseline/result.json), [Basic](results/basic/result.json), [Implementation](results/implementation/result.json), and [Advanced](results/advanced/result.json), with their raw evaluations, manifests, and training logs. The completed checks and their scope are recorded in [verification.json](results/verification.json).
