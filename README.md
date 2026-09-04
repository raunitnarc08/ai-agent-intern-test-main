# Aster & Row AI Support Agent

An AI customer-support agent for Aster & Row, a fictional ecommerce company selling bags, drinkware, and travel accessories.

The agent uses a RAG system over the company's knowledge base and an order-lookup tool. It also handles conflicting sources, stale/internal content, and privacy-sensitive order data.

## 1. Setup and Run

### Clone the repository

```bash
git clone <https://github.com/raunitnarc08/ai-agent-intern-test-main.git>
cd ai-agent-intern-test-main
```

### Create a virtual environment

```bash
python3.11 -m venv venv
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Configure environment variables

Copy the example environment file:

```bash
cp .env.example .env
```

Then add your Groq API key to `.env`.

**Do not commit `.env` to GitHub.**

### Run the CLI

```bash
python3.11 cli.py
```

The CLI supports:

```text
/reset
```

to clear the current conversation, and:

```text
/exit
```

to exit.

The CLI preserves conversation context across turns, allowing follow-up questions such as asking when an order will arrive after looking up its status.

---

## 2. Environment Variables

Create a `.env` file containing:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
```

The primary model is `openai/gpt-oss-120b`, with a secondary model used as fallback.

No other credentials are required, and no customer data or real API keys should be committed to the repository.

---

## 3. Tech Stack

| Component | Choice |
|---|---|
| LLM | Groq `openai/gpt-oss-120b` with fallback |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Retrieval | Hybrid embedding + keyword/category scoring |
| Framework | FastAPI + CLI |
| Storage | In-memory index |
| Order lookup | `data/orders.json` |

The retrieval system combines embedding similarity with keyword/category boosts and applies document precedence to prefer active, customer-facing sources over legacy, draft, or internal content.

The order lookup tool returns a sanitized result and does not expose customer email, address, internal notes, or risk score.

---

## 4. Architecture

```text
User message
     |
     v
Order ID / follow-up detection
     |
     +------> Order lookup tool
     |              |
     |              v
     |        Sanitized result
     |
     v
KB retrieval
(hybrid scoring + precedence filter)
     |
     v
Conflict detector
     |
     v
Prompt assembly
     |
     v
LLM
(primary model -> fallback)
     |
     +------> Deterministic fallback
     |        if LLM unavailable
     |
     v
Response + sources + handoff flag
```

Retrieved passages and order-tool results are treated as untrusted data rather than instructions. The order tool also sanitizes information at the tool layer so sensitive fields are not placed into the model's context.

---

## 5. Running the Evaluation

Run the evaluation suite with:

```bash
python evaluation/run_eval.py
```

This runs **29 test cases**:

- 15 supplied visible cases
- 14 original cases

The detailed results are written to:

```text
evaluation/eval_results.json
```

Unit tests can be run separately with:

```bash
python3.11 -m pytest tests/
```

The evaluation harness deliberately spaces LLM calls to stay within Groq's rate limits. A complete evaluation can therefore take approximately **20–40 minutes**.

---

## 6. Evaluation Results

### Last independently verified result

The last clean, independently verified evaluation result was:

**19 / 29 — 65.5%**

| Category | Total | Passed | Pass Rate |
|---|---:|---:|---:|
| Abstention | 1 | 1 | 100.0% |
| Conversation | 1 | 0 | 0.0% |
| Groundedness | 2 | 2 | 100.0% |
| Multi-source grounding | 3 | 2 | 66.7% |
| Privacy | 3 | 2 | 66.7% |
| Prompt security | 3 | 1 | 33.3% |
| Retrieval | 6 | 3 | 50.0% |
| Source conflict | 1 | 0 | 0.0% |
| Tool reliability | 5 | 4 | 80.0% |
| Tool use | 4 | 4 | 100.0% |
| **Overall** | **29** | **19** | **65.5%** |

The 19/29 result is the reported trustworthy result. A later run was not treated as valid because the Groq daily token quota was exhausted partway through the evaluation.

---

## 7. Known Limitations

The main remaining gaps are in:

- Retrieval
- Source conflict
- Prompt security
- Conversation handling

There are also known limitations around the current out-of-scope detection, HTTP API session persistence, and evaluation contamination when the daily token quota is exhausted.

The CLI correctly preserves multi-turn context; the HTTP API requires the caller to pass the same `session_id` between requests.

---

## 8. Bug Diary

### Bug 1 — Hardcoded answer key

An earlier version contained hardcoded answers in the system prompt. This was removed and replaced with generic grounding and conflict-handling instructions.

A regression test was added to ensure that changing a KB source changes the agent's answer accordingly.

### Bug 2 — Rate-limit pacing

The original evaluation pacing caused frequent Groq rate-limit errors, which caused responses to fall through to the deterministic fallback.

The evaluation harness was updated with configurable pacing and retry settings.

### Bug 3 — LLM paraphrasing

Some evaluation cases failed because the model paraphrased specific phrases that the evaluator expected.

Additional prompt rules were added for delayed status, escalation wording, source conflicts, and return-window exceptions. The effect of these changes has **not been independently verified** because of the daily quota issue.

### Bug 4 — Daily token quota

Repeated evaluation runs can exhaust Groq's daily token quota.

When this happens, requests may fall through to fallback responses, making the resulting evaluation score unreliable.

This issue is documented but was not fully fixed due to time constraints.

---

## 9. AI Coding Tools Used

Claude and ChatGPT were used throughout development and debugging.

AI-generated suggestions were checked against the actual source code and raw evaluation output rather than being treated as ground truth.

Two examples of incorrect AI-generated claims were identified during development, including a fabricated report about a rate-limit fix and an incorrect claim about evaluation regressions.

---

## 10. Demo Video

The demo shows:
link -> https://drive.google.com/file/d/1HxI0nctm5HINjon48e528gZ8vD8bom0Y/view?usp=drive_link
1. A knowledge-base question with citations
2. An order lookup
3. A multi-turn conversation
4. A case where the agent detects conflicting information and recommends human help
5. The evaluation suite running

