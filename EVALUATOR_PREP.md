# Automated Data Filler Agent — Project Explanation & Evaluator Prep

## 1. The one-sentence pitch

"I built an AI agent that reads unstructured sales meeting notes, extracts structured CRM data (contact, company, deal info) using a local LLM, validates it, shows it to the user for confirmation, and writes it into a CRM — with duplicate detection so the same contact never gets created twice."

That's your opening line if asked "what did you build."

---

## 2. Why this project (mapping to the problem statement)

Your problem statement is literally titled **"Automated Data Filler Agent"** and its core theme is:

> *"Build an intelligent AI agent that automatically processes sales meeting summaries and populates CRM systems with structured data."*

And explicitly frames it as a **Context Engineering Hackathon** — meaning the evaluator isn't just grading "does it work," they're grading **how well you designed the AI's inputs (prompts, examples, schema) to get reliable structured output.** That's the single most important thing to emphasize.

---

## 3. The full pipeline (say this out loud, it's your architecture explanation)

```
Raw meeting text
      ↓
[1] Prompt Builder  → builds a system prompt with few-shot examples
      ↓
[2] Extractor       → sends text + prompt to an LLM, forces structured JSON output
      ↓
[3] Validator       → checks the JSON is well-formed and sane before trusting it
      ↓
[4] Confirmation    → shows the human what was extracted, asks for approval
      ↓
[5] CRM Client      → checks for duplicates, then creates/links Contact + Account + Opportunity
```

Every box above is a separate Python file. That's intentional — it's called **separation of concerns**: each piece does one job and can be tested/swapped independently. If an evaluator asks "why so many files instead of one script," that's your answer.

---

## 4. File-by-file walkthrough

### `agent/schema.py` — "What shape must the AI's answer take?"

This defines the **exact structure** the AI is allowed to return — Contact, Account, and Opportunity, each with specific fields (name, title, industry, deal amount, stage, etc.), modeled on Salesforce's real object fields.

**Why it matters:** Instead of asking the AI to "extract the important info" (vague, inconsistent output), we force it into a strict JSON schema. This is done using the LLM's **structured output / tool-calling** feature — the model literally cannot return anything that doesn't match this shape.

**Key line to explain:** Fields use `["string", "null"]` types — meaning every field can either have a real value OR be `null`. This is deliberate: it forces the model to be honest about what it doesn't know rather than making something up.

### `agent/prompt_builder.py` — "How do we teach the AI to do this well?"

This is your **Context Engineering** centerpiece. It builds the system prompt sent to the AI, and critically, it doesn't just describe the task — it shows **real examples**:
- 3 examples pulled from your labeled synthetic data (showing correct full extraction)
- 1 example from real sparse conversation data (showing the AI what to do when there's *nothing* to extract — i.e., correctly return `null`, not a guess)

**Why it matters:** This is called **few-shot prompting**. Research and practice both show LLMs perform far more reliably when shown examples of the exact input/output pattern, rather than just instructions. The explicit rule baked into the prompt — *"if a field is not mentioned, it MUST be null, never guess"* — is your direct defense against the #1 risk in this kind of system: **hallucination** (the AI inventing a company name, budget, or deal that was never actually mentioned).

### `agent/extractor_ollama.py` (and `extractor.py`) — "Where the actual AI call happens"

This sends the meeting text + the engineered prompt to an LLM and gets back structured JSON.

Two versions exist:
- `extractor.py` — calls Anthropic's Claude API (cloud, requires paid API key)
- `extractor_ollama.py` — calls **Ollama**, running an open-source LLM (Llama 3.2) **locally on your own machine**, completely free, no internet needed after the model is downloaded

**Why two versions:** This shows you understand that "the AI" is swappable — the *pipeline* (schema, prompt, validation) is the real engineering work; the specific model behind it is a plug-in component. If asked "why not GPT-4/Claude," you say: *"The pipeline is model-agnostic by design — I used a local open-source model for the working demo to avoid API costs, but the same schema and prompt work unchanged with Claude, GPT, or any other model that supports structured output."*

**Technical detail worth knowing:** `extractor_ollama.py` uses Ollama's `format` parameter — passing the JSON schema directly so the model's output generation is *constrained* token-by-token to match the schema. This is different from just asking nicely in the prompt; it's a hard guarantee at the decoding level.

### `agent/validator.py` — "Never trust AI output blindly"

Before any extracted data is allowed near a CRM, this checks:
- Are the data types correct? (Is "Amount" actually a number, not a string like "$30K"?)
- Is "StageName" one of the real allowed Salesforce stages, or did the model invent one?
- Is the email address at least shaped like an email?
- Is the overall confidence low? (flag for human review)

**Why it matters:** This is defensive engineering. LLMs occasionally produce malformed or slightly-wrong output even with a good prompt. A validator is the safety net between "AI said so" and "we trust it enough to write it into a real business system."

### `agent/mock_crm_client.py` — "The CRM integration layer"

This simulates a real CRM (like Salesforce) using a local JSON file as the database. It implements:
- **Exact duplicate detection**: before creating a new Contact, it checks if one with that email already exists. Same for Account by exact company name.
- **Fuzzy duplicate detection (fallback)**: meeting summaries very often don't include an email address, so exact matching alone misses a lot of real duplicates. When exact match fails, it falls back to fuzzy string matching (via the `rapidfuzz` library) on the person's full name or company name. Company names get extra handling — corporate suffixes like "Inc"/"LLC"/"Corp" are stripped before comparing, and both a spaced and despaced comparison are taken (the max of the two), since real company names vary a lot in spacing and formatting ("GrowthTech" vs "Growth Tech Inc" should match; "Vertex Systems" vs "Apex Systems" should *not*). I tested this against several true-positive and false-positive cases to tune the threshold (88 for names, 90 for companies) before trusting it.
- **Record linking**: Contacts and Opportunities are linked to their Account via an `AccountId`, mirroring how real CRMs relate objects.
- **create_record()**: generic method that writes any object type, used identically by Contact, Account, and Opportunity creation.

**Why fuzzy matching matters for this specific problem statement:** the problem statement explicitly calls out "Handle duplicate detection and data merging" as a requirement. Exact-match-only duplicate detection is fragile in practice — a rep might write "GrowthTech" one week and "Growth Tech Inc" the next, or an AI extraction might slightly misspell a name. Fuzzy matching is what makes duplicate detection actually robust in the real world, not just in a clean happy-path demo.

**Why a mock instead of real Salesforce:** I also built `salesforce_client.py`, with identical logic, that authenticates to real Salesforce and makes real API calls. The code is complete and correct, but Salesforce's newer "External Client App" OAuth type doesn't support the simple username-password login flow — it requires more complex OAuth flows (Authorization Code, JWT Bearer) that need a running web server for the callback, which wasn't practical for this timeline. **The mock CRM has the exact same interface (`write_crm_record()`)**, so switching from mock to real Salesforce is a one-line change once that auth is resolved — this is a deliberate software design choice (interface consistency), not a workaround that changes the architecture.

### `agent/run_agent.py` — "The conversational flow that ties it together"

This is the actual demo script matching your problem statement's exact example conversation:
1. Prompts for a meeting summary
2. Extracts (calls extractor)
3. Validates
4. **Shows the human what was found** and asks for confirmation (y/n) — nothing is written to the CRM without approval
5. On confirmation, writes to the CRM and reports what was created

**Why the confirmation step matters:** Your problem statement's target user is a sales rep — real businesses would never want an AI silently writing to their CRM with no human check. This human-in-the-loop step is a deliberate trust/safety design choice.

### `agent/evaluate.py` — "How do we know it's actually accurate?"

Runs the extractor against held-out labeled test data and reports:
- Field-by-field accuracy (did it get FirstName right? Amount? StageName?) on fully-labeled examples
- Null-handling accuracy (did it correctly avoid inventing data on sparse/non-business text?)

**If you've run this**, quote your actual numbers here. If you haven't run it yet, run it before your evaluation — a real accuracy percentage is far more convincing than "it seemed to work when I tried it."

### `data/` — the dataset

Three sources merged into one 5,072-row dataset:
1. **`sales_conversations`** (3,252 rows) — real B2B sales dialogue from a public Hugging Face dataset
2. **`samsum`** (1,500 rows) — real general dialogue/summary pairs, also from Hugging Face, added for linguistic diversity
3. **`synthetic`** (320 rows) — data I generated myself, the *only* source with full, correct CRM labels attached

**Why merge three sources instead of just using synthetic data:** No public dataset exists that pairs real sales meeting notes with correct CRM field labels — that data is private to companies' actual CRMs. So the approach was: use real dialogue for realistic *style* and *edge cases* (including "no info here" cases), and synthetic data for actual ground-truth labels to build the few-shot prompt from. Every row is tagged with its `source` so provenance is always traceable — I never pretend real unlabeled data has labels it doesn't.

### `notebooks/run_agent.ipynb` — the interactive demo

A Jupyter notebook version of the same pipeline, for live/visual demonstration — runs a single extraction, validates it, and includes a quick evaluation loop against test data. Good to have open during your actual evaluation for a live demo.

---

## 5. Core AI/ML concepts used (say these terms explicitly — evaluators listen for them)

| Concept | Where it's used |
|---|---|
| **Large Language Model (Transformer)** | The extraction engine itself (Claude / Llama) |
| **Few-shot prompting** | `prompt_builder.py` — examples shown before the real task |
| **Structured / constrained output** | `schema.py` + `format` parameter — forces valid JSON |
| **In-context learning** | The model learns the extraction pattern from prompt examples, no fine-tuning/training needed |
| **Context Engineering** | The overall discipline of designing prompt + schema + examples for reliable output (your hackathon's actual theme) |
| **Data validation** | `validator.py` |
| **Duplicate detection / entity resolution** | `mock_crm_client.py`'s find-by-email/name logic |
| **Human-in-the-loop** | The confirmation step in `run_agent.py` |

**If asked "did you use RAG":** Be honest — *"I use a simplified retrieval approach: relevant few-shot examples are pulled from a labeled dataset and injected into the prompt. It's not full vector-embedding-based RAG, since there's no semantic similarity search — examples are sampled, not retrieved by relevance to the specific input. That would be a natural next enhancement."* This honest answer is stronger than overclaiming.

**If asked "did you train a model":** No — *"I didn't train or fine-tune a model. I used in-context learning: providing examples directly in the prompt so a pre-trained model (Llama 3.2 via Ollama, or Claude) learns the extraction pattern at inference time. This is the modern standard approach for this kind of task and avoids needing large labeled datasets or GPU training infrastructure."*

---

## 6. Known limitations — own these, don't hide them

Evaluators respect honesty about limitations far more than a project that pretends to be perfect. Be ready to state:

1. **Not connected to live Salesforce** — the integration code is complete (`salesforce_client.py`) and uses the identical interface as the working mock CRM, but authentication is blocked by Salesforce's OAuth policy restrictions on the newer "External Client App" type, which requires a more complex OAuth flow than fit this project's timeline.
2. **Local LLM accuracy is lower than a cloud model like GPT-4/Claude** — smaller open-source models occasionally hallucinate fields (e.g., guessing a "LeadSource" that wasn't stated). This is a known, documented tradeoff of running for free locally vs. paying for a larger cloud model.
3. **Real-world meeting summaries are messier than test data** — the dataset includes real dialogue for robustness, but production deployment would need broader testing across more industries, accents, and formats.
4. **No retry/backoff logic** for transient API failures — a production system would need more robust error recovery.

---

## 7. Likely evaluator questions — with answers

**Q: Why not just use regex or manual rules instead of an LLM?**
A: Sales meeting summaries have huge variation in phrasing, structure, and completeness — regex can't generalize across "Had a call with X" vs "Notes from the Y call, Z was on" vs completely unstructured text. LLMs handle this natural language variability without needing a rule for every possible phrasing.

**Q: How do you know it's not just making things up?**
A: Two layers of defense: (1) the prompt explicitly instructs null-over-guessing, backed by an example showing correct null behavior on sparse text; (2) the validator checks output sanity before it's trusted, and the confirmation step means a human reviews every extraction before it's written anywhere.

**Q: What happens if the AI extracts something wrong and the user doesn't catch it?**
A: The confirmation step is the primary safeguard — the user sees exactly what was extracted before anything is written. Beyond that, since it's a CRM, wrong data can always be corrected/edited afterward like any manual entry error; this isn't different in kind from a human mistyping something.

**Q: Why three data sources merged instead of one clean dataset?**
A: No single public dataset provides both realistic sales dialogue *and* correct CRM labels (that data is private to companies). Merging let me combine real dialogue diversity (2 public sources) with labeled ground truth (my synthetic data), each serving a different purpose, and each tagged by source so I never conflate real-unlabeled data with actually-verified labels.

**Q: Is this actually solving the business problem stated (reps spending 2-3 hrs/day on data entry)?**
A: Yes in principle — the pipeline demonstrates the core mechanism (extract → validate → confirm → write) that would eliminate manual re-typing. Scaling to production would mean connecting to a real CRM (architecture is ready, code exists), handling more edge cases, and likely fine-tuning or using a stronger model for higher accuracy — but the core time-saving mechanism is proven working.

**Q: What would you improve with more time?**
A: (1) Resolve the Salesforce live connection via a proper OAuth web-server callback flow, (2) upgrade few-shot example selection from random sampling to real semantic retrieval (true RAG), (3) run the full evaluation suite for hard accuracy numbers across more test cases, (4) add fine-tuning on a larger labeled dataset if scaling beyond few-shot prompting.

---

## 8. Suggested live demo order (if you get to demo it)

1. Show `notebooks/run_agent.ipynb` or run `python -m agent.run_agent` in a terminal
2. Paste in a meeting summary (use the PDF's exact example — it's the one your evaluator already knows)
3. Show the extracted output on screen
4. Confirm (`y`) and show the CRM write result
5. Run it a *second time* with a similar summary to show duplicate detection kicking in (existing Account/Contact reused, not duplicated) — this is a great "wow" moment since it directly matches a named requirement in the problem statement
6. If time allows, open `agent/schema.py` and `agent/prompt_builder.py` briefly to show the "engineering" behind the AI call, not just the AI call itself

Good luck — you built a real, working, honestly-documented system that maps directly onto every major requirement in your problem statement. That's a strong position to defend.
