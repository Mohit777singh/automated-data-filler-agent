\# Automated Data Filler Agent



An AI agent that extracts structured CRM data from unstructured B2B sales meeting summaries, validates it, and writes it into a CRM with duplicate detection. Built for a Context Engineering hackathon focused on automating manual CRM data entry.



\## Architecture



Raw meeting text

│

▼

prompt\_builder.py → builds a system prompt with few-shot examples

│

▼

extractor\_ollama.py → sends text + prompt to a local LLM, forces

│ structured JSON output via schema.py

▼

validator.py → checks the extracted data is well-formed and sane

│

▼

run\_agent.py / app.py → surfaces the extraction for human confirmation

│

▼

mock\_crm\_client.py → duplicate detection (exact + fuzzy match),

(or salesforce\_client.py) then creates/links Contact, Account, Opportunity





Full design rationale, per-file breakdown, and the concepts behind each stage are in `EVALUATOR\_PREP.md`.



\### CRM layer



`mock\_crm\_client.py` and `salesforce\_client.py` share an identical interface — both implement `write\_crm\_record(extraction: dict) -> dict` with the same duplicate-detection logic (exact match, then fuzzy match via `rapidfuzz`). The mock persists to a local JSON file rather than a live Salesforce org, which keeps the demo fully offline and dependency-free. Switching to live Salesforce is a one-line import change once org access is configured — not a redesign.



\## Project structure



automated-data-filler-agent/

├── agent/

│ ├── schema.py # extraction schema (forced structured output)

│ ├── prompt\_builder.py # few-shot prompt construction

│ ├── extractor.py # Claude API extraction

│ ├── extractor\_ollama.py # local LLM extraction (Ollama)

│ ├── validator.py # pre-write data validation

│ ├── mock\_crm\_client.py # local CRM simulator, fuzzy duplicate detection

│ ├── salesforce\_client.py # live Salesforce integration

│ ├── http\_utils.py # HTTP retry wrapper

│ ├── run\_agent.py # CLI pipeline: extract → validate → confirm → write

│ └── evaluate.py # accuracy evaluation harness

├── data/

│ ├── merged\_train.jsonl # few-shot source data

│ └── merged\_test.jsonl # evaluation set

├── notebooks/run\_agent.ipynb

├── app.py # Streamlit demo UI

├── requirements.txt

├── .env.example

└── EVALUATOR\_PREP.md





\## Setup



\*\*Local LLM (recommended — free, no API key):\*\*



```bash

ollama pull llama3.2

python3 -m venv venv \&\& source venv/bin/activate   # Windows: venv\\Scripts\\activate

pip install -r requirements.txt

```



Ensure `MODEL` in `agent/extractor\_ollama.py` matches a model you've actually pulled (`ollama list` to check).



\*\*Claude API (alternative — higher accuracy, paid after free trial credit):\*\*



```bash

cp .env.example .env   # add your ANTHROPIC\_API\_KEY

```

Use `agent.extractor` in place of `agent.extractor\_ollama`.



\## Usage



\*\*Web UI:\*\*

```bash

streamlit run app.py

```



\*\*Notebook:\*\*

```bash

jupyter notebook notebooks/run\_agent.ipynb

```



\*\*CLI:\*\*

```bash

python -m agent.run\_agent

```



\*\*Programmatic:\*\*

```python

from agent.extractor\_ollama import extract\_crm\_data

from agent.validator import validate\_extraction



result = extract\_crm\_data("Had a call with Sarah Johnson, Marketing Director at GrowthTech...")

warnings = validate\_extraction(result)

```



\## Accuracy



```bash

python -m agent.evaluate

```



Evaluated against 42 fully-labeled examples and 40 sparse (non-business) examples:



| Metric | Result |

|---|---|

| Field accuracy (fields present in source text) | \*\*93.8%\*\* (315/336) |

| Field accuracy (all fields, including some never stated in text) | 63.1% (318/504) |

| Hallucination-avoidance rate | \*\*95.0%\*\* (38/40) |



The test set's ground truth includes a handful of fields (email, exact employee count, deal stage) generated independently of the source text — a correct agent returns `null` on these rather than guessing, which is why the adjusted figure (93.8%) is the representative measure of extraction quality, not the raw one. Both are reported for transparency; `evaluate.py` breaks the distinction down field-by-field.



Full per-example results are written to `eval\_results.json`.



\## Known limitations



\- Live Salesforce write is implemented but not currently connected — OAuth on Salesforce's newer External Client App type requires a flow beyond username/password auth. The mock CRM exercises identical logic and is used for the working demo.

\- Local model accuracy trails a larger hosted model; occasional field-level hallucination is possible on ambiguous input.

\- No retry/backoff beyond a single retry on transient 5xx errors.



\## Next steps



Live Salesforce connection via proper OAuth (Authorization Code or JWT Bearer flow), and upgrading few-shot example selection from random sampling to embedding-based retrieval.

Built by Mohit Singh 🚀










