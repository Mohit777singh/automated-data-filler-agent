# Automated Data Filler Agent

An AI agent that extracts structured CRM data (Contact, Account, Opportunity)
from unstructured B2B sales meeting summaries, validates it, and writes it
into a CRM with duplicate detection -- built for the "Context Engineering"
capstone/hackathon problem statement.

## Architecture

```
Raw meeting text
      |
      v
[1] prompt_builder.py   -- builds a system prompt with few-shot examples
      |
      v
[2] extractor_ollama.py -- sends text+prompt to a local LLM, forces
      |                     structured JSON output (schema.py)
      v
[3] validator.py        -- checks the JSON is well-formed and sane
      |
      v
[4] run_agent.py / app.py -- shows the human what was extracted,
      |                       asks for confirmation
      v
[5] mock_crm_client.py  -- checks for duplicates (exact + fuzzy match),
    (or salesforce_client.py)   then creates/links Contact + Account + Opportunity
```

See `EVALUATOR_PREP.md` for the full walkthrough of each file, the AI/ML
concepts used, and anticipated evaluator Q&A.

### CRMClient interface -- mock vs. real Salesforce

`mock_crm_client.py` and `salesforce_client.py` expose the **identical**
interface: both have a `write_crm_record(extraction: dict) -> dict` method
with the same duplicate-detection logic (exact match, then fuzzy match via
rapidfuzz as a fallback). The mock stores records in a local JSON file
instead of a real Salesforce org, which means:

- The demo works fully offline, with zero setup/auth
- Swapping to live Salesforce later is a one-line import change, not a
  redesign, once Salesforce OAuth access is sorted out
- Both clients were tested with the same scenarios (see `EVALUATOR_PREP.md`)

## Folder structure

```
automated-data-filler-agent/
├── agent/
│   ├── __init__.py
│   ├── schema.py            # CRM extraction schema (forced structured output)
│   ├── prompt_builder.py    # builds system prompt w/ few-shot examples from data/
│   ├── extractor.py         # calls Claude API (needs ANTHROPIC_API_KEY, paid)
│   ├── extractor_ollama.py  # calls a LOCAL Ollama model (free, no key needed)
│   ├── validator.py         # validates extracted data before CRM writing
│   ├── mock_crm_client.py   # local JSON "CRM" -- fuzzy + exact duplicate detection
│   ├── salesforce_client.py # real Salesforce integration (same interface as mock)
│   ├── http_utils.py        # shared retry-once-on-5xx wrapper for HTTP calls
│   ├── run_agent.py         # full CLI pipeline: extract -> validate -> confirm -> write
│   └── evaluate.py          # accuracy evaluation against labeled test data
├── data/
│   ├── merged_train.jsonl   # few-shot examples pulled from here
│   └── merged_test.jsonl    # used for evaluation
├── notebooks/
│   └── run_agent.ipynb      # interactive Jupyter demo
├── app.py                   # Streamlit web UI (recommended for live demos)
├── requirements.txt
├── .env.example
├── .gitignore
├── EVALUATOR_PREP.md         # full explanation + Q&A prep for your evaluation
└── README.md
```

## Setup (free, local -- using Ollama)

This is the recommended path if you don't want to pay for API access.
Everything runs on your own machine.

**1. Install Ollama:**
Download and install from [ollama.com/download](https://ollama.com/download)
(Windows, Mac, and Linux are all supported). Open the Ollama app once
installed -- it runs a local server in the background at
`http://localhost:11434`. Leave it running whenever you use the agent.

**2. Pull a model** (one-time download, run in your terminal):

```bash
ollama pull llama3.1
```

This downloads an ~4.7GB model. If your machine is lower-spec or the
download/inference is too slow, use a smaller model instead:

```bash
ollama pull llama3.2
```

(and change `MODEL = "llama3.1"` to `MODEL = "llama3.2"` in
`agent/extractor_ollama.py`)

**3. Open a terminal in the `automated-data-filler-agent/` folder**
(the one containing this README), then create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
```

**4. Install Python dependencies:**

```bash
pip install -r requirements.txt
```

No API key or `.env` file needed for the Ollama path -- skip straight to
"Running it" below.

## Setup (alternative -- using the Claude API, paid after $5 trial)

If you'd rather use the hosted Claude API instead of a local model (better
extraction quality, but costs money after the initial free trial credit):

Copy the example env file and fill it in:

```bash
cp .env.example .env
```

Then open `.env` in any text editor and replace `your_api_key_here` with your
real key from [console.anthropic.com](https://console.anthropic.com). Use
`agent.extractor` (Claude) instead of `agent.extractor_ollama` in the code
examples below.

## Running it

### Option A -- Jupyter Notebook (recommended for exploring/demoing)

From the project root:

```bash
jupyter notebook notebooks/run_agent.ipynb
```

This opens the notebook in your browser. Run the cells top to bottom — it
walks through extraction on a sample summary, validation, testing your own
text, and a quick accuracy check against the test set.

If `jupyter notebook` doesn't launch a browser automatically, it'll print a
URL like `http://localhost:8888/?token=...` — copy that into your browser.

### Option B -- Command line

Make sure the Ollama app is open and running, then from the project root:

```bash
python3 -m agent.extractor_ollama
```

This runs the built-in test example and prints the extracted JSON. First
run may take a bit longer while the model loads into memory.

### Option C -- Import in your own script

```python
from agent.extractor_ollama import extract_crm_data
from agent.validator import validate_extraction

result = extract_crm_data("Had a call with Sarah Johnson, Marketing Director at GrowthTech...")
warnings = validate_extraction(result)
print(result, warnings)
```

Run this from the project root (or add the project root to `sys.path` first,
same as the notebook does).

## Common issues

- **`Could not reach Ollama at http://localhost:11434`** -- the Ollama app
  isn't running. Open it (it should show an icon in your system tray/menu
  bar once running) and try again.
- **Extraction is slow or the machine gets very hot/loud** -- normal for
  local LLM inference on a laptop, especially without a dedicated GPU.
  Switch to the smaller `llama3.2` model if `llama3.1` is too heavy.
- **`Model did not return valid JSON`** -- rare, but can happen with smaller
  models on unusual input. Re-running the same input often works; if it
  persists, try `llama3.1` if you were using a smaller model.
- **`ModuleNotFoundError: No module named 'agent'`** -- you're running from
  the wrong folder, or `sys.path` wasn't set. Always run commands from the
  project root, or from `notebooks/` after the `sys.path.append(...)` cell.
- **`No API key found`** (only relevant if using the Claude path) -- means
  `.env` wasn't picked up. Double check the file is literally named `.env`
  (not `.env.example` or `.env.txt`) and sits in the project root.

## Running the Streamlit demo UI

For live demos/judging, the web UI reads far better than a terminal:

```bash
streamlit run app.py
```

Opens in your browser. Paste a summary (or click "Use example text"),
click "Extract CRM Data", review the Contact/Account/Opportunity cards,
then click "Confirm and write to CRM". Includes a "View mock CRM contents"
panel at the bottom to show the full database live, and a reset button
between demo runs.

## Accuracy (run `agent/evaluate.py` to fill this in)

```bash
python -m agent.evaluate
```

This tests extraction against the full labeled test set (42 synthetic
examples with ground truth) plus a sparse sample (40 examples) to check
the model correctly avoids hallucinating on non-business text. It prints
a copy-paste-ready summary at the end -- paste your real numbers here:

- **Field-level extraction accuracy:** _(run evaluate.py and fill in)_
- **Hallucination-avoidance rate:** _(run evaluate.py and fill in)_

Detailed per-example results are saved to `eval_results.json` (gitignored
-- may contain full meeting-summary text from the test set).

## Recording a demo video

A strong 1-2 minute demo, using `streamlit run app.py`:

1. Paste a meeting summary (the GrowthTech example from the problem
   statement works well -- your evaluator will recognize it)
2. Click Extract -- show the Contact/Account/Opportunity cards appearing
3. Click Confirm -- show the "created" CRM actions
4. Paste the **same or a very similar** summary again (try a slightly
   different name spelling, e.g. "Sara Johnsen" instead of "Sarah
   Johnson") and Extract + Confirm again
5. Show the actions now say `reused_existing` / `reused_existing_fuzzy`
   instead of creating duplicates -- this directly demonstrates the
   "duplicate detection and data merging" requirement from the problem
   statement

## Next steps

Once extraction is validated and accurate, the next component to build is
the Salesforce API integration layer — mapping validated JSON into real
`Contact`/`Account`/`Opportunity` records, with duplicate detection before
creating new ones.
