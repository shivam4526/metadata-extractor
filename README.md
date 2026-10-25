# Agreement Metadata Extractor

An AI-powered REST service and command-line pipeline that extracts six fields from rental/lease agreements in `.docx`, `.png`, `.jpg`, and `.jpeg` formats:

- Agreement Value
- Agreement Start Date
- Agreement End Date
- Renewal Notice (Days)
- Party One
- Party Two

## Solution architecture

```text
DOCX ──> paragraphs + tables + headers/footers ─┐
                                                ├─> LLM extraction ─> Pydantic validation ─> minimal formatting ─> API / CSV
Image ──> OpenCV preprocessing + Tesseract OCR ─┘          │
     └─> optional direct vision input                       └─> disk cache + retries

LLM provider: Anthropic when configured, otherwise Groq
```

### Why this is not rule based

The LLM makes every document-understanding decision: selecting the agreement value rather than a deposit, identifying the contractual dates, resolving party order, and deciding whether notice is present. There is no regex, keyword lookup, or fixed template logic used to locate values. Post-processing only formats already selected values (for example, a date as `DD.MM.YYYY`) and validates the returned JSON.

Three compact labelled examples are built only from `data/train`; test labels are never used in the extraction prompt. Responses are cached in `.cache/extractions/` by document, provider, model, and prompt version.

## Project structure

```text
data/                                     Supplied train/test documents and CSV labels
src/
  ingest.py                               DOCX reading and image OCR preprocessing
  extractor.py                            Anthropic/Groq provider selection, prompts, cache, retry
  normalize.py                            Minimal output-only value formatting
  evaluate.py                             Exact-match, per-field recall and mismatch output
  run_predictions.py                      Prediction/evaluation CLI
  api.py                                  FastAPI service: /health and /extract
  schemas.py                              Pydantic extraction contract and CSV mapping
tests/test_pipeline.py                    Automated unit/API/ingestion tests
metadata_extraction_walkthrough.ipynb     Interactive Jupyter notebook walkthrough
pytest.ini                                Pytest configuration (pythonpath = .)
predictions_test.csv                      Generated test-set predictions
mismatches_test.csv                       Generated test-set error-analysis report
```


## Requirements

- Python 3.11 or newer
- Tesseract OCR (required for PNG/JPG/JPEG extraction)
- One provider API key: Groq **or** Anthropic

### Install Tesseract

macOS:

```bash
brew install tesseract
tesseract --version
```

Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr
tesseract --version
```

## Setup from a clean machine

Run all commands from the project root.

```bash
cd "/path/to/Metadata-extractor"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set **one** provider. Do not commit this file.

Recommended Groq configuration:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_key
GROQ_MODEL=openai/gpt-oss-20b
USE_VISION_FOR_IMAGES=false
MAX_WORKERS=1
MIN_REQUEST_INTERVAL_SECONDS=35
```

Anthropic configuration:

```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your_anthropic_key
ANTHROPIC_MODEL=claude-3-5-sonnet-latest
USE_VISION_FOR_IMAGES=false
```

With `LLM_PROVIDER=auto`, Anthropic is selected if both keys are present; otherwise Groq is used. Groq defaults to paced single-document calls to work within lower API rate limits. Set `USE_VISION_FOR_IMAGES=true` to use a vision model for images instead of OCR text, then compare recall before retaining that setting.

## Generate predictions and recall

Start with a small smoke test:

```bash
python -m src.run_predictions --split train --limit 5
```

Run the complete evaluation:

```bash
python -m src.run_predictions --split train
python -m src.run_predictions --split test
```

Each command creates:

- `predictions_train.csv` or `predictions_test.csv` — same headers as the corresponding source CSV
- `mismatches_train.csv` or `mismatches_test.csv` — filename, field, expected value, predicted value

Recall is calculated exactly as requested:

```text
field recall = documents with an exact field match / documents evaluated
```

Only leading/trailing whitespace is stripped during evaluation. A missing, mismatched, or failed prediction is counted as false.

## Measured test result

### 1. Per-Field Exact-Match Recall Score

The included [predictions_test.csv](predictions_test.csv) was generated using Groq (`openai/gpt-oss-20b`) on the test documents. Exact-match recall against `data/test.csv`:

| Field | Recall | Exact Matches | Total Evaluated |
| :--- | :---: | :---: | :---: |
| **Agreement Value** | **100.00%** | 4 | 4 |
| **Agreement Start Date** | **100.00%** | 4 | 4 |
| **Agreement End Date** | **75.00%** | 3 | 4 |
| **Renewal Notice (Days)** | **75.00%** | 3 | 4 |
| **Party One** | **25.00%** | 1 | 4 |
| **Party Two** | **25.00%** | 1 | 4 |

### 2. Test Set Predictions Table (`predictions_test.csv`)

| File Name | Agreement Value | Agreement Start Date | Agreement End Date | Renewal Notice (Days) | Party One | Party Two |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **24158401-Rental-Agreement** (`.png`) | 12000 | 01.04.2008 | 31.03.2009 | 60 | Sri Hanumaiah | Sri Vishal Bhardwaj S/O Charnel Singh |
| **95980236-Rental-Agreement** (`.png`) | 9000 | 01.04.2010 | 31.03.2011 | 30 | Mrs. S.Sakunthala | V.V.Ravi Kian |
| **156155545-Rental-Agreement-Kns-Home** (`.docx`) | 12000 | 15.12.2012 | 15.11.2013 | 30 | V.K.NATARAJ | SRI VYSHNAVI DAIRY SPECIALITIES Private Ltd. |
| **228094620-Rental-Agreement** (`.docx`) | 15000 | 07.07.2013 | 06.06.2014 | *(null)* | Mr. KAPIL MEHROTRA | Mr.B.Kishore |

### 3. Error Analysis and Mismatch Disclosures

A per-document analysis of discrepancies in [mismatches_test.csv](mismatches_test.csv):

- **Party Names:** The semantic model extracts the full legal entity reference (e.g. `Mrs. S.Sakunthala`, `Mr. KAPIL MEHROTRA`, `Sri Hanumaiah`), while ground truth annotations stripped titles (`S.Sakunthala`, `KAPIL MEHROTRA`, `Hanumaiah`). In `228094620`, the ground truth label `.B.Kishore` retains a typographical leading dot from a naive title removal on the raw text `Mr.B.Kishore`.
- **Renewal Notice (Days):** In `228094620`, the document mentions "vacate the flat with one months notice" under tenant violation remedies rather than a dedicated renewal clause; the model returned null while annotations mapped this to 30 days.
- **Agreement End Date:** In `156155545`, the lease runs for 11 months from 15.12.2012; the model inferred the 11-month calendar anniversary 15.11.2013 while the ground truth computed the concluding day (14.11.2013).
- **Adherence to Constraints:** In strict compliance with the directive **"Please do not use a rule-based/oriented approach (RegEx, static conditions, etc)"**, no post-hoc regex string manipulations were added to force matches. All values are determined by the semantic AI model.

### Dataset caveat

`data/train.csv` contains the row `24158401-Rental-Agreement`, but its source document is absent from `data/train/` (the PNG is under `data/test/`). The CLI safely writes nulls and records mismatches for that row rather than fabricating output. This affects train recall and should be disclosed during evaluation.

## Run the REST API

Start the service:

```bash
python -m uvicorn src.api:app --reload
```

Expected server URL: `http://127.0.0.1:8000`

Open the interactive API documentation in a browser:

```text
http://127.0.0.1:8000/docs
```

### Manual demo commands

Open a **second Terminal** in the same project folder while the server is running.

Health check:

```bash
curl -i http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok"}
```

Extract from a scanned agreement:

```bash
curl -sS -X POST http://127.0.0.1:8000/extract \
  -F 'file=@data/test/24158401-Rental-Agreement.png'
```

Extract from a Word agreement:

```bash
curl -sS -X POST http://127.0.0.1:8000/extract \
  -F 'file=@data/test/228094620-Rental-Agreement.pdf.docx'
```

Expected shape (values vary by uploaded document):

```json
{
  "agreement_value": "12000",
  "agreement_start_date": "01.04.2008",
  "agreement_end_date": "31.03.2009",
  "renewal_notice_days": "60",
  "party_one": "...",
  "party_two": "..."
}
```

Unsupported file check (expected HTTP `415`):

```bash
curl -i -X POST http://127.0.0.1:8000/extract \
  -F 'file=@README.md'
```

## Run automated tests

```bash
python -m pytest tests/test_pipeline.py -q
```

Expected result:

```text
5 passed
```

Tests cover real DOCX ingestion, output formatting, exact-match evaluation, provider selection, API health, and invalid file-type handling.

## Demo checklist

1. Show `python -m pytest tests/test_pipeline.py -q` passing.
2. Start the API and demonstrate `GET /health`.
3. Upload one image and one DOCX through `/docs` or the two `curl` commands.
4. Show `predictions_test.csv` and `mismatches_test.csv`.
5. Explain that extraction is LLM-based, not template/regex based.
6. Explain the measured recall, party-name limitation, and the supplied train-file mismatch transparently.

## Limitations and future improvements

- OCR quality can limit image accuracy; evaluate optional direct vision on a validation split.
- Contract parties can include titles, representatives, and addresses that differ from annotation conventions.
- Very long documents may benefit from semantic chunking/retrieval before extraction.
- A labelled larger dataset could support a fine-tuned layout-aware extraction model.
