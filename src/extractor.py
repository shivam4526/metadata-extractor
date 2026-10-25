"""Provider-backed, schema-validated agreement metadata extraction."""
from __future__ import annotations

import base64
import csv
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from threading import Lock
from typing import Any

import anthropic
from dotenv import load_dotenv
from groq import APIError as GroqAPIError
from groq import Groq
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .ingest import ingest
from .normalize import normalize
from .schemas import CSV_COLUMNS, FIELDS, ExtractionResult

logger = logging.getLogger(__name__)
PROMPT_VERSION = "2026-10-24.3"
_examples_lock = Lock()
_examples: str | None = None

GROQ_RESULT_SCHEMA = {
    "type": "object",
    "properties": {field: {"type": ["string", "null"]} for field in FIELDS},
    "required": list(FIELDS),
    "additionalProperties": False,
}

SYSTEM = """You extract metadata from rental and lease agreements. You are a semantic document-understanding model, not a keyword or pattern matcher.
Return exactly one JSON object and no Markdown. Keys must be agreement_value, agreement_start_date, agreement_end_date, renewal_notice_days, party_one, party_two.
agreement_value is the recurring agreed rent/value, not a deposit or advance. agreement_start_date and agreement_end_date are the contractual term boundaries. renewal_notice_days is the required notice length in days for renewal, non-renewal, or termination; return null if absent. party_one and party_two must be the first and second contracting parties in the agreement's stated order (for example owner then tenant), retaining names but excluding addresses and descriptions.
Use DD.MM.YYYY for dates, digits only for agreement_value and renewal_notice_days, and preserve meaningful party-name capitalization. Use null when a field is genuinely absent or unreadable. Do not infer a value merely because it is common in other agreements."""


def _short_text(path: Path) -> str:
    try:
        _, text = ingest(path)
        return text[:5000]
    except Exception as error:  # few-shots are helpful, never mandatory
        logger.warning("Could not create few-shot text for %s: %s", path.name, error)
        return ""


def build_few_shots(data_dir: Path = Path("data")) -> str:
    """Build up to five labelled train examples; no test labels are ever used."""
    global _examples
    with _examples_lock:
        if _examples is not None:
            return _examples
        csv_path = data_dir / "train.csv"
        train_dir = data_dir / "train"
        if not csv_path.exists() or not train_dir.exists():
            _examples = ""
            return _examples
        paths = {path.name.rsplit(".", 1)[0]: path for path in train_dir.iterdir() if path.is_file()}
        doc_candidates: list[tuple[Path, dict[str, str]]] = []
        image_candidates: list[tuple[Path, dict[str, str]]] = []
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                path = paths.get(row["File Name"])
                if path is None:
                    continue
                target = image_candidates if path.suffix.lower() in {".png", ".jpg", ".jpeg"} else doc_candidates
                target.append((path, row))
        # Three compact examples keep requests below entry-tier token limits while
        # still covering both native DOCX and OCR-derived image text.
        examples: list[str] = []
        for path, row in [*doc_candidates[:2], *image_candidates[:1]]:
            text = _short_text(path)[:700]
            if text:
                answer = {field: row[CSV_COLUMNS[field]].strip() or None for field in FIELDS}
                examples.append(f"TRAINING EXAMPLE (document excerpt):\n{text}\nEXPECTED JSON:\n{json.dumps(answer)}")
        _examples = "\n\n".join(examples)
        return _examples


class AgreementExtractor:
    def __init__(self, data_dir: Path = Path("data")) -> None:
        load_dotenv()
        self.data_dir = data_dir
        self.use_vision = os.getenv("USE_VISION_FOR_IMAGES", "false").lower() == "true"
        self.cache_dir = Path(os.getenv("CACHE_DIR", ".cache/extractions"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._request_lock = Lock()
        self._last_request_at = 0.0
        requested_provider = os.getenv("LLM_PROVIDER", "auto").lower()
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        groq_key = os.getenv("GROQ_API_KEY")
        if requested_provider not in {"auto", "anthropic", "groq"}:
            raise RuntimeError("LLM_PROVIDER must be auto, anthropic, or groq")
        if requested_provider in {"auto", "anthropic"} and anthropic_key:
            self.provider = "anthropic"
            self.model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
            self.client = anthropic.Anthropic(api_key=anthropic_key)
        elif requested_provider in {"auto", "groq"} and groq_key:
            self.provider = "groq"
            self.model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
            self.vision_model = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.8-27b")
            self.client = Groq(api_key=groq_key)
            self.min_request_interval = float(os.getenv("MIN_REQUEST_INTERVAL_SECONDS", "35"))
        elif requested_provider == "anthropic":
            raise RuntimeError("LLM_PROVIDER=anthropic requires ANTHROPIC_API_KEY")
        elif requested_provider == "groq":
            raise RuntimeError("LLM_PROVIDER=groq requires GROQ_API_KEY")
        else:
            raise RuntimeError("Set ANTHROPIC_API_KEY or GROQ_API_KEY in .env. Anthropic is preferred when both are set.")

    def _cache_path(self, path: Path, model: str) -> Path:
        cache_identity = f"{PROMPT_VERSION}:{self.provider}:{model}".encode()
        digest = hashlib.sha256(path.read_bytes() + cache_identity).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _prompt(self) -> str:
        examples = build_few_shots(self.data_dir)
        return SYSTEM + ("\n\nLearn the target annotation conventions from these labelled examples:\n" + examples if examples else "")

    @retry(retry=retry_if_exception_type((anthropic.APIError, GroqAPIError)), wait=wait_exponential(min=1, max=8), stop=stop_after_attempt(3), reraise=True)
    def _request(self, content: Any, model: str) -> str:
        if self.provider == "anthropic":
            message = self.client.messages.create(
                model=model, max_tokens=700, temperature=0, system=self._prompt(),
                messages=[{"role": "user", "content": content}],
            )
            return "".join(block.text for block in message.content if block.type == "text")
        with self._request_lock:
            delay = self.min_request_interval - (time.monotonic() - self._last_request_at)
            if delay > 0:
                time.sleep(delay)
            response = self.client.chat.completions.create(
                model=model, max_completion_tokens=700, temperature=0.000001,
            messages=[{"role": "system", "content": self._prompt()}, {"role": "user", "content": content}],
            )
            self._last_request_at = time.monotonic()
        return response.choices[0].message.content or ""

    def extract(self, path: Path) -> ExtractionResult:
        is_vision = path.suffix.lower() in {".png", ".jpg", ".jpeg"} and self.use_vision
        request_model = self.vision_model if self.provider == "groq" and is_vision else self.model
        cache_path = self._cache_path(path, request_model)
        if cache_path.exists():
            return normalize(ExtractionResult.model_validate_json(cache_path.read_text(encoding="utf-8")))
        kind, text = ingest(path)
        if kind == "image" and self.use_vision:
            media_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
            encoded = base64.b64encode(path.read_bytes()).decode()
            if self.provider == "anthropic":
                content: Any = [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": encoded}},
                    {"type": "text", "text": "Extract the six agreement fields from this document image."},
                ]
            else:
                content = [
                    {"type": "text", "text": "Extract the six agreement fields from this document image."},
                    {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{encoded}"}},
                ]
        else:
            content = "Extract the six agreement fields from this agreement text:\n\n" + text[:4000]
        last_error: Exception | None = None
        for _ in range(2):  # retry once if the model violates the JSON response contract
            try:
                result = normalize(ExtractionResult.model_validate_json(self._request(content, request_model)))
                cache_path.write_text(result.model_dump_json(), encoding="utf-8")
                return result
            except (ValueError, json.JSONDecodeError) as error:
                last_error = error
                content = str(content) + "\nReturn valid JSON only, matching the requested schema."
        raise RuntimeError(f"Model returned invalid JSON for {path.name}: {last_error}")
