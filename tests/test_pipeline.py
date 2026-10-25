from pathlib import Path

from fastapi.testclient import TestClient

from src.api import app
from src.evaluate import evaluate
from src.extractor import AgreementExtractor
from src.ingest import read_docx
from src.normalize import normalize
from src.schemas import ExtractionResult


def test_docx_ingestion_reads_real_training_document() -> None:
    text = read_docx(Path("data/train/18325926-Rental-Agreement-1.docx"))
    assert "RENTAL AGREEMENT" in text
    assert "Kuttan" in text


def test_normalization_only_formats_model_values() -> None:
    result = normalize(ExtractionResult(
        agreement_value="Rs. 12,000.00", agreement_start_date="1 April 2008",
        agreement_end_date="31 March 2009", renewal_notice_days="60.0",
        party_one=" Hanumaiah  ", party_two="Vishal Bhardwaj",
    ))
    assert result.model_dump() == {
        "agreement_value": "12000", "agreement_start_date": "01.04.2008",
        "agreement_end_date": "31.03.2009", "renewal_notice_days": "60",
        "party_one": "Hanumaiah", "party_two": "Vishal Bhardwaj",
    }


def test_exact_match_recall_and_mismatch_report() -> None:
    rows = [{"File Name": "a", "Aggrement Value": "100", "Aggrement Start Date": "01.01.2020", "Aggrement End Date": "02.01.2020", "Renewal Notice (Days)": "30", "Party One": "A", "Party Two": "B"}]
    recall, mismatches = evaluate(rows, {"a": {"agreement_value": "100", "agreement_start_date": "01.01.2020", "agreement_end_date": None, "renewal_notice_days": "30", "party_one": "A", "party_two": "B"}})
    assert recall["agreement_value"] == 1
    assert recall["agreement_end_date"] == 0
    assert mismatches[0]["field"] == "agreement_end_date"


def test_api_health_and_invalid_extension() -> None:
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}
    response = client.post("/extract", files={"file": ("agreement.txt", b"hello", "text/plain")})
    assert response.status_code == 415


def test_provider_selection_prefers_anthropic_then_uses_groq(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.setenv("LLM_PROVIDER", "auto")
    assert AgreementExtractor().provider == "anthropic"
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert AgreementExtractor().provider == "groq"
