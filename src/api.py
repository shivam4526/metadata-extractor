"""FastAPI interface for one agreement document at a time."""
from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from .extractor import AgreementExtractor

logger = logging.getLogger(__name__)
app = FastAPI(title="Agreement Metadata Extractor", version="1.0.0")
ALLOWED_SUFFIXES = {".docx", ".png", ".jpg", ".jpeg"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/extract")
async def extract(file: UploadFile = File(...)) -> dict[str, str | None]:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=415, detail="Upload a .docx, .png, .jpg, or .jpeg file")
    temp_dir = Path(tempfile.mkdtemp(prefix="agreement-extract-"))
    temp_file = temp_dir / f"upload{suffix}"
    try:
        with temp_file.open("wb") as destination:
            shutil.copyfileobj(file.file, destination)
        return AgreementExtractor().extract(temp_file).model_dump()
    except RuntimeError as error:
        logger.exception("Extraction failed")
        raise HTTPException(status_code=502, detail=str(error)) from error
    except Exception as error:
        logger.exception("Invalid upload")
        raise HTTPException(status_code=400, detail=f"Could not process file: {error}") from error
    finally:
        await file.close()
        shutil.rmtree(temp_dir, ignore_errors=True)
