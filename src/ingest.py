"""Document ingestion for DOCX and image agreements."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import pytesseract
from docx import Document

logger = logging.getLogger(__name__)
InputKind = Literal["text", "image"]


def read_docx(path: Path) -> str:
    """Read visible body text and table cells from a Word agreement."""
    document = Document(path)
    blocks: list[str] = []
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text:
            blocks.append(text)
    for table in document.tables:
        for row in table.rows:
            text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
            if text:
                blocks.append(text)
    for section in document.sections:
        for container in (section.header, section.footer):
            for paragraph in container.paragraphs:
                text = paragraph.text.strip()
                if text:
                    blocks.append(text)
    return "\n".join(blocks)


def _deskew(image: np.ndarray) -> np.ndarray:
    coordinates = np.column_stack(np.where(image < 250))
    if len(coordinates) < 20:
        return image
    angle = cv2.minAreaRect(coordinates[:, ::-1])[ -1]
    angle = -(90 + angle) if angle < -45 else -angle
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)


def ocr_image(path: Path) -> str:
    """Preprocess a scanned agreement before passing it to Tesseract."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to read image: {path.name}")
    if max(image.shape) < 2200:
        image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    image = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    image = _deskew(image)
    try:
        return pytesseract.image_to_string(image, config="--oem 3 --psm 6")
    except pytesseract.TesseractNotFoundError as error:
        raise RuntimeError("Tesseract OCR is not installed. Install it with `brew install tesseract` (macOS).") from error


def ingest(path: Path) -> tuple[InputKind, str]:
    """Return either extracted text or an image marker for the LLM client."""
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return "text", read_docx(path)
    if suffix in {".png", ".jpg", ".jpeg"}:
        return "image", ocr_image(path)
    raise ValueError("Supported files are .docx, .png, .jpg, and .jpeg")
