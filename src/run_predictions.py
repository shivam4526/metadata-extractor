"""CLI runner for cached, concurrent predictions and exact-match evaluation."""
from __future__ import annotations

import argparse
import csv
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .evaluate import evaluate, write_mismatches
from .extractor import AgreementExtractor
from .schemas import CSV_COLUMNS, FIELDS

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _file_index(directory: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in directory.iterdir():
        if not path.is_file():
            continue
        stem = path.name
        # Supports source names such as "agreement.pdf.docx" while CSV stores "agreement".
        while "." in stem:
            stem = stem.rsplit(".", 1)[0]
        index[stem] = path
    return index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    with (args.data_dir / f"{args.split}.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if args.limit:
        rows = rows[:args.limit]
    index = _file_index(args.data_dir / args.split)
    extractor = AgreementExtractor(args.data_dir)
    predictions: dict[str, dict[str, str | None]] = {}

    def extract_row(row: dict[str, str]) -> tuple[str, dict[str, str | None]]:
        name = row["File Name"]
        path = index.get(name)
        if not path:
            raise FileNotFoundError(f"No input document for CSV row {name}")
        return name, extractor.extract(path).model_dump()

    with ThreadPoolExecutor(max_workers=int(os.getenv("MAX_WORKERS", "4"))) as pool:
        futures = {pool.submit(extract_row, row): row["File Name"] for row in rows}
        for future in as_completed(futures):
            name = futures[future]
            try:
                file_name, prediction = future.result()
                predictions[file_name] = prediction
            except Exception as error:
                logger.exception("Prediction failed for %s: %s", name, error)
                predictions[name] = {field: None for field in FIELDS}

    output_columns = ["File Name", *CSV_COLUMNS.values()]
    output = Path(f"predictions_{args.split}.csv")
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_columns)
        writer.writeheader()
        for row in rows:
            prediction = predictions[row["File Name"]]
            writer.writerow({"File Name": row["File Name"], **{column: prediction[field] or "" for field, column in CSV_COLUMNS.items()}})
    recall, mismatches = evaluate(rows, predictions)
    write_mismatches(Path(f"mismatches_{args.split}.csv"), mismatches)
    print("Per-field exact-match recall")
    for field, score in recall.items():
        print(f"{field}: {score:.2%}")
    print(f"Wrote {output} and mismatches_{args.split}.csv")


if __name__ == "__main__":
    main()
