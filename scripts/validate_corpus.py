"""Validate local source documents against data/manifest.json."""

import argparse
import hashlib
import json
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(*, inspect_text: bool) -> list[dict]:
    manifest_path = ROOT / "data" / "manifest.json"
    raw_dir = ROOT / "data" / "raw"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = []

    for document in manifest["documents"]:
        path = raw_dir / document["file_name"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing required document: {path}")

        actual_hash = sha256_file(path)
        if actual_hash != document["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {path.name}")
        if path.stat().st_size != document["file_size_bytes"]:
            raise ValueError(f"File-size mismatch: {path.name}")

        reader = PdfReader(str(path))
        if len(reader.pages) != document["page_count"]:
            raise ValueError(
                f"Page-count mismatch for {path.name}: "
                f"{len(reader.pages)} != {document['page_count']}"
            )

        result = {
            "document_id": document["document_id"],
            "file_name": path.name,
            "pages": len(reader.pages),
            "sha256": actual_hash,
            "text_checked": inspect_text,
        }
        if inspect_text:
            texts = [(page.extract_text() or "") for page in reader.pages]
            result.update(
                {
                    "characters": sum(len(text) for text in texts),
                    "empty_pages": sum(not text.strip() for text in texts),
                    "ocr_candidates": [
                        index + 1
                        for index, text in enumerate(texts)
                        if len(text.strip()) < 20
                    ],
                }
            )
            if result["empty_pages"]:
                raise ValueError(
                    f"{path.name} contains {result['empty_pages']} empty pages"
                )
        results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-text-check",
        action="store_true",
        help="Validate metadata and hashes without extracting all PDF text.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            validate(inspect_text=not args.skip_text_check),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
