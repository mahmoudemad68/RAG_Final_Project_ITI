"""Execute the project notebook reproducibly and persist its cell outputs.

Examples:
    .venv/bin/python scripts/execute_notebook.py --through-cell 20
    .venv/bin/python scripts/execute_notebook.py

The partial mode is useful when Ollama is not installed: cell 20 includes the
complete ingestion, index build, reload check, and retrieval evaluation while
the later cells exercise local generation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NOTEBOOK = ROOT / "notebooks" / "rag_pipeline.ipynb"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--notebook",
        type=Path,
        default=DEFAULT_NOTEBOOK,
        help="Notebook to execute (default: notebooks/rag_pipeline.ipynb)",
    )
    parser.add_argument(
        "--through-cell",
        type=int,
        default=None,
        help="Stop after this zero-based cell index; omit to run every cell",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Per-cell timeout in seconds (default: 3600)",
    )
    return parser.parse_args()


def execute(notebook_path: Path, through_cell: int | None, timeout: int) -> None:
    notebook_path = notebook_path.resolve()
    notebook = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )

    with client.setup_kernel():
        for index, cell in enumerate(notebook.cells):
            if through_cell is not None and index > through_cell:
                break
            if cell.cell_type != "code":
                continue
            print(f"Executing cell {index}/{len(notebook.cells) - 1}", flush=True)
            client.execute_cell(cell, index)
            # Preserve evidence from completed cells even if a later cell fails.
            nbformat.write(notebook, notebook_path)

    nbformat.write(notebook, notebook_path)
    mode = "all cells" if through_cell is None else f"through cell {through_cell}"
    print(f"Executed {mode}: {notebook_path}")


def main() -> None:
    args = parse_args()
    if args.through_cell is not None and args.through_cell < 0:
        raise SystemExit("--through-cell must be zero or greater")
    execute(args.notebook, args.through_cell, args.timeout)


if __name__ == "__main__":
    main()
