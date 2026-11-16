import json
from pathlib import Path

NOTEBOOKS = Path(__file__).parent.parent / "notebooks"


def test_the_project_has_its_exploring_notebook() -> None:
    assert (NOTEBOOKS / "explore.ipynb").is_file()


def test_every_notebook_is_saved_without_outputs() -> None:
    # An output can hold a key or a paid answer; Git keeps it for good.
    for path in NOTEBOOKS.glob("*.ipynb"):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert cell["outputs"] == [], path.name
                assert cell["execution_count"] is None, path.name
