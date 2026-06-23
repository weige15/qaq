from pathlib import Path

import qaq


def test_package_imports() -> None:
    assert qaq.__version__ == "0.0.0"


def test_expected_scaffold_directories_exist() -> None:
    root = Path(__file__).resolve().parents[1]

    for relative_path in (
        "qaq",
        "qaq/runtime",
        "qaq/router",
        "tests/unit",
        "tests/fixtures",
        "configs",
    ):
        assert (root / relative_path).is_dir()
