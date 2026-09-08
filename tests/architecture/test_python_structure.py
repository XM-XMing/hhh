from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "python" / "planning"


def test_cli_package_removed():
    assert not (PACKAGE / "cli").exists()


def test_only_package_root_modules_remain():
    assert {path.name for path in PACKAGE.glob("*.py")} == {"__init__.py", "version.py"}


def test_domain_packages_exist():
    expected = {
        "awac",
        "bc",
        "common",
        "contracts",
        "data",
        "diagnostics",
        "evaluation",
        "mission",
        "primitives",
        "protocol",
        "runtime",
        "safety",
        "teacher",
    }
    actual = {path.name for path in PACKAGE.iterdir() if path.is_dir() and not path.name.startswith("__")}
    assert expected <= actual


def test_scripts_do_not_import_cli():
    for path in (ROOT / "scripts").glob("*.py"):
        assert "planning.cli" not in path.read_text(encoding="utf-8"), path


def test_production_does_not_import_cli():
    for path in PACKAGE.rglob("*.py"):
        assert "planning.cli" not in path.read_text(encoding="utf-8"), path
