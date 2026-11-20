
from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ============================================================
# PA-MS PROJECT VALIDATOR
# Read-only project inspection, except for diagnostic reports.
# Does not connect to MT5 or place trades.
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCAN_ROOTS = (
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "research",
    PROJECT_ROOT / "tests",
)

REPORT_DIRECTORY = PROJECT_ROOT / "reports" / "file_validation"

EXPECTED_FILES = (
    "src/config.py",
    "src/logger.py",
    "src/market_data.py",
    "src/mt5_connection.py",
    "src/data_validation.py",
    "research/historical_data.py",
    "research/historical_pipeline.py",
    "research/historical_storage.py",
    "research/historical_validation.py",
    "research/swing_detection/fractal_detector.py",
    "research/swing_detection/swing_models.py",
    "research/swing_detection/structure_classifier.py",
    "research/swing_detection/structure_models.py",
    "research/swing_detection/bos_detector.py",
    "research/swing_detection/choch_detector.py",
    "research/swing_detection/trend_classifier.py",
    "research/swing_detection/alignment_engine.py",
    "research/swing_detection/market_structure_engine.py",
    "research/swing_detection/protected_swing_tracker.py",
    "tests/test_market_data.py",
    "tests/test_historical_pipeline.py",
    "tests/test_fractal_detector.py",
    "tests/test_market_structure_end_to_end.py",
    "tests/diagnostics/run_full_system_diagnostic.py",
)

REQUIRED_DEPENDENCIES = {
    "MetaTrader5": "MetaTrader5",
    "pandas": "pandas",
    "numpy": "numpy",
    "PyYAML": "yaml",
    "python-dotenv": "dotenv",
    "pyarrow": "pyarrow",
    "pytest": "pytest",
}

OPTIONAL_DEPENDENCIES = {
    "matplotlib": "matplotlib",
    "requests": "requests",
    "websocket-client": "websocket",
}

BACKUP_MARKERS = (
    ".backup_",
    ".bak_",
    ".backup",
    ".bak",
    "~",
)


def relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def is_backup_file(path: Path) -> bool:
    name = path.name.lower()
    return any(marker in name for marker in BACKUP_MARKERS)


def is_cache_path(path: Path) -> bool:
    return "__pycache__" in path.parts or path.suffix.lower() == ".pyc"


def discover_files() -> dict[str, Any]:
    result: dict[str, Any] = {
        "all_files": [],
        "python_files": [],
        "backup_files": [],
        "cache_files": [],
        "missing_roots": [],
    }

    for root in SCAN_ROOTS:
        if not root.is_dir():
            result["missing_roots"].append(relative_path(root))
            continue

        for path in root.rglob("*"):
            if not path.is_file():
                continue

            if is_cache_path(path):
                result["cache_files"].append(relative_path(path))
                continue

            result["all_files"].append(relative_path(path))

            if is_backup_file(path):
                result["backup_files"].append(relative_path(path))
                continue

            if path.suffix.lower() == ".py":
                result["python_files"].append(path)

    result["all_files"].sort()
    result["python_files"].sort()
    result["backup_files"].sort()
    result["cache_files"].sort()

    return result


def check_expected_files() -> dict[str, Any]:
    present = []
    missing = []

    for relative in EXPECTED_FILES:
        path = PROJECT_ROOT / relative

        if path.is_file():
            present.append(relative)
        else:
            missing.append(relative)

    return {
        "expected_count": len(EXPECTED_FILES),
        "present": present,
        "missing": missing,
    }


def check_syntax(files: list[Path]) -> dict[str, Any]:
    passed = []
    failed = []

    for path in files:
        try:
            source = path.read_text(
                encoding="utf-8-sig",
                errors="strict",
            )
            compile(source, str(path), "exec")
            passed.append(relative_path(path))

        except (SyntaxError, UnicodeError, OSError) as exc:
            failed.append({
                "file": relative_path(path),
                "error": f"{type(exc).__name__}: {exc}",
            })

    return {
        "checked": len(files),
        "passed": len(passed),
        "failed": failed,
    }


def build_module_index(files: list[Path]) -> tuple[set[str], set[str]]:
    modules: set[str] = set()
    local_roots: set[str] = set()

    def add_module(name: str) -> None:
        if name:
            modules.add(name)
            local_roots.add(name.split(".")[0])

    for root in SCAN_ROOTS:
        if not root.is_dir():
            continue

        root_name = root.name

        # Support package and namespace-package directory imports.
        add_module(root_name)

        for directory in root.rglob("*"):
            if not directory.is_dir():
                continue
            if "__pycache__" in directory.parts:
                continue

            relative = directory.relative_to(root)
            parts = relative.parts

            if parts:
                add_module(".".join((root_name, *parts)))

                if root_name == "src":
                    add_module(".".join(parts))

                if root_name == "research" and parts[0] == "swing_detection":
                    add_module(".".join(parts))

        for path in files:
            try:
                relative = path.relative_to(root)
            except ValueError:
                continue

            if relative.suffix.lower() != ".py":
                continue

            parts = list(relative.with_suffix("").parts)

            if parts[-1] == "__init__":
                parts.pop()

            if not parts:
                continue

            add_module(".".join((root_name, *parts)))

            if root_name == "src":
                add_module(".".join(parts))

            if root_name == "tests":
                add_module(".".join(parts))

            if root_name == "research":
                add_module(".".join(parts))

                if parts[0] == "swing_detection":
                    add_module(".".join(parts))

    return modules, local_roots


def resolve_relative_import(
    current_file: Path,
    import_module: str | None,
    level: int,
) -> str | None:
    try:
        relative = current_file.relative_to(PROJECT_ROOT)
    except ValueError:
        return None

    parts = list(relative.with_suffix("").parts)

    if parts[-1] == "__init__":
        package_parts = parts[:-1]
    else:
        package_parts = parts[:-1]

    if level < 1 or level - 1 > len(package_parts):
        return None

    if level > 1:
        package_parts = package_parts[:len(package_parts) - (level - 1)]

    if import_module:
        package_parts.extend(import_module.split("."))

    return ".".join(package_parts)


def check_internal_imports(
    files: list[Path],
) -> dict[str, Any]:
    modules, local_roots = build_module_index(files)
    unresolved = []
    inspected = 0

    def module_is_available(name: str) -> bool:
        if not name:
            return True

        if name in modules:
            return True

        # A namespace/package parent may be imported directly.
        parts = name.split(".")
        for index in range(1, len(parts)):
            if ".".join(parts[:index]) in modules:
                return True

        return False

    for path in files:
        try:
            source = path.read_text(
                encoding="utf-8-sig",
                errors="strict",
            )
            tree = ast.parse(source, filename=str(path))
        except (SyntaxError, UnicodeError, OSError):
            continue

        for node in ast.walk(tree):
            inspected += 1

            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported = alias.name
                    first_part = imported.split(".")[0]

                    if first_part not in local_roots:
                        continue

                    if not module_is_available(imported):
                        unresolved.append({
                            "file": relative_path(path),
                            "line": node.lineno,
                            "import": imported,
                        })

            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    imported = resolve_relative_import(
                        path,
                        node.module,
                        node.level,
                    )

                    if imported and not module_is_available(imported):
                        unresolved.append({
                            "file": relative_path(path),
                            "line": node.lineno,
                            "import": "." * node.level + (node.module or ""),
                        })

                elif node.module:
                    imported = node.module
                    first_part = imported.split(".")[0]

                    if first_part not in local_roots:
                        continue

                    if not module_is_available(imported):
                        unresolved.append({
                            "file": relative_path(path),
                            "line": node.lineno,
                            "import": imported,
                        })

    return {
        "ast_nodes_inspected": inspected,
        "unresolved_count": len(unresolved),
        "unresolved": unresolved,
        "note": (
            "Static import inspection only. It does not prove that "
            "all imported attributes or runtime dependencies work."
        ),
    }


def check_dependencies() -> dict[str, Any]:
    required = {}
    optional = {}

    for display_name, module_name in REQUIRED_DEPENDENCIES.items():
        try:
            available = importlib.util.find_spec(module_name) is not None
        except (ImportError, ValueError):
            available = False

        required[display_name] = "AVAILABLE" if available else "MISSING"

    for display_name, module_name in OPTIONAL_DEPENDENCIES.items():
        try:
            available = importlib.util.find_spec(module_name) is not None
        except (ImportError, ValueError):
            available = False

        optional[display_name] = "AVAILABLE" if available else "NOT INSTALLED"

    return {
        "required": required,
        "optional": optional,
    }


def run_tests(timeout_seconds: int = 600) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--import-mode=prepend",
        "-p",
        "no:cacheprovider",
        "tests",
    ]

    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"

    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )

        output = (completed.stdout or "") + "\n" + (completed.stderr or "")

        counts = {
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "skipped": 0,
            "warnings": 0,
        }

        patterns = {
            "passed": r"(\d+)\s+passed",
            "failed": r"(\d+)\s+failed",
            "errors": r"(\d+)\s+errors?",
            "skipped": r"(\d+)\s+skipped",
            "warnings": r"(\d+)\s+warnings?",
        }

        for key, pattern in patterns.items():
            matches = re.findall(pattern, output, flags=re.IGNORECASE)
            if matches:
                counts[key] = int(matches[-1])

        return {
            "command": " ".join(command),
            "return_code": completed.returncode,
            "status": "PASS" if completed.returncode == 0 else "FAIL",
            "counts": counts,
            "output": output.strip(),
            "timed_out": False,
        }

    except subprocess.TimeoutExpired as exc:
        return {
            "command": " ".join(command),
            "return_code": None,
            "status": "TIMEOUT",
            "counts": {},
            "output": str(exc),
            "timed_out": True,
        }

    except Exception as exc:
        return {
            "command": " ".join(command),
            "return_code": None,
            "status": "ERROR",
            "counts": {},
            "output": f"{type(exc).__name__}: {exc}",
            "timed_out": False,
        }


def determine_status(report: dict[str, Any]) -> str:
    if report["inventory"]["missing_roots"]:
        return "FAIL"

    if report["architecture"]["missing"]:
        return "FAIL"

    if report["syntax"]["failed"]:
        return "FAIL"

    if report["tests"]["status"] != "PASS":
        return "FAIL"

    required_missing = [
        name
        for name, status in report["dependencies"]["required"].items()
        if status == "MISSING"
    ]

    if required_missing:
        return "FAIL"

    if report["imports"]["unresolved_count"]:
        return "REVIEW"

    if report["inventory"]["backup_files"]:
        return "REVIEW"

    if report["dependencies"]["optional"]:
        if any(
            status == "NOT INSTALLED"
            for status in report["dependencies"]["optional"].values()
        ):
            return "REVIEW"

    return "PASS"


def save_reports(report: dict[str, Any]) -> tuple[Path, Path]:
    REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base_name = f"project_validation_{timestamp}"

    json_path = REPORT_DIRECTORY / f"{base_name}.json"
    text_path = REPORT_DIRECTORY / f"{base_name}.txt"

    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "========== PA-MS PROJECT VALIDATION ==========",
        f"Run time UTC: {report['run_time_utc']}",
        f"Project: {report['project_root']}",
        "",
        f"OVERALL STATUS: {report['overall_status']}",
        "",
        "FILE INVENTORY",
        f"Files discovered: {report['inventory']['total_files']}",
        f"Python files scanned: {report['inventory']['python_files_scanned']}",
        f"Backup files excluded: {len(report['inventory']['backup_files'])}",
        f"Cache files excluded: {len(report['inventory']['cache_files'])}",
        "",
        "SYNTAX",
        f"Checked: {report['syntax']['checked']}",
        f"Passed: {report['syntax']['passed']}",
        f"Failed: {len(report['syntax']['failed'])}",
        "",
        "ARCHITECTURE",
        f"Expected files: {report['architecture']['expected_count']}",
        f"Present: {len(report['architecture']['present'])}",
        f"Missing: {len(report['architecture']['missing'])}",
        "",
        "INTERNAL IMPORTS",
        f"Unresolved references: {report['imports']['unresolved_count']}",
        "",
        "DEPENDENCIES",
    ]

    for name, status in report["dependencies"]["required"].items():
        lines.append(f"Required: {name}: {status}")

    for name, status in report["dependencies"]["optional"].items():
        lines.append(f"Optional: {name}: {status}")

    test_counts = report["tests"].get("counts", {})
    lines.extend([
        "",
        "TEST RESULTS",
        f"Status: {report['tests']['status']}",
        f"Passed: {test_counts.get('passed', 'N/A')}",
        f"Failed: {test_counts.get('failed', 'N/A')}",
        f"Errors: {test_counts.get('errors', 'N/A')}",
        f"Skipped: {test_counts.get('skipped', 'N/A')}",
        "",
        "ITEMS REQUIRING ATTENTION",
    ])

    attention_items = []

    for item in report["inventory"]["missing_roots"]:
        attention_items.append(f"Missing scan directory: {item}")

    for item in report["architecture"]["missing"]:
        attention_items.append(f"Missing expected file: {item}")

    for item in report["syntax"]["failed"]:
        attention_items.append(
            f"Syntax error: {item['file']} - {item['error']}"
        )

    for item in report["imports"]["unresolved"]:
        attention_items.append(
            f"Import review: {item['file']}:{item['line']} -> {item['import']}"
        )

    for item in report["inventory"]["backup_files"]:
        attention_items.append(f"Backup excluded from scan: {item}")

    if report["tests"]["status"] != "PASS":
        attention_items.append(
            f"Test suite status: {report['tests']['status']}"
        )

    if not attention_items:
        lines.append("None.")
    else:
        lines.extend(f"- {item}" for item in attention_items)

    lines.extend([
        "",
        "DETAILED TEST OUTPUT",
        report["tests"].get("output", "No test output available."),
        "",
        "Reports are diagnostic records. Review findings before changing code.",
    ])

    text_path.write_text("\n".join(lines), encoding="utf-8")

    return text_path, json_path


def main() -> int:
    print("\n========== PA-MS PROJECT VALIDATION ==========")
    print("Inspecting project files. No MT5 connection or trade execution.")

    inventory = discover_files()
    architecture = check_expected_files()
    syntax = check_syntax(inventory["python_files"])
    imports = check_internal_imports(inventory["python_files"])
    dependencies = check_dependencies()

    print("\nRunning the existing pytest suite...")
    tests = run_tests()

    report: dict[str, Any] = {
        "run_time_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "inventory": {
            "total_files": len(inventory["all_files"]),
            "python_files_scanned": len(inventory["python_files"]),
            "python_files": [
                relative_path(path)
                for path in inventory["python_files"]
            ],
            "backup_files": inventory["backup_files"],
            "cache_files": inventory["cache_files"],
            "missing_roots": inventory["missing_roots"],
        },
        "architecture": architecture,
        "syntax": syntax,
        "imports": imports,
        "dependencies": dependencies,
        "tests": tests,
    }

    report["overall_status"] = determine_status(report)

    text_path, json_path = save_reports(report)

    print("\n========== VALIDATION SUMMARY ==========")
    print(f"Files discovered:       {report['inventory']['total_files']}")
    print(f"Python files scanned:   {report['inventory']['python_files_scanned']}")
    print(
        f"Syntax checks:          "
        f"{syntax['passed']} passed, {len(syntax['failed'])} failed"
    )
    print(
        f"Architecture:           "
        f"{len(architecture['present'])} present, "
        f"{len(architecture['missing'])} missing"
    )
    print(f"Unresolved imports:     {imports['unresolved_count']}")
    print(f"Test status:            {tests['status']}")

    counts = tests.get("counts", {})
    if counts:
        print(
            f"Tests:                  "
            f"{counts.get('passed', 0)} passed, "
            f"{counts.get('failed', 0)} failed, "
            f"{counts.get('errors', 0)} errors, "
            f"{counts.get('skipped', 0)} skipped"
        )

    print(f"\nOVERALL STATUS: {report['overall_status']}")
    print(f"\nText report: {text_path}")
    print(f"JSON report: {json_path}")

    if syntax["failed"]:
        print("\nSYNTAX ISSUES")
        for item in syntax["failed"]:
            print(f"- {item['file']}: {item['error']}")

    if architecture["missing"]:
        print("\nMISSING EXPECTED FILES")
        for item in architecture["missing"]:
            print(f"- {item}")

    if imports["unresolved"]:
        print("\nIMPORTS REQUIRING REVIEW")
        for item in imports["unresolved"]:
            print(
                f"- {item['file']}:{item['line']} -> {item['import']}"
            )

    if inventory["backup_files"]:
        print(
            f"\nBackup files excluded from code checks: "
            f"{len(inventory['backup_files'])}"
        )

    return 0 if report["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
