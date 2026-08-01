"""Fail when public project files contain likely secrets or internal identifiers."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    "data",
}
TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

PATTERNS = {
    "private key": re.compile("BEGIN " + "PRIVATE KEY"),
    "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "hard-coded password": re.compile(
        r"(?i)(pass" + r"word|pwd)\s*[:=]\s*[\"'][^\"']{4,}[\"']"
    ),
    "hard-coded token": re.compile(
        r"(?i)(api[_-]?key|access[_-]?token)\s*[:=]\s*[\"'][^\"']{8,}[\"']"
    ),
    "employer identifier": re.compile(
        r"(?i)(digi" + r"kala|ods" + r"digikala|sales\.fact_)"
    ),
    "private network address": re.compile(
        r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3})\b"
    ),
}


def candidate_files() -> list[Path]:
    try:
        output = subprocess.check_output(
            ["git", "ls-files"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        files = [ROOT / line for line in output.splitlines()]
    except (FileNotFoundError, subprocess.CalledProcessError):
        files = [path for path in ROOT.rglob("*") if path.is_file()]

    return [
        path
        for path in files
        if path.suffix.lower() in TEXT_SUFFIXES
        and path.name != Path(__file__).name
        and not any(part in EXCLUDED_PARTS for part in path.relative_to(ROOT).parts)
    ]


def main() -> int:
    findings: list[str] = []
    for path in candidate_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for label, pattern in PATTERNS.items():
            if pattern.search(text):
                findings.append(f"{path.relative_to(ROOT)}: {label}")

    if findings:
        print("Sensitive-content check failed:")
        for finding in findings:
            print(f"- {finding}")
        return 1

    print(f"Sensitive-content check passed for {len(candidate_files())} text files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
