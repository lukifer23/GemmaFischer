from __future__ import annotations

import ast
import hashlib
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

FORBIDDEN_PREFIXES = (
    "checkpoints/",
    "models/",
    "reports/",
    "training_reports/",
    "src/config/",
    "src/inference/",
    "src/training/",
    "src/web/",
)
FORBIDDEN_IMPORT_ROOTS = {"config", "inference", "training", "web"}
FORBIDDEN_EXACT_PATHS = {"src/__init__.py"}
GENERATED_PARTS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^]]+\]\(([^)]+)\)")


def audit_repository(root: Path) -> dict[str, Any]:
    repository_files = tuple(
        line
        for line in subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        if line
    )
    hashes: defaultdict[str, list[str]] = defaultdict(list)
    generated: list[str] = []
    forbidden_paths: list[str] = []
    forbidden_imports: list[dict[str, object]] = []
    broken_local_links: list[dict[str, str]] = []
    for relative in repository_files:
        path = root / relative
        if not path.is_file():
            continue
        if any(part in GENERATED_PARTS for part in path.parts):
            generated.append(relative)
        if relative in FORBIDDEN_EXACT_PATHS or relative.startswith(FORBIDDEN_PREFIXES):
            forbidden_paths.append(relative)
        content = path.read_bytes()
        if content:
            hashes[hashlib.sha256(content).hexdigest()].append(relative)
        if relative.startswith("src/gemmafischer/") and path.suffix == ".py":
            tree = ast.parse(content, filename=relative)
            for node in ast.walk(tree):
                imported: str | None = None
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imported = alias.name.split(".", 1)[0]
                        if imported in FORBIDDEN_IMPORT_ROOTS:
                            forbidden_imports.append(
                                {"path": relative, "line": node.lineno, "module": imported}
                            )
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    imported = node.module.split(".", 1)[0]
                    if imported in FORBIDDEN_IMPORT_ROOTS:
                            forbidden_imports.append(
                                {"path": relative, "line": node.lineno, "module": imported}
                            )
        if path.suffix.lower() == ".md":
            text = content.decode("utf-8")
            for target in MARKDOWN_LINK.findall(text):
                target = target.strip().split("#", 1)[0].split("?", 1)[0]
                if not target or "://" in target or target.startswith(("mailto:", "/")):
                    continue
                resolved = (path.parent / target).resolve()
                if not resolved.exists():
                    broken_local_links.append({"path": relative, "target": target})
    duplicates = [paths for paths in hashes.values() if len(paths) > 1]
    findings = {
        "exact_duplicate_groups": duplicates,
        "forbidden_paths": sorted(forbidden_paths),
        "generated_files": sorted(generated),
        "forbidden_imports": forbidden_imports,
        "broken_local_links": broken_local_links,
    }
    return {
        "schema_version": "1.0",
        "status": "passed" if not any(findings.values()) else "blocked",
        "repository_file_count": len(repository_files),
        "findings": findings,
    }
