"""
dependency_resolver.py — Import-to-file-path resolution for cross-file dependency tracking.

Resolves Python import statements (absolute and relative) and JS/TS imports to
actual project-relative file paths. Returns only paths that exist in the project.
"""

from __future__ import annotations

import ast
import logging
import posixpath
import re
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Matches: import X from './path', import X from "../path", require('./path')
_JS_IMPORT_RE = re.compile(
    r"""(?:import\s+[^'"]*?\s+from\s+|import\s*\(\s*|require\s*\(\s*)['"](\.\.?/[^'"]+)['"]""",
    re.MULTILINE,
)

_JS_EXTS = (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs")


def resolve_imports(
    file_path: str,
    content: str,
    language: Optional[str],
    all_paths: Set[str],
) -> List[str]:
    """
    Resolve import statements in a source file to project-relative file paths.

    Only returns paths that actually exist in the project (present in all_paths).
    Third-party and stdlib imports are silently skipped.

    Args:
        file_path: project-relative path of the file being analyzed (forward slashes)
        content: file content string
        language: detected language identifier
        all_paths: set of all project-relative file paths (forward slashes)

    Returns:
        Deduplicated list of resolved project-relative paths this file imports from.
    """
    lang = (language or "").lower()
    if lang == "python":
        return _resolve_python(file_path, content, all_paths)
    if lang in ("javascript", "typescript", "tsx", "jsx"):
        return _resolve_js(file_path, content, all_paths)
    return []


# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------

def _resolve_python(file_path: str, content: str, all_paths: Set[str]) -> List[str]:
    resolved: List[str] = []
    seen: Set[str] = set()

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []

    # directory parts of the importing file (e.g. ["localforge", "application"])
    norm = file_path.replace("\\", "/")
    file_dir_parts = norm.split("/")[:-1]

    def _add(path: str) -> None:
        if path not in seen:
            seen.add(path)
            resolved.append(path)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            # import foo.bar  →  check foo/bar.py, foo/bar/__init__.py
            for alias in node.names:
                for p in _py_module_to_paths(alias.name, all_paths):
                    _add(p)

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            level = node.level  # 0=absolute, 1=current pkg, 2=parent pkg, …

            if level == 0:
                # Absolute: from package.module import X
                if module:
                    for p in _py_module_to_paths(module, all_paths):
                        _add(p)
                    # X might itself be a submodule
                    for alias in node.names:
                        for p in _py_module_to_paths(f"{module}.{alias.name}", all_paths):
                            _add(p)
                else:
                    for alias in node.names:
                        for p in _py_module_to_paths(alias.name, all_paths):
                            _add(p)

            else:
                # Relative: navigate up (level-1) directories from the file's dir
                base_parts = list(file_dir_parts)
                for _ in range(level - 1):
                    if base_parts:
                        base_parts.pop()
                base_dir = "/".join(base_parts)

                if module:
                    mod_path = module.replace(".", "/")
                    candidate_base = f"{base_dir}/{mod_path}" if base_dir else mod_path
                    for p in _path_candidates(candidate_base, all_paths):
                        _add(p)
                    # also try module.name as submodule
                    for alias in node.names:
                        sub = f"{candidate_base}/{alias.name}"
                        for p in _path_candidates(sub, all_paths):
                            _add(p)
                else:
                    # from . import name1, name2  →  name might be a module file
                    for alias in node.names:
                        name_base = f"{base_dir}/{alias.name}" if base_dir else alias.name
                        for p in _path_candidates(name_base, all_paths):
                            _add(p)

    return resolved


def _py_module_to_paths(module: str, all_paths: Set[str]) -> List[str]:
    return _path_candidates(module.replace(".", "/"), all_paths)


def _path_candidates(slash_path: str, all_paths: Set[str]) -> List[str]:
    """Return existing project paths for a slash-separated module path."""
    results: List[str] = []
    for candidate in (f"{slash_path}.py", f"{slash_path}/__init__.py"):
        if candidate in all_paths:
            results.append(candidate)
    return results


# ---------------------------------------------------------------------------
# JavaScript / TypeScript
# ---------------------------------------------------------------------------

def _resolve_js(file_path: str, content: str, all_paths: Set[str]) -> List[str]:
    resolved: List[str] = []
    seen: Set[str] = set()

    file_dir = "/".join(file_path.replace("\\", "/").split("/")[:-1])

    for match in _JS_IMPORT_RE.finditer(content):
        raw = match.group(1)
        base = f"{file_dir}/{raw}" if file_dir else raw
        normalized = posixpath.normpath(base)

        # Try with explicit extension, then common extension guesses
        candidates: List[str] = []
        last_seg = normalized.split("/")[-1]
        if "." in last_seg and not last_seg.endswith("/"):
            candidates.append(normalized)
        else:
            for ext in _JS_EXTS:
                candidates.append(normalized + ext)
            candidates.append(normalized + "/index.js")
            candidates.append(normalized + "/index.ts")
            candidates.append(normalized + "/index.tsx")

        for c in candidates:
            if c in all_paths and c not in seen:
                seen.add(c)
                resolved.append(c)
                break

    return resolved


# ---------------------------------------------------------------------------
# Dependency graph helpers (used by AnalysisService)
# ---------------------------------------------------------------------------

def build_imported_by(chunks) -> Dict[str, List[str]]:
    """
    Build a reverse dependency map: file → list of files that import it.

    Args:
        chunks: iterable of FileChunk objects with imports_resolved populated

    Returns:
        Dict mapping each imported file path to a list of files that import it.
    """
    imported_by: Dict[str, List[str]] = {}
    for chunk in chunks:
        for dep in chunk.imports_resolved:
            if dep not in imported_by:
                imported_by[dep] = []
            if chunk.path not in imported_by[dep]:
                imported_by[dep].append(chunk.path)
    return imported_by
