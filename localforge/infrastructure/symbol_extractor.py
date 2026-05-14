"""
symbol_extractor.py — tree-sitter を使ったコードシンボル抽出。
Python / JS / TS は tree-sitter AST で精確に抽出する。
SQL は DDL/DML パターンの正規表現で構造を抽出する。
その他の言語は正規表現フォールバックを使用する。

tree-sitter 系パッケージは optional — インストールされていなければ
正規表現フォールバックに自動的に切り替わる。
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

from localforge.domain.models import Symbol

logger = logging.getLogger(__name__)

# tree-sitter パーサーのキャッシュ（言語ごとに1回だけ初期化）
_parsers: dict = {}
_parser_init_attempted: set = set()


def _get_parser(language: str):
    """tree-sitter パーサーを遅延初期化してキャッシュする。失敗時は None。"""
    if language in _parser_init_attempted:
        return _parsers.get(language)
    _parser_init_attempted.add(language)

    try:
        from tree_sitter import Parser
    except ImportError:
        logger.debug("tree-sitter がインストールされていません — 正規表現フォールバックを使用")
        return None

    try:
        if language == "python":
            import tree_sitter_python as tspython
            lang_obj = tspython.language()
        elif language in ("javascript", "js"):
            import tree_sitter_javascript as tsjs
            lang_obj = tsjs.language()
        elif language in ("typescript", "ts"):
            import tree_sitter_typescript as tsts
            lang_obj = tsts.language_typescript()
        elif language in ("tsx",):
            import tree_sitter_typescript as tsts
            lang_obj = tsts.language_tsx()
        else:
            return None

        from tree_sitter import Language
        parser = Parser(Language(lang_obj))
        _parsers[language] = parser
        logger.debug("tree-sitter パーサー初期化完了: %s", language)
        return parser
    except Exception as exc:
        logger.debug("tree-sitter パーサー初期化失敗 (%s): %s", language, exc)
        return None


# ---------------------------------------------------------------------------
# Python 抽出
# ---------------------------------------------------------------------------

def _extract_python(content: str) -> List[Symbol]:
    parser = _get_parser("python")
    if parser is None:
        return _extract_python_regex(content)

    try:
        tree = parser.parse(content.encode("utf-8", errors="replace"))
        return _walk_python(tree.root_node, content, parent=None)
    except Exception as exc:
        logger.debug("tree-sitter Python 解析エラー: %s", exc)
        return _extract_python_regex(content)


def _walk_python(node, content: str, parent: Optional[str]) -> List[Symbol]:
    symbols: List[Symbol] = []
    lines = content.splitlines()

    for child in node.children:
        if child.type == "import_statement":
            names = [n.text.decode("utf-8", errors="replace") for n in child.children if n.type == "dotted_name"]
            for name in names[:3]:  # 最大3つのインポート名
                symbols.append(Symbol(kind="import", name=name, line_start=child.start_point[0] + 1))
        elif child.type == "import_from_statement":
            module_nodes = [n for n in child.children if n.type == "dotted_name"]
            module = module_nodes[0].text.decode("utf-8", errors="replace") if module_nodes else ""
            alias_nodes = [n for n in child.children if n.type in ("dotted_name", "aliased_import") and n != (module_nodes[0] if module_nodes else None)]
            for alias in alias_nodes[:3]:
                name = alias.text.decode("utf-8", errors="replace")
                symbols.append(Symbol(kind="import", name=f"{module}.{name}" if module else name, line_start=child.start_point[0] + 1))
        elif child.type == "class_definition":
            name_node = next((n for n in child.children if n.type == "identifier"), None)
            if name_node:
                class_name = name_node.text.decode("utf-8", errors="replace")
                line_s = child.start_point[0] + 1
                line_e = child.end_point[0] + 1
                sig = lines[child.start_point[0]] if child.start_point[0] < len(lines) else ""
                docstring = _get_python_docstring(child, content)
                symbols.append(Symbol(
                    kind="class", name=class_name, signature=sig.strip(),
                    docstring=docstring, line_start=line_s, line_end=line_e,
                ))
                # クラス内のメソッドを再帰的に抽出
                symbols.extend(_walk_python(child, content, parent=class_name))
        elif child.type == "function_definition":
            name_node = next((n for n in child.children if n.type == "identifier"), None)
            if name_node:
                func_name = name_node.text.decode("utf-8", errors="replace")
                kind = "method" if parent else "function"
                line_s = child.start_point[0] + 1
                line_e = child.end_point[0] + 1
                sig = lines[child.start_point[0]] if child.start_point[0] < len(lines) else ""
                docstring = _get_python_docstring(child, content)
                symbols.append(Symbol(
                    kind=kind, name=func_name, signature=sig.strip(),
                    docstring=docstring, line_start=line_s, line_end=line_e,
                    parent=parent,
                ))
        elif child.type in ("decorated_definition",):
            symbols.extend(_walk_python(child, content, parent=parent))

    return symbols


def _get_python_docstring(node, content: str) -> str:
    """関数またはクラスノードから docstring を取得する。"""
    try:
        body = next((n for n in node.children if n.type == "block"), None)
        if body is None:
            return ""
        first_stmt = next((n for n in body.children if n.type == "expression_statement"), None)
        if first_stmt is None:
            return ""
        string_node = next((n for n in first_stmt.children if n.type in ("string", "concatenated_string")), None)
        if string_node is None:
            return ""
        raw = string_node.text.decode("utf-8", errors="replace")
        # 引用符を除去して最初の100文字
        clean = raw.strip('"\' \t\n').replace('"""', "").replace("'''", "").strip()
        return clean[:100]
    except Exception:
        return ""


def _extract_python_regex(content: str) -> List[Symbol]:
    """tree-sitter 未使用時の Python 正規表現フォールバック。"""
    symbols: List[Symbol] = []
    current_class: Optional[str] = None
    for i, line in enumerate(content.splitlines(), 1):
        stripped = line.lstrip()
        if stripped.startswith("class "):
            m = re.match(r"class\s+(\w+)", stripped)
            if m:
                current_class = m.group(1)
                symbols.append(Symbol(kind="class", name=current_class, signature=stripped.rstrip(":"), line_start=i))
        elif stripped.startswith("def "):
            m = re.match(r"def\s+(\w+)\s*\(", stripped)
            if m:
                indent = len(line) - len(stripped)
                kind = "method" if indent > 0 and current_class else "function"
                if indent == 0:
                    current_class = None
                symbols.append(Symbol(kind=kind, name=m.group(1), signature=stripped.split(":")[0].strip(), line_start=i, parent=current_class if kind == "method" else None))
        elif stripped.startswith("import ") or stripped.startswith("from "):
            m = re.match(r"(?:from\s+(\S+)\s+)?import\s+(\S+)", stripped)
            if m:
                module = m.group(1) or ""
                name = m.group(2).split(" as ")[0]
                symbols.append(Symbol(kind="import", name=f"{module}.{name}" if module else name, line_start=i))
    return symbols


# ---------------------------------------------------------------------------
# JavaScript / TypeScript 抽出
# ---------------------------------------------------------------------------

def _extract_js(content: str, language: str = "javascript") -> List[Symbol]:
    parser = _get_parser(language)
    if parser is None:
        return _extract_js_regex(content)
    try:
        tree = parser.parse(content.encode("utf-8", errors="replace"))
        return _walk_js(tree.root_node, content, parent=None)
    except Exception as exc:
        logger.debug("tree-sitter JS/TS 解析エラー: %s", exc)
        return _extract_js_regex(content)


def _walk_js(node, content: str, parent: Optional[str]) -> List[Symbol]:
    symbols: List[Symbol] = []
    lines = content.splitlines()

    for child in node.children:
        ntype = child.type
        if ntype in ("import_statement", "import_declaration"):
            symbols.extend(_extract_js_imports(child))
        elif ntype in ("class_declaration", "class_expression", "abstract_class_declaration"):
            name_node = next((n for n in child.children if n.type == "type_identifier" or n.type == "identifier"), None)
            if name_node:
                class_name = name_node.text.decode("utf-8", errors="replace")
                line_s = child.start_point[0] + 1
                line_e = child.end_point[0] + 1
                sig_line = lines[child.start_point[0]] if child.start_point[0] < len(lines) else ""
                symbols.append(Symbol(kind="class", name=class_name, signature=sig_line.strip(), line_start=line_s, line_end=line_e))
                symbols.extend(_walk_js(child, content, parent=class_name))
        elif ntype in ("function_declaration", "generator_function_declaration"):
            name_node = next((n for n in child.children if n.type == "identifier"), None)
            if name_node:
                func_name = name_node.text.decode("utf-8", errors="replace")
                line_s = child.start_point[0] + 1
                line_e = child.end_point[0] + 1
                sig_line = lines[child.start_point[0]] if child.start_point[0] < len(lines) else ""
                symbols.append(Symbol(kind="function", name=func_name, signature=sig_line.strip(), line_start=line_s, line_end=line_e, parent=parent))
        elif ntype in ("method_definition", "public_field_definition") and parent:
            name_node = next((n for n in child.children if n.type in ("property_identifier", "private_property_identifier")), None)
            if name_node:
                mname = name_node.text.decode("utf-8", errors="replace")
                line_s = child.start_point[0] + 1
                sig_line = lines[child.start_point[0]] if child.start_point[0] < len(lines) else ""
                symbols.append(Symbol(kind="method", name=mname, signature=sig_line.strip(), line_start=line_s, parent=parent))
        elif ntype in ("lexical_declaration", "variable_declaration", "export_statement"):
            # export function / const fn = () => {}
            symbols.extend(_walk_js(child, content, parent=parent))
        elif ntype in ("arrow_function", "function_expression") and parent is None:
            pass  # 無名関数は除外

    return symbols


def _extract_js_imports(node) -> List[Symbol]:
    symbols: List[Symbol] = []
    try:
        source = next((n for n in node.children if n.type == "string"), None)
        module = source.text.decode("utf-8", errors="replace").strip("'\"`") if source else ""
        symbols.append(Symbol(kind="import", name=module, line_start=node.start_point[0] + 1))
    except Exception:
        pass
    return symbols


def _extract_js_regex(content: str) -> List[Symbol]:
    """tree-sitter 未使用時の JS/TS 正規表現フォールバック。"""
    symbols: List[Symbol] = []
    patterns = [
        (r"^(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(", "function"),
        (r"^(?:export\s+)?class\s+(\w+)", "class"),
        (r"^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\(", "function"),
        (r"^\s+(\w+)\s*\([^)]*\)\s*\{", "method"),
    ]
    for i, line in enumerate(content.splitlines(), 1):
        stripped = line.lstrip()
        for pattern, kind in patterns:
            m = re.match(pattern, stripped)
            if m:
                symbols.append(Symbol(kind=kind, name=m.group(1), signature=stripped[:80], line_start=i))
                break
        if re.match(r"^import\s+", stripped):
            m = re.search(r"from\s+['\"]([^'\"]+)['\"]", stripped)
            if m:
                symbols.append(Symbol(kind="import", name=m.group(1), line_start=i))
    return symbols


# ---------------------------------------------------------------------------
# SQL 抽出 (DDL + DML)
# ---------------------------------------------------------------------------

# DDL パターン: (regex, kind) — 大文字小文字を無視して適用
# グループ 1 = オブジェクト名、グループ 2 = ON 節のテーブル名（INDEX のみ）
_SQL_DDL: list[tuple[re.Pattern, str, bool]] = [
    # CREATE TABLE [IF NOT EXISTS] [schema.]name
    (re.compile(
        r"CREATE\s+(?:OR\s+REPLACE\s+)?(?:TEMP(?:ORARY)?\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(?:\w+\.)?(\w+)",
        re.I), "table", False),
    # CREATE [MATERIALIZED] VIEW [IF NOT EXISTS] [schema.]name
    (re.compile(
        r"CREATE\s+(?:OR\s+REPLACE\s+)?(?:MATERIALIZED\s+)?VIEW\s+(?:IF\s+NOT\s+EXISTS\s+)?(?:\w+\.)?(\w+)",
        re.I), "view", False),
    # CREATE [OR REPLACE] PROCEDURE [schema.]name
    (re.compile(
        r"CREATE\s+(?:OR\s+REPLACE\s+)?PROCEDURE\s+(?:\w+\.)?(\w+)",
        re.I), "procedure", False),
    # CREATE [OR REPLACE] FUNCTION [schema.]name
    (re.compile(
        r"CREATE\s+(?:OR\s+REPLACE\s+)?FUNCTION\s+(?:\w+\.)?(\w+)",
        re.I), "function", False),
    # CREATE [OR REPLACE] TRIGGER [schema.]name
    (re.compile(
        r"CREATE\s+(?:OR\s+REPLACE\s+)?TRIGGER\s+(?:\w+\.)?(\w+)",
        re.I), "trigger", False),
    # CREATE [UNIQUE] INDEX [IF NOT EXISTS] [schema.]index_name ON [schema.]table_name
    (re.compile(
        r"CREATE\s+(?:UNIQUE\s+)?INDEX\s+(?:IF\s+NOT\s+EXISTS\s+)?(?:\w+\.)?(\w+)\s+ON\s+(?:\w+\.)?(\w+)",
        re.I), "index", True),   # True = has ON-table in group 2
    # ALTER TABLE [schema.]name  (track which tables are modified)
    (re.compile(
        r"ALTER\s+TABLE\s+(?:\w+\.)?(\w+)",
        re.I), "alter", False),
]

# FK reference: REFERENCES [schema.]table (col)
_SQL_REFERENCES = re.compile(
    r"REFERENCES\s+(?:\w+\.)?(\w+)\s*\(", re.I
)

# DML target tables
_SQL_DML: list[tuple[re.Pattern, str]] = [
    (re.compile(r"INSERT\s+INTO\s+(?:\w+\.)?(\w+)", re.I), "insert"),
    (re.compile(r"UPDATE\s+(?:\w+\.)?(\w+)\s+SET", re.I), "update"),
    (re.compile(r"DELETE\s+FROM\s+(?:\w+\.)?(\w+)", re.I), "delete"),
    (re.compile(r"MERGE\s+INTO\s+(?:\w+\.)?(\w+)", re.I), "merge"),
]


def _extract_sql(content: str) -> List[Symbol]:
    """
    SQL ファイルから DDL オブジェクト・FK 参照・DML ターゲットを抽出する。

    抽出するもの:
      - CREATE TABLE / VIEW / PROCEDURE / FUNCTION / TRIGGER
      - CREATE [UNIQUE] INDEX … ON table
      - ALTER TABLE
      - REFERENCES table (外部キー参照)
      - INSERT INTO / UPDATE … SET / DELETE FROM / MERGE INTO
    """
    symbols: List[Symbol] = []
    # コメント除去せずに行単位で処理（複数行DDLも先頭行でマッチする）
    lines = content.splitlines()

    # DDL: 各行に対してすべての DDL パターンを試みる
    # 複数行にまたがる CREATE 文のために前後2行を結合してスキャンする
    for i, line in enumerate(lines):
        # 前後を結合してマルチライン対応（最大3行）
        window = " ".join(lines[i : i + 3])
        stripped = line.strip()

        # コメント行をスキップ
        if stripped.startswith("--") or stripped.startswith("/*") or stripped.startswith("*"):
            continue

        for pattern, kind, has_on_table in _SQL_DDL:
            m = pattern.search(window)
            if m:
                name = m.group(1)
                if not name or name.upper() in ("IF", "NOT", "EXISTS", "OR", "REPLACE"):
                    continue
                parent = m.group(2) if has_on_table else None
                sig = stripped[:100]
                symbols.append(Symbol(
                    kind=kind, name=name, signature=sig,
                    line_start=i + 1, parent=parent,
                ))
                break  # 1行に複数の DDL は通常ない

        # FK 参照（同一行内に複数あり得る）
        for m in _SQL_REFERENCES.finditer(window):
            ref_table = m.group(1)
            if ref_table and ref_table.upper() not in ("IF", "NOT", "EXISTS"):
                symbols.append(Symbol(
                    kind="reference", name=ref_table,
                    signature=f"REFERENCES {ref_table}", line_start=i + 1,
                ))

        # DML ターゲット
        for dml_pat, dml_kind in _SQL_DML:
            m = dml_pat.search(window)
            if m:
                tbl = m.group(1)
                if tbl:
                    symbols.append(Symbol(
                        kind=dml_kind, name=tbl,
                        signature=stripped[:80], line_start=i + 1,
                    ))
                break

    # 重複排除: 同一 (kind, name) は先頭行のみ残す
    seen: set[tuple[str, str]] = set()
    deduped: List[Symbol] = []
    for sym in symbols:
        key = (sym.kind, sym.name.lower())
        if key not in seen:
            seen.add(key)
            deduped.append(sym)

    return deduped


# ---------------------------------------------------------------------------
# 正規表現フォールバック（Go, Rust, Java など）
# ---------------------------------------------------------------------------

_REGEX_PATTERNS: dict[str, list[tuple[str, str]]] = {
    "go": [
        (r"^func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(", "function"),
        (r"^type\s+(\w+)\s+struct", "class"),
        (r"^import\s+\"([^\"]+)\"", "import"),
    ],
    "rust": [
        (r"^(?:pub\s+)?fn\s+(\w+)\s*[<(]", "function"),
        (r"^(?:pub\s+)?struct\s+(\w+)", "class"),
        (r"^(?:pub\s+)?enum\s+(\w+)", "class"),
        (r"^use\s+([^;]+);", "import"),
    ],
    "java": [
        (r"(?:public|private|protected|static|\s)+\s+\w+\s+(\w+)\s*\(", "function"),
        (r"(?:public|private|protected)?\s*class\s+(\w+)", "class"),
        (r"^import\s+([^;]+);", "import"),
    ],
    "ruby": [
        (r"^\s*def\s+(\w+)", "function"),
        (r"^\s*class\s+(\w+)", "class"),
        (r"^\s*require\s+['\"]([^'\"]+)['\"]", "import"),
    ],
    "php": [
        (r"^(?:public|private|protected|static|\s)*function\s+(\w+)\s*\(", "function"),
        (r"^(?:abstract\s+)?class\s+(\w+)", "class"),
        (r"^use\s+([^;]+);", "import"),
    ],
}


def _extract_regex_generic(content: str, language: str) -> List[Symbol]:
    patterns = _REGEX_PATTERNS.get(language, [])
    if not patterns:
        # 最後の手段: 汎用 def/func/class パターン
        patterns = [
            (r"(?:def|func|function)\s+(\w+)\s*[\(\{]", "function"),
            (r"class\s+(\w+)", "class"),
        ]
    symbols: List[Symbol] = []
    for i, line in enumerate(content.splitlines(), 1):
        stripped = line.lstrip()
        for pattern, kind in patterns:
            m = re.search(pattern, stripped)
            if m:
                symbols.append(Symbol(kind=kind, name=m.group(1), signature=stripped[:80], line_start=i))
                break
    return symbols


# ---------------------------------------------------------------------------
# メインエントリポイント
# ---------------------------------------------------------------------------

_LANGUAGE_MAP: dict[str, str] = {
    "python": "python",
    "javascript": "javascript",
    "typescript": "typescript",
    "tsx": "tsx",
    "jsx": "javascript",
    "go": "go",
    "rust": "rust",
    "java": "java",
    "ruby": "ruby",
    "php": "php",
    "sql": "sql",
}


def extract_symbols(path: str, content: str, language: Optional[str]) -> List[Symbol]:
    """
    ファイルパス・内容・言語からシンボルを抽出する。

    Args:
        path: ファイルの相対パス（言語判定のフォールバックとして使用）
        content: ファイルの内容
        language: 言語識別子（None の場合はパスから推定）

    Returns:
        抽出された Symbol のリスト
    """
    lang = (language or "").lower()
    if not lang:
        lang = _guess_language(path)

    normalized = _LANGUAGE_MAP.get(lang, lang)

    if normalized == "python":
        return _extract_python(content)
    elif normalized in ("javascript", "js"):
        return _extract_js(content, "javascript")
    elif normalized in ("typescript", "ts"):
        return _extract_js(content, "typescript")
    elif normalized == "tsx":
        return _extract_js(content, "tsx")
    elif normalized == "sql":
        return _extract_sql(content)
    else:
        return _extract_regex_generic(content, normalized)


def _guess_language(path: str) -> str:
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    return {
        "py": "python", "js": "javascript", "mjs": "javascript", "cjs": "javascript",
        "ts": "typescript", "tsx": "tsx", "jsx": "javascript",
        "go": "go", "rs": "rust", "java": "java", "rb": "ruby", "php": "php",
        "c": "c", "cpp": "cpp", "h": "c", "hpp": "cpp",
        "cs": "csharp", "swift": "swift", "kt": "kotlin",
        "sql": "sql", "ddl": "sql", "dml": "sql",
    }.get(ext, "")


def symbols_to_summary(path: str, language: Optional[str], symbols: List[Symbol]) -> str:
    """
    シンボルリストから人間が読みやすい構造化サマリーテキストを生成する。
    このテキストが FileChunk.summary として保存され、BM25 と埋め込みの両方に使われる。

    Args:
        path: ファイルパス
        language: 言語識別子
        symbols: 抽出されたシンボルのリスト

    Returns:
        構造化サマリー文字列
    """
    if not symbols:
        return ""

    lang = (language or "").lower()

    # SQL ファイルはスキーマ特化のサマリーを生成する
    if lang == "sql" or any(s.kind in ("table", "view", "procedure", "trigger", "index", "alter", "reference", "insert", "update", "delete", "merge") for s in symbols):
        return _sql_symbols_to_summary(symbols)

    classes = [s for s in symbols if s.kind == "class"]
    funcs = [s for s in symbols if s.kind in ("function", "method")]
    imports = [s for s in symbols if s.kind == "import"]

    parts: List[str] = []

    if classes:
        class_names = ", ".join(c.name for c in classes[:10])
        parts.append(f"Classes: {class_names}")

    if funcs:
        func_names = ", ".join(f.name for f in funcs[:20])
        parts.append(f"Functions: {func_names}")

    if imports:
        import_names = ", ".join(i.name for i in imports[:10])
        parts.append(f"Imports: {import_names}")

    # シグネチャがある関数の詳細（上位5件）
    detailed = [s for s in funcs[:5] if s.signature and s.name not in ("__init__", "constructor")]
    for sym in detailed:
        sig = sym.signature[:120]
        if sym.docstring:
            parts.append(f"  {sig} — {sym.docstring[:60]}")
        else:
            parts.append(f"  {sig}")

    return "\n".join(parts)


def _sql_symbols_to_summary(symbols: List[Symbol]) -> str:
    """SQL シンボルをスキーマ・DML 別に整理したサマリーを生成する。"""
    tables      = [s for s in symbols if s.kind == "table"]
    altered     = [s for s in symbols if s.kind == "alter"]
    views       = [s for s in symbols if s.kind == "view"]
    procedures  = [s for s in symbols if s.kind == "procedure"]
    functions   = [s for s in symbols if s.kind == "function"]
    triggers    = [s for s in symbols if s.kind == "trigger"]
    indexes     = [s for s in symbols if s.kind == "index"]
    refs        = [s for s in symbols if s.kind == "reference"]
    dml_inserts = [s for s in symbols if s.kind == "insert"]
    dml_updates = [s for s in symbols if s.kind == "update"]
    dml_deletes = [s for s in symbols if s.kind == "delete"]
    dml_merges  = [s for s in symbols if s.kind == "merge"]

    parts: List[str] = []

    if tables:
        parts.append("Tables: " + ", ".join(s.name for s in tables[:20]))
    if views:
        parts.append("Views: " + ", ".join(s.name for s in views[:10]))
    if procedures:
        parts.append("Procedures: " + ", ".join(s.name for s in procedures[:10]))
    if functions:
        parts.append("Functions: " + ", ".join(s.name for s in functions[:10]))
    if triggers:
        parts.append("Triggers: " + ", ".join(s.name for s in triggers[:10]))
    if indexes:
        idx_strs = [
            f"{s.name} ON {s.parent}" if s.parent else s.name
            for s in indexes[:10]
        ]
        parts.append("Indexes: " + ", ".join(idx_strs))
    if refs:
        parts.append("FK References: " + ", ".join(s.name for s in refs[:15]))
    if altered:
        parts.append("Altered tables: " + ", ".join(s.name for s in altered[:10]))

    dml_parts = []
    if dml_inserts:
        dml_parts.append("INSERT→" + ", ".join(s.name for s in dml_inserts[:5]))
    if dml_updates:
        dml_parts.append("UPDATE→" + ", ".join(s.name for s in dml_updates[:5]))
    if dml_deletes:
        dml_parts.append("DELETE→" + ", ".join(s.name for s in dml_deletes[:5]))
    if dml_merges:
        dml_parts.append("MERGE→" + ", ".join(s.name for s in dml_merges[:5]))
    if dml_parts:
        parts.append("DML: " + "; ".join(dml_parts))

    return "\n".join(parts)
