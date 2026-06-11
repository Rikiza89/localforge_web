"""
コードバリデーター — 生成/編集後ファイルの構文チェックと自動ロールバックを提供する。
"""

from __future__ import annotations

import ast
import logging
import shutil
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)


def validate(path: Path, content: str) -> Tuple[bool, str]:
    """
    ファイル内容を構文チェックする。

    Returns:
        (ok, error_message) — ok=True なら検証成功
    """
    suffix = path.suffix.lower()

    if suffix == ".py":
        return _validate_python(content, path)

    if suffix in {".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs"}:
        return _validate_braces(content, path)

    if suffix in {".sql", ".ddl", ".dml"}:
        return _validate_sql(content, path)

    return True, ""


def _validate_python(content: str, path: Path) -> Tuple[bool, str]:
    try:
        ast.parse(content, filename=str(path))
        return True, ""
    except SyntaxError as exc:
        return False, f"Python構文エラー: {exc.msg} (行{exc.lineno})"


def _validate_braces(content: str, path: Path) -> Tuple[bool, str]:
    """中括弧・括弧・角括弧のバランスをチェックする。"""
    openers = {"{": "}", "(": ")", "[": "]"}
    closers = {v: k for k, v in openers.items()}
    stack = []
    in_str_single = False
    in_str_double = False
    in_template = False
    i = 0
    while i < len(content):
        ch = content[i]
        # 文字列リテラルの簡易スキップ（ネスト非対応だが大半のケースをカバー）
        if ch == "'" and not in_str_double and not in_template:
            in_str_single = not in_str_single
        elif ch == '"' and not in_str_single and not in_template:
            in_str_double = not in_str_double
        elif ch == "`" and not in_str_single and not in_str_double:
            in_template = not in_template
        elif not in_str_single and not in_str_double and not in_template:
            if ch in openers:
                stack.append(ch)
            elif ch in closers:
                expected_opener = closers[ch]
                if not stack or stack[-1] != expected_opener:
                    return False, f"括弧の不一致: '{ch}' (位置{i})"
                stack.pop()
        i += 1

    if stack:
        return False, f"括弧が閉じられていません: {''.join(stack)}"
    return True, ""


def _validate_sql(content: str, path: Path) -> Tuple[bool, str]:
    """SQL: 空でなければ少なくとも1つのセミコロンを期待する（緩い検証）。"""
    stripped = content.strip()
    if stripped and ";" not in stripped:
        # セミコロンなしは警告レベル — エラーにはしない
        logger.debug("SQL検証: セミコロンなし (%s)", path)
    return True, ""


def restore_backup(file_path: Path) -> bool:
    """
    .bak ファイルが存在する場合に元のファイルを復元する。

    Returns:
        True if restored, False if no backup found
    """
    backup_path = file_path.with_suffix(file_path.suffix + ".bak")
    if not backup_path.exists():
        return False
    try:
        # バイト単位でコピーし、改行コードや BOM を完全に復元する
        shutil.copy2(backup_path, file_path)
        backup_path.unlink(missing_ok=True)
        logger.info("バックアップから復元: %s", file_path)
        return True
    except Exception as exc:
        logger.error("バックアップ復元失敗 [%s]: %s", file_path, exc)
        return False


def delete_backup(file_path: Path) -> None:
    """.bak ファイルを削除する（検証成功後のクリーンアップ用）。"""
    backup_path = file_path.with_suffix(file_path.suffix + ".bak")
    try:
        backup_path.unlink(missing_ok=True)
    except Exception:
        pass
