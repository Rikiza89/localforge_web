"""
インデックスアダプター — ProjectIndex JSONL永続化の実装。
IndexPortインターフェースを実装する。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from localforge.domain.models import FileChunk, GenerationLogEntry, ProjectIndex

logger = logging.getLogger(__name__)

class IndexAdapter:
    """
    ProjectIndexおよびFileChunkのJSONL永続化を担当するアダプタークラス。
    すべてのファイル操作にpathlib.Pathを使用する。
    """

    def save_chunks(self, path: Path, chunks: List[FileChunk]) -> None:
        """
        FileChunkのリストをJSONL形式で保存する（上書き）。

        Args:
            path: 保存先ファイルパス（index.jsonl）
            chunks: 保存するFileChunkのリスト
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for chunk in chunks:
                line = chunk.model_dump_json()
                fh.write(line + "\n")
        logger.debug("チャンク保存完了: %d件 → %s", len(chunks), path)

    def load_chunks(self, path: Path) -> List[FileChunk]:
        """
        JSONL形式のFileChunkを読み込む。

        Args:
            path: 読み込み元ファイルパス（index.jsonl）

        Returns:
            FileChunkのリスト（ファイルが存在しない場合は空リスト）
        """
        if not path.exists():
            return []

        chunks: List[FileChunk] = []
        with path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    chunks.append(FileChunk.model_validate_json(line))
                except (json.JSONDecodeError, ValueError) as exc:
                    logger.warning(
                        "チャンクパースエラー (行 %d): %s", lineno, exc
                    )
        logger.debug("チャンク読み込み完了: %d件 ← %s", len(chunks), path)
        return chunks

    def save_index(self, path: Path, index: ProjectIndex) -> None:
        """
        ProjectIndexをJSONファイルとして保存する。

        Args:
            path: 保存先ファイルパス（project_index.json）
            index: 保存するProjectIndex
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(index.model_dump_json(indent=2), encoding="utf-8")
        logger.debug("ProjectIndex保存完了: %s", path)

    def load_index(self, path: Path) -> Optional[ProjectIndex]:
        """
        JSONファイルからProjectIndexを読み込む。

        Args:
            path: 読み込み元ファイルパス（project_index.json）

        Returns:
            ProjectIndex（ファイルが存在しない場合はNone）
        """
        if not path.exists():
            return None
        try:
            return ProjectIndex.model_validate_json(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error("ProjectIndexパースエラー: %s", exc)
            return None

    def append_log_entry(self, path: Path, entry: GenerationLogEntry) -> None:
        """
        生成ログエントリをJSONL形式で追記する。

        Args:
            path: ログファイルパス（generation_log.jsonl）
            entry: 追記するログエントリ
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(entry.model_dump_json() + "\n")

    def load_log_entries(self, path: Path) -> List[GenerationLogEntry]:
        """
        生成ログエントリをすべて読み込む。

        Args:
            path: ログファイルパス（generation_log.jsonl）

        Returns:
            GenerationLogEntryのリスト
        """
        if not path.exists():
            return []

        entries: List[GenerationLogEntry] = []
        with path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(GenerationLogEntry.model_validate_json(line))
                except (json.JSONDecodeError, ValueError) as exc:
                    logger.warning(
                        "ログエントリパースエラー (行 %d): %s", lineno, exc
                    )
        return entries
