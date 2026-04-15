"""
生成サービス — ファイル生成オーケストレーターの責務を担う。
プランからファイルを順次生成・書き込み・コミットする。
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Generator, List, Optional

from localforge.application.context_service import ContextService
from localforge.domain.exceptions import PlanParseError
from localforge.domain.models import GenerationLogEntry, GenerationPlan, PlannedFile
from localforge.infrastructure.filesystem_adapter import FileSystemAdapter
from localforge.infrastructure.git_adapter import GitAdapter
from localforge.infrastructure.index_adapter import IndexAdapter
from localforge.infrastructure.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

_LOCALFORGE_DIR = ".localforge"
# 生成キャンセルフラグ
_cancel_flag: bool = False


def request_cancel() -> None:
    """生成キャンセルを要求する（グローバルフラグを設定）。"""
    global _cancel_flag
    _cancel_flag = True
    logger.info("生成キャンセルが要求されました")


def reset_cancel() -> None:
    """キャンセルフラグをリセットする。"""
    global _cancel_flag
    _cancel_flag = False


class GenerationService:
    """
    プロジェクトファイルのAI生成を担当するサービスクラス。
    プラン生成・承認・ファイル逐次生成・コミットを管理する。
    """

    def __init__(
        self,
        fs: FileSystemAdapter,
        git: GitAdapter,
        index_adapter: IndexAdapter,
        llm: OllamaClient,
        context: ContextService,
    ) -> None:
        """
        GenerationServiceを初期化する。

        Args:
            fs: ファイルシステムアダプター
            git: gitアダプター
            index_adapter: インデックスアダプター
            llm: OllamaクライアントLLMバックエンド
            context: コンテキストサービス
        """
        self._fs = fs
        self._git = git
        self._index_adapter = index_adapter
        self._llm = llm
        self._context = context

    def stream_plan(
        self,
        root: Path,
        model: str,
        user_prompt: str,
        folder_name: str,
        file_tree_text: str,
        context_md: str,
        git_log: str,
    ) -> Generator[dict, None, None]:
        """
        ユーザープロンプトからプロジェクト生成プランをストリーミング生成する。

        Args:
            root: プロジェクトルート
            model: 使用するOllamaモデル名
            user_prompt: ユーザーの自然言語プロンプト
            folder_name: プロジェクトフォルダ名
            file_tree_text: ファイルツリーのテキスト表現
            context_md: context.mdの内容
            git_log: gitログテキスト

        Yields:
            SSEペイロード辞書（token, done, error）
        """
        reset_cancel()
        prompt = self._context.build_plan_prompt(
            user_prompt=user_prompt,
            folder_name=folder_name,
            file_tree_text=file_tree_text,
            context_md=context_md,
            git_log=git_log,
        )

        start_time = time.time()
        try:
            for token in self._llm.stream_completion(model, prompt):
                if _cancel_flag:
                    yield {"error": "キャンセルされました"}
                    return
                yield {"token": token}
        except Exception as exc:
            logger.error("プラン生成エラー: %s", exc)
            yield {"error": str(exc)}
            return

        elapsed = (time.time() - start_time) * 1000
        log_entry = GenerationLogEntry(
            mode="generate",
            model=model,
            operation="plan",
            response_time_ms=elapsed,
        )
        log_path = root / _LOCALFORGE_DIR / "generation_log.jsonl"
        self._index_adapter.append_log_entry(log_path, log_entry)

        yield {"done": True}

    def parse_plan(self, plan_text: str) -> GenerationPlan:
        """
        LLMが生成したプランテキストをGenerationPlanに変換する。
        テキスト内のJSON部分を抽出してパースする。

        Args:
            plan_text: LLMが生成したプランテキスト

        Returns:
            GenerationPlan

        Raises:
            PlanParseError: JSONパースに失敗した場合
        """
        # コードブロックからJSONを抽出
        text = plan_text.strip()
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start)
            text = text[start:end].strip()

        # JSON部分を探す
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            text = text[json_start:json_end]

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise PlanParseError(f"プランJSONのパースに失敗しました: {exc}") from exc

        try:
            return GenerationPlan(
                project_name=data.get("project_name", "unnamed"),
                description=data.get("description", ""),
                files=[
                    PlannedFile(
                        path=f.get("path", ""),
                        description=f.get("description", ""),
                        dependencies=f.get("dependencies", []),
                    )
                    for f in data.get("files", [])
                ],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PlanParseError(f"プランの構造が無効です: {exc}") from exc

    def stream_all_files(
        self,
        root: Path,
        plan: GenerationPlan,
        model: str,
        context_md: str,
        start_from: Optional[int] = None,
    ) -> Generator[dict, None, None]:
        """
        プランに基づいてすべてのファイルを順次生成・書き込みする。
        生成されたファイルは即座にgitコミットされる。

        Args:
            root: プロジェクトルート
            plan: 承認済みGenerationPlan
            model: 使用するOllamaモデル名
            context_md: context.mdの内容
            start_from: 開始インデックス（再開時）

        Yields:
            SSEペイロード辞書（progress, file_written, token, done, error）
        """
        reset_cancel()
        files = plan.files
        total = len(files)
        plan_json = plan.model_dump_json(indent=2)
        start_idx = start_from or 0

        # gitリポジトリの初期化（必要な場合）
        if not (root / ".git").is_dir():
            try:
                self._git.init(root)
            except Exception as exc:
                logger.warning("git init失敗: %s", exc)

        for idx, planned_file in enumerate(files[start_idx:], start=start_idx):
            if _cancel_flag:
                yield {"error": "キャンセルされました"}
                return

            yield {
                "progress": {
                    "done": idx,
                    "total": total,
                    "current_file": planned_file.path,
                }
            }

            # 依存ファイルのコンテンツを収集
            dependency_contents: List[tuple[str, str]] = []
            for dep_path in planned_file.dependencies:
                dep_full = root / dep_path
                if dep_full.exists():
                    try:
                        content = self._fs.read_text(dep_full)
                        dependency_contents.append((dep_path, content))
                    except Exception:
                        pass

            # ファイル生成プロンプトを組み立て
            prompt = self._context.build_file_generation_prompt(
                target_file=planned_file.path,
                target_description=planned_file.description,
                context_md=context_md,
                plan_json=plan_json,
                dependency_contents=dependency_contents,
            )

            # ログエントリを生成開始として記録
            log_path = root / _LOCALFORGE_DIR / "generation_log.jsonl"
            log_entry = GenerationLogEntry(
                mode="generate",
                model=model,
                operation="generate_file",
                file_path=planned_file.path,
                status="pending",
            )
            self._index_adapter.append_log_entry(log_path, log_entry)

            start_time = time.time()
            file_content_parts: List[str] = []

            try:
                for token in self._llm.stream_completion(model, prompt):
                    if _cancel_flag:
                        yield {"error": "キャンセルされました"}
                        return
                    file_content_parts.append(token)
                    yield {"token": token}
            except Exception as exc:
                logger.error("ファイル生成エラー [%s]: %s", planned_file.path, exc)
                yield {"error": str(exc)}
                continue

            elapsed = (time.time() - start_time) * 1000
            file_content = "".join(file_content_parts)

            # ファイルを書き込み
            file_path = root / planned_file.path
            try:
                self._fs.write_text(file_path, file_content)
            except Exception as exc:
                logger.error("ファイル書き込みエラー [%s]: %s", planned_file.path, exc)
                yield {"error": str(exc)}
                continue

            # ファイルを即座にgitコミット
            try:
                self._git.commit_all(
                    root,
                    f"LocalForge: {planned_file.path} を生成",
                )
            except Exception as exc:
                logger.warning("git commit失敗 [%s]: %s", planned_file.path, exc)

            # ログエントリを完了に更新
            self._update_log_status(log_path, planned_file.path, elapsed)

            yield {"file_written": planned_file.path}

        yield {
            "progress": {
                "done": total,
                "total": total,
                "current_file": "完了",
            }
        }
        yield {"done": True}

    def stream_regenerate_file(
        self,
        root: Path,
        plan: GenerationPlan,
        model: str,
        context_md: str,
        file_path: str,
    ) -> Generator[dict, None, None]:
        """
        単一ファイルを再生成してSSEでストリーミングする。

        Args:
            root: プロジェクトルート
            plan: 生成プラン
            model: 使用するモデル名
            context_md: context.mdの内容
            file_path: 再生成するファイルの相対パス

        Yields:
            SSEペイロード辞書
        """
        reset_cancel()
        planned_file = next(
            (f for f in plan.files if f.path == file_path), None
        )
        if not planned_file:
            yield {"error": f"プランにファイルが見つかりません: {file_path}"}
            return

        # 依存ファイルのコンテンツを収集
        dependency_contents: List[tuple[str, str]] = []
        for dep_path in planned_file.dependencies:
            dep_full = root / dep_path
            if dep_full.exists():
                try:
                    content = self._fs.read_text(dep_full)
                    dependency_contents.append((dep_path, content))
                except Exception:
                    pass

        prompt = self._context.build_file_generation_prompt(
            target_file=planned_file.path,
            target_description=planned_file.description,
            context_md=context_md,
            plan_json=plan.model_dump_json(indent=2),
            dependency_contents=dependency_contents,
        )

        file_content_parts: List[str] = []
        try:
            for token in self._llm.stream_completion(model, prompt):
                if _cancel_flag:
                    yield {"error": "キャンセルされました"}
                    return
                file_content_parts.append(token)
                yield {"token": token}
        except Exception as exc:
            yield {"error": str(exc)}
            return

        file_content = "".join(file_content_parts)
        full_path = root / planned_file.path
        try:
            self._fs.write_text(full_path, file_content)
        except Exception as exc:
            yield {"error": str(exc)}
            return

        try:
            self._git.commit_all(
                root,
                f"LocalForge: {planned_file.path} を再生成",
            )
        except Exception as exc:
            logger.warning("git commit失敗 [%s]: %s", planned_file.path, exc)

        yield {"file_written": planned_file.path}
        yield {"done": True}

    def _update_log_status(
        self, log_path: Path, file_path: str, elapsed_ms: float
    ) -> None:
        """
        generation_log.jsonl内の特定ファイルの最新エントリを完了状態に更新する内部メソッド。

        Args:
            log_path: ログファイルパス
            file_path: 対象ファイルパス
            elapsed_ms: 処理時間（ミリ秒）
        """
        if not log_path.exists():
            return

        entries = self._index_adapter.load_log_entries(log_path)
        updated = []
        last_updated = False

        # 後ろから検索して最新のpendingエントリを更新
        for e in reversed(entries):
            if not last_updated and e.file_path == file_path and e.status == "pending":
                e.status = "completed"
                e.response_time_ms = elapsed_ms
                last_updated = True
            updated.append(e)

        updated.reverse()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as fh:
            for e in updated:
                fh.write(e.model_dump_json() + "\n")
