"""
生成サービス — ファイル生成オーケストレーターの責務を担う。
プランからファイルを順次生成・書き込み・コミットする。
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Generator, List, Optional, Tuple

from localforge.application.context_service import ContextService
from localforge.domain.exceptions import PlanParseError
from localforge.domain.models import LOCALFORGE_DIR as _LOCALFORGE_DIR, GenerationLogEntry, GenerationPlan, PlannedFile
from localforge.infrastructure.code_validator import delete_backup, restore_backup, validate
from localforge.infrastructure.filesystem_adapter import FileSystemAdapter
from localforge.infrastructure.git_adapter import GitAdapter
from localforge.infrastructure.index_adapter import IndexAdapter
from localforge.infrastructure.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

# 生成キャンセルフラグ (threading.Event for thread-safety across concurrent requests)
_cancel_event: threading.Event = threading.Event()


def _sanitize_path(raw: str) -> str:
    """
    LLMが出力したファイルパスをプロジェクトルート相対の安全なパスに正規化する。
    絶対パス・..セグメント・NULバイトを含むパスは空文字列を返す（拒否）。

    Args:
        raw: LLMが出力した生パス文字列

    Returns:
        正規化済みの相対パス。安全でない場合は空文字列。
    """
    if not raw or "\x00" in raw:
        return ""
    stripped = raw.strip()
    # 絶対パス（/ や Windows の C:\ 形式）は即座に拒否
    if stripped.startswith("/") or (len(stripped) >= 2 and stripped[1] == ":"):
        return ""
    # 先頭の ./ を除去
    p = Path(stripped)
    if p.is_absolute():
        return ""
    # ".." セグメントを検出
    if any(part == ".." for part in p.parts):
        return ""
    normalized = str(p)
    if not normalized or normalized == ".":
        return ""
    return normalized


def is_cancelled() -> bool:
    """現在キャンセルが要求されているかを返す。"""
    return _cancel_event.is_set()


def request_cancel() -> None:
    """生成キャンセルを要求する。"""
    _cancel_event.set()
    logger.info("生成キャンセルが要求されました")


def reset_cancel() -> None:
    """キャンセルフラグをリセットする。"""
    _cancel_event.clear()


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

    def generate_context_md(self, root: Path, model: str, plan: "GenerationPlan", project_svc: object) -> None:
        """
        完成したプランからcontext.mdを一括生成して保存する。
        バックグラウンドスレッドで呼び出すことを前提とした非ストリーミング実装。

        Args:
            root: プロジェクトルート
            model: 使用するOllamaモデル名
            plan: 完了済みGenerationPlan
            project_svc: save_context_md(root, content)を持つProjectServiceインスタンス
        """
        try:
            plan_json = plan.model_dump_json(indent=2)
            prompt = self._context.build_context_from_plan_prompt(plan_json)
            tokens = []
            for token in self._llm.stream_completion(model, prompt, keep_alive="2h"):
                tokens.append(token)
            content = "".join(tokens).strip()
            if content:
                project_svc.save_context_md(root, content)
                logger.info("context.md を生成しました (%d 文字): %s", len(content), root)
        except Exception as exc:
            logger.warning("context.md 生成エラー: %s", exc)

    def update_context_md_incremental(self, root: Path, model: str, file_path: str, project_svc: object) -> None:
        """
        単一ファイル再生成後にcontext.mdをインクリメンタル更新する。
        バックグラウンドスレッドで呼び出すことを前提とした非ストリーミング実装。

        Args:
            root: プロジェクトルート
            model: 使用するOllamaモデル名
            file_path: 再生成したファイルの相対パス
            project_svc: get_context_md / save_context_md を持つProjectServiceインスタンス
        """
        try:
            fp = root / file_path
            if not fp.exists():
                return
            lines = fp.read_text(encoding="utf-8", errors="replace").splitlines()
            first_200 = "\n".join(lines[:200])
            existing = project_svc.get_context_md(root)
            prompt = self._context.build_context_update_prompt(existing, file_path, first_200)
            tokens = []
            for token in self._llm.stream_completion(model, prompt, keep_alive="2h"):
                tokens.append(token)
            content = "".join(tokens).strip()
            if content:
                project_svc.save_context_md(root, content)
                logger.info("context.md をインクリメンタル更新しました [%s]: %s", file_path, root)
        except Exception as exc:
            logger.warning("context.md インクリメンタル更新エラー: %s", exc)

    def stream_plan(
        self,
        root: Path,
        model: str,
        user_prompt: str,
        folder_name: str,
        file_tree_text: str,
        context_md: str,
        git_log: str,
        file_summaries: Optional[List[tuple[str, str]]] = None,
        project_index_json: Optional[str] = None,
        pinned_contents: Optional[List[tuple[str, str]]] = None,
        workspace_summaries: Optional[List[tuple[str, str]]] = None,
        max_files: Optional[int] = None,
        min_files: Optional[int] = None,
        language: str = "en",
    ) -> Generator[dict, None, None]:
        """
        ユーザープロンプトからプロジェクト生成プランをストリーミング生成する。
        既存プロジェクトのRAGサマリーがあれば渡すことでコンテキスト精度が上がる。

        Args:
            root: プロジェクトルート
            model: 使用するOllamaモデル名
            user_prompt: ユーザーの自然言語プロンプト
            folder_name: プロジェクトフォルダ名
            file_tree_text: ファイルツリーのテキスト表現
            context_md: context.mdの内容
            git_log: gitログテキスト
            file_summaries: RAGで選出した既存ファイルサマリーのリスト（任意）
            project_index_json: ProjectIndex JSON（プロジェクト全体概要、任意）

        Yields:
            SSEペイロード辞書（token, done, error）
        """
        reset_cancel()
        prompt, tokens = self._context.build_plan_prompt(
            user_prompt=user_prompt,
            folder_name=folder_name,
            file_tree_text=file_tree_text,
            context_md=context_md,
            git_log=git_log,
            file_summaries=file_summaries,
            model_name=model,
            project_index_json=project_index_json,
            pinned_contents=pinned_contents,
            workspace_summaries=workspace_summaries,
            max_files=max_files,
            min_files=min_files,
            language=language,
        )

        start_time = time.time()
        try:
            for token in self._llm.stream_completion(model, prompt):
                if is_cancelled():
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
            prompt_tokens_estimated=tokens,
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
            raw_files = data.get("files", [])
            validated_files: list[PlannedFile] = []
            skipped: list[str] = []
            for f in raw_files:
                raw_path = f.get("path", "").strip()
                safe_path = _sanitize_path(raw_path)
                if not safe_path:
                    skipped.append(raw_path or "<空>")
                    logger.warning("プランパスを拒否しました（安全でないパス）: %r", raw_path)
                    continue
                validated_files.append(PlannedFile(
                    path=safe_path,
                    description=f.get("description", ""),
                    dependencies=[
                        s for d in f.get("dependencies", [])
                        if (s := _sanitize_path(str(d)))
                    ],
                    action=f.get("action", "create"),
                    modification_notes=f.get("modification_notes"),
                ))
            plan = GenerationPlan(
                project_name=data.get("project_name", "unnamed"),
                description=data.get("description", ""),
                files=validated_files,
            )
            if skipped:
                logger.warning(
                    "プランから %d 個の安全でないパスを除外しました: %s",
                    len(skipped), skipped,
                )
            return plan
        except (KeyError, TypeError, ValueError) as exc:
            raise PlanParseError(f"プランの構造が無効です: {exc}") from exc

    def stream_all_files(
        self,
        root: Path,
        plan: GenerationPlan,
        model: str,
        context_md: str,
        start_from: Optional[int] = None,
        language: str = "en",
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

        # .bak ファイルを git 管理対象外にする（.gitignore への追記）
        self._ensure_bak_ignored(root)

        # 生成前チェックポイント（ロールバック基点）
        if start_idx == 0:
            cp_hash = self._git.create_checkpoint(root, plan.project_name)
            if cp_hash:
                yield {"checkpoint": cp_hash, "status": f"🔖 チェックポイント作成: {cp_hash}"}

        for idx, planned_file in enumerate(files[start_idx:], start=start_idx):
            if is_cancelled():
                yield {"error": "キャンセルされました"}
                return

            yield {
                "progress": {
                    "done": idx,
                    "total": total,
                    "current_file": planned_file.path,
                },
                "status": f"⏳ {planned_file.path} を生成中...",
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

            file_path = root / planned_file.path
            # パス traversal 防御: 解決済みパスがプロジェクトルート内に収まることを確認する
            try:
                resolved_file = file_path.resolve()
                resolved_root = root.resolve()
                if not resolved_file.is_relative_to(resolved_root):
                    logger.error(
                        "パス traversal を検出してスキップ: %s → %s",
                        planned_file.path, resolved_file,
                    )
                    yield {
                        "warning": f"スキップ: プロジェクト外パス — {planned_file.path}",
                        "status": f"⚠ スキップ: {planned_file.path}",
                    }
                    continue
            except Exception as exc:
                logger.error("パス検証エラー: %s — %s", planned_file.path, exc)
                yield {"warning": f"スキップ: パス検証エラー — {planned_file.path}"}
                continue

            # ファイルが既に存在する場合は action="create" でも修正路に昇格する
            file_exists_on_disk = file_path.exists()
            is_modify = planned_file.action == "modify" or (
                planned_file.action == "create" and file_exists_on_disk
            )
            if planned_file.action == "create" and file_exists_on_disk:
                logger.info(
                    "既存ファイルのため create → modify に昇格: %s", planned_file.path
                )
            log_path = root / _LOCALFORGE_DIR / "generation_log.jsonl"
            start_time = time.time()

            if is_modify:
                # ── DIFF路: SEARCH/REPLACE ブロックで差分のみ生成して適用 ──
                operation = "edit_file"
                log_entry = GenerationLogEntry(
                    mode="generate", model=model, operation=operation,
                    file_path=planned_file.path, status="pending",
                )
                self._index_adapter.append_log_entry(log_path, log_entry)

                existing_content = self._create_backup(file_path)
                modification_notes = planned_file.modification_notes or planned_file.description

                # ファイルをチャンク分割（コンテキスト窓に収まるサイズに）
                max_chars = self._context.max_chunk_chars()
                chunks = self._split_into_chunks(existing_content, max_chars)
                total_chunks = len(chunks)

                all_diff_parts: List[str] = []
                generation_error = False

                for chunk_idx, chunk_content in enumerate(chunks):
                    if is_cancelled():
                        yield {"error": "キャンセルされました"}
                        return

                    if total_chunks > 1:
                        yield {"status": (
                            f"{planned_file.path}: "
                            f"チャンク {chunk_idx + 1}/{total_chunks} を分析中..."
                        )}

                    prompt, tokens = self._context.build_file_diff_prompt(
                        target_file=planned_file.path,
                        modification_notes=modification_notes,
                        chunk_content=chunk_content,
                        context_md=context_md,
                        chunk_idx=chunk_idx,
                        total_chunks=total_chunks,
                    )
                    log_entry.prompt_tokens_estimated += tokens

                    chunk_parts: List[str] = []
                    try:
                        for token in self._llm.stream_completion(model, prompt):
                            if is_cancelled():
                                yield {"error": "キャンセルされました"}
                                return
                            chunk_parts.append(token)
                            yield {"token": token}
                    except Exception as exc:
                        logger.error(
                            "diff生成エラー [%s] チャンク%d: %s",
                            planned_file.path, chunk_idx, exc,
                        )
                        yield {"error": str(exc)}
                        generation_error = True
                        break

                    all_diff_parts.append("".join(chunk_parts))

                if generation_error:
                    continue

                # 全チャンクの SEARCH/REPLACE ブロックを original に適用
                combined_diff = "\n".join(all_diff_parts)
                modified_content, applied, failed = self._apply_search_replace_blocks(
                    existing_content, combined_diff
                )

                # 失敗した場合のリトライ（1回のみ）
                if failed and not is_cancelled():
                    yield {"status": f"🔄 {planned_file.path}: ブロック不一致のためリトライ中..."}
                    retry_diff_parts = []
                    for chunk_idx, chunk_content in enumerate(chunks):
                        prompt = self._context.build_file_diff_prompt(
                            target_file=planned_file.path,
                            modification_notes=f"【リトライ】以前の生成で以下のコードブロックが一致しませんでした。より正確なSEARCHブロックを使用して再試行してください：\n" + "\n".join(failed),
                            chunk_content=chunk_content,
                            context_md=context_md,
                            chunk_idx=chunk_idx,
                            total_chunks=total_chunks,
                        )
                        chunk_parts = []
                        for token in self._llm.stream_completion(model, prompt):
                            if is_cancelled(): break
                            chunk_parts.append(token)
                        retry_diff_parts.append("".join(chunk_parts))
                        if is_cancelled(): break

                    if not is_cancelled():
                        combined_diff = "\n".join(retry_diff_parts)
                        modified_content, applied, failed = self._apply_search_replace_blocks(
                            existing_content, combined_diff
                        )

                elapsed = (time.time() - start_time) * 1000
                self._update_log_status(log_path, planned_file.path, elapsed)

                if applied == 0:
                    yield {"status": (
                        f"ℹ {planned_file.path}: 変更なし — ファイルはそのまま維持されます"
                    )}
                    yield {"file_written": planned_file.path}
                    continue

                # 部分失敗ガード: 失敗率 > 50% の場合はバックアップから復元してスキップ
                total_blocks = applied + len(failed)
                if total_blocks > 0 and len(failed) / total_blocks > 0.5:
                    yield {"warning": (
                        f"⚠ {planned_file.path}: {len(failed)}/{total_blocks}件のブロックが失敗 "
                        f"— バックアップから復元してスキップします"
                    )}
                    restore_backup(file_path)
                    continue

                if failed:
                    yield {"status": (
                        f"⚠ {planned_file.path}: {len(failed)}件のブロックが最終的に一致しませんでした"
                    )}

                try:
                    yield {"status": f"💾 {planned_file.path} に変更を保存中..."}
                    self._fs.write_text(file_path, modified_content)
                except Exception as exc:
                    logger.error("ファイル書き込みエラー [%s]: %s", planned_file.path, exc)
                    yield {"error": str(exc)}
                    continue

                # 構文検証
                ok, err_msg = validate(file_path, modified_content)
                if not ok:
                    yield {"warning": f"⚠ {planned_file.path}: 構文エラー — {err_msg} — バックアップから復元"}
                    restore_backup(file_path)
                    continue
                delete_backup(file_path)

                line_count = modified_content.count("\n") + 1
                try:
                    self._git.commit_all(
                        root,
                        f"LocalForge [modify] {planned_file.path} — {applied}件のブロック適用, {line_count}行, 検証済み ✓",
                    )
                except Exception as exc:
                    logger.warning("git commit失敗 [%s]: %s", planned_file.path, exc)

            else:
                # ── CREATE路: ファイル全体をゼロから生成（従来通り） ──
                operation = "generate_file"
                prompt, tokens = self._context.build_file_generation_prompt(
                    target_file=planned_file.path,
                    target_description=planned_file.description,
                    context_md=context_md,
                    plan_json=plan_json,
                    dependency_contents=dependency_contents,
                    language=language,
                )
                log_entry = GenerationLogEntry(
                    mode="generate", model=model, operation=operation,
                    file_path=planned_file.path, status="pending",
                    prompt_tokens_estimated=tokens,
                )
                self._index_adapter.append_log_entry(log_path, log_entry)

                file_content_parts: List[str] = []
                try:
                    for token in self._llm.stream_completion(model, prompt):
                        if is_cancelled():
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
                self._update_log_status(log_path, planned_file.path, elapsed)

                try:
                    yield {"status": f"💾 {planned_file.path} を新規作成中..."}
                    self._fs.write_text(file_path, file_content)
                except Exception as exc:
                    logger.error("ファイル書き込みエラー [%s]: %s", planned_file.path, exc)
                    yield {"error": str(exc)}
                    continue

                # 構文検証
                ok, err_msg = validate(file_path, file_content)
                if not ok:
                    yield {"warning": f"⚠ {planned_file.path}: 構文エラー — {err_msg}"}
                    # 新規作成の場合はファイルを削除してスキップ
                    try:
                        file_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    continue

                line_count = file_content.count("\n") + 1
                try:
                    self._git.commit_all(
                        root,
                        f"LocalForge [create] {planned_file.path} — {line_count}行, 検証済み ✓",
                    )
                except Exception as exc:
                    logger.warning("git commit失敗 [%s]: %s", planned_file.path, exc)

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
        language: str = "en",
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

        full_path = root / planned_file.path
        file_exists_on_disk = full_path.exists()
        is_modify = planned_file.action == "modify" or (
            planned_file.action == "create" and file_exists_on_disk
        )

        if is_modify:
            self._ensure_bak_ignored(root)
            existing_content = self._create_backup(full_path)
            modification_notes = planned_file.modification_notes or planned_file.description
            context_md_val = context_md

            max_chars = self._context.max_chunk_chars()
            chunks = self._split_into_chunks(existing_content, max_chars)
            total_chunks = len(chunks)
            all_diff_parts: List[str] = []

            for chunk_idx, chunk_content in enumerate(chunks):
                if is_cancelled():
                    yield {"error": "キャンセルされました"}
                    return

                if total_chunks > 1:
                    yield {"status": (
                        f"{planned_file.path}: チャンク {chunk_idx + 1}/{total_chunks} を分析中..."
                    )}

                prompt = self._context.build_file_diff_prompt(
                    target_file=planned_file.path,
                    modification_notes=modification_notes,
                    chunk_content=chunk_content,
                    context_md=context_md_val,
                    chunk_idx=chunk_idx,
                    total_chunks=total_chunks,
                )

                chunk_parts: List[str] = []
                try:
                    for token in self._llm.stream_completion(model, prompt):
                        if is_cancelled():
                            yield {"error": "キャンセルされました"}
                            return
                        chunk_parts.append(token)
                        yield {"token": token}
                except Exception as exc:
                    yield {"error": str(exc)}
                    return
                all_diff_parts.append("".join(chunk_parts))

            combined_diff = "\n".join(all_diff_parts)
            modified_content, applied, failed = self._apply_search_replace_blocks(
                existing_content, combined_diff
            )

            if applied == 0:
                yield {"status": f"ℹ {planned_file.path}: 変更なし"}
                yield {"file_written": planned_file.path}
                yield {"done": True}
                return

            # 部分失敗ガード
            total_blocks = applied + len(failed)
            if total_blocks > 0 and len(failed) / total_blocks > 0.5:
                yield {"warning": (
                    f"⚠ {planned_file.path}: {len(failed)}/{total_blocks}件のブロックが失敗 "
                    f"— バックアップから復元"
                )}
                restore_backup(full_path)
                yield {"done": True}
                return

            if failed:
                yield {"status": f"⚠ {len(failed)}件のブロックが一致しませんでした"}

            try:
                self._fs.write_text(full_path, modified_content)
            except Exception as exc:
                yield {"error": str(exc)}
                return

            # 構文検証
            ok, err_msg = validate(full_path, modified_content)
            if not ok:
                yield {"warning": f"⚠ {planned_file.path}: 構文エラー — {err_msg} — バックアップから復元"}
                restore_backup(full_path)
                yield {"done": True}
                return
            delete_backup(full_path)

            line_count = modified_content.count("\n") + 1
            try:
                self._git.commit_all(
                    root,
                    f"LocalForge [modify] {planned_file.path} — {applied}件のブロック適用, {line_count}行, 検証済み ✓",
                )
            except Exception as exc:
                logger.warning("git commit失敗 [%s]: %s", planned_file.path, exc)

        else:
            prompt = self._context.build_file_generation_prompt(
                target_file=planned_file.path,
                target_description=planned_file.description,
                context_md=context_md,
                plan_json=plan.model_dump_json(indent=2),
                dependency_contents=dependency_contents,
                language=language,
            )
            file_content_parts: List[str] = []
            try:
                for token in self._llm.stream_completion(model, prompt):
                    if is_cancelled():
                        yield {"error": "キャンセルされました"}
                        return
                    file_content_parts.append(token)
                    yield {"token": token}
            except Exception as exc:
                yield {"error": str(exc)}
                return

            file_content = "".join(file_content_parts)
            try:
                self._fs.write_text(full_path, file_content)
            except Exception as exc:
                yield {"error": str(exc)}
                return

            ok, err_msg = validate(full_path, file_content)
            if not ok:
                yield {"warning": f"⚠ {planned_file.path}: 構文エラー — {err_msg}"}
                try:
                    full_path.unlink(missing_ok=True)
                except Exception:
                    pass
                yield {"done": True}
                return

            line_count = file_content.count("\n") + 1
            try:
                self._git.commit_all(
                    root,
                    f"LocalForge [create] {planned_file.path} — {line_count}行, 検証済み ✓",
                )
            except Exception as exc:
                logger.warning("git commit失敗 [%s]: %s", planned_file.path, exc)

        yield {"file_written": planned_file.path}
        yield {"done": True}

    def _split_into_chunks(self, content: str, max_chars: int) -> List[str]:
        """
        コードをチャンクに分割する。
        空行・トップレベルのdef/class/function宣言を優先的な分割点として使い、
        論理的なまとまりが壊れにくいようにする。

        Args:
            content: 分割するコード全文
            max_chars: 1チャンクの最大文字数

        Returns:
            チャンク文字列のリスト
        """
        if len(content) <= max_chars:
            return [content]

        lines = content.split("\n")
        chunks: List[str] = []
        current: List[str] = []
        current_len = 0

        # トップレベル境界パターン（インデントなし）
        _boundary = re.compile(
            r"^(def |class |async def |function |export |module\.exports)"
        )

        for line in lines:
            line_len = len(line) + 1  # +1 for newline
            if current_len + line_len > max_chars and current:
                # 好ましい分割点: 直前の空行またはトップレベル宣言
                split_at = len(current)
                for i in range(len(current) - 1, max(0, len(current) - 30), -1):
                    if current[i].strip() == "" or _boundary.match(current[i]):
                        split_at = i
                        break
                chunks.append("\n".join(current[:split_at]))
                current = current[split_at:]
                current_len = sum(len(l) + 1 for l in current)

            current.append(line)
            current_len += line_len

        if current:
            chunks.append("\n".join(current))

        return [c for c in chunks if c.strip()]

    def _apply_search_replace_blocks(
        self,
        original: str,
        llm_output: str,
    ) -> Tuple[str, int, List[str]]:
        """
        LLM出力からSEARCH/REPLACEブロックを抽出してoriginalに適用する。
        <<<<<<< SEARCH / ======= / >>>>>>> REPLACE 形式（記号数は3以上で許容）。

        Returns:
            (modified_content, applied_count, list_of_unmatched_search_snippets)
        """
        pattern = re.compile(
            r"<{3,}[ \t]*SEARCH[ \t]*\r?\n(.*?)\r?\n={5,}[ \t]*\r?\n(.*?)\r?\n>{3,}[ \t]*REPLACE",
            re.DOTALL,
        )

        result = original
        applied = 0
        failed: List[str] = []

        for match in pattern.finditer(llm_output):
            search_text = match.group(1)
            replace_text = match.group(2)

            # 1) 完全一致
            if search_text in result:
                result = result.replace(search_text, replace_text, 1)
                applied += 1
                continue

            # 2) 改行コード正規化
            norm_result = result.replace("\r\n", "\n").replace("\r", "\n")
            norm_search = search_text.replace("\r\n", "\n").replace("\r", "\n")
            if norm_search in norm_result:
                result = norm_result.replace(norm_search, replace_text, 1)
                applied += 1
                continue

            # 3) 行末空白を除去して再試行（LLMが行末スペースを省略する場合）
            def rstrip_lines(s: str) -> str:
                return "\n".join(l.rstrip() for l in s.replace("\r\n", "\n").split("\n"))

            stripped_result = rstrip_lines(norm_result)
            stripped_search = rstrip_lines(norm_search)

            if stripped_search in stripped_result:
                pos = stripped_result.find(stripped_search)
                line_start = stripped_result[:pos].count("\n")
                search_line_count = stripped_search.count("\n") + 1
                result_lines = norm_result.split("\n")
                replace_lines = replace_text.replace("\r\n", "\n").split("\n")
                result_lines[line_start : line_start + search_line_count] = replace_lines
                result = "\n".join(result_lines)
                applied += 1
                continue

            failed.append(search_text[:120])
            logger.warning(
                "SEARCH/REPLACE: 一致するコードが見つかりません: %s...", search_text[:80]
            )

        return result, applied, failed

    def _ensure_bak_ignored(self, root: Path) -> None:
        """
        .gitignore に *.bak エントリがなければ追記して、
        バックアップファイルが誤ってgit管理されないようにする。
        """
        gitignore = root / ".gitignore"
        try:
            existing = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""
            if "*.bak" not in existing:
                with gitignore.open("a", encoding="utf-8") as fh:
                    fh.write("\n# LocalForge バックアップファイル\n*.bak\n")
        except Exception as exc:
            logger.warning(".gitignore への *.bak 追記失敗: %s", exc)

    def _create_backup(self, file_path: Path) -> str:
        """
        ファイルを編集前に .bak ファイルとしてバックアップし、
        既存コンテンツを返す。ファイルが存在しない場合は空文字列を返す。

        Args:
            file_path: バックアップするファイルのフルパス

        Returns:
            既存ファイルの内容（存在しない場合は空文字列）
        """
        if not file_path.exists():
            return ""
        try:
            existing = self._fs.read_text(file_path)
            backup_path = file_path.with_suffix(file_path.suffix + ".bak")
            self._fs.write_text(backup_path, existing)
            logger.info("バックアップ作成: %s", backup_path)
            return existing
        except Exception as exc:
            logger.warning("バックアップ作成失敗 [%s]: %s", file_path, exc)
            return ""

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
