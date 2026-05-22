"""
説明サービス — レポート生成・Q&Aオーケストレーターの責務を担う。
11セクションのレポートをSSEでストリーミング生成する。
"""

from __future__ import annotations

import json
import logging
import re as _re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Generator, List, Optional

from localforge.application.analysis_service import AnalysisService
from localforge.application.context_service import ContextService
from localforge.application.generation_service import is_cancelled, reset_cancel
from localforge.domain.models import LOCALFORGE_DIR as _LOCALFORGE_DIR, FileChunk, GenerationLogEntry, Message, ProjectIndex
from localforge.infrastructure.disk_cache import DiskCache
from localforge.infrastructure.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

# レポートの11セクション定義
REPORT_SECTIONS = [
    "Project Overview",
    "Module Map",
    "Entry Points & Startup Flow",
    "Data Flow",
    "Key Interfaces & Contracts",
    "External Dependencies",
    "Configuration",
    "Test Coverage",
    "Notable Patterns & Design Decisions",
    "Potential Issues & Technical Debt",
    "Project Health & Code Quality Analysis",
    "How to Extend This Project",
]

_HEADING_RE = _re.compile(r"^#{1,3}\s+", _re.MULTILINE)


def _strip_leading_heading(content: str, section_name: str) -> str:
    """
    LLM がセクションタイトルを先頭に出力した場合に除去する。
    section_name に一致する最初の見出し行、または単なる先頭見出しを削除する。
    """
    text = content.strip()
    lines = text.splitlines(keepends=True)
    if not lines:
        return text
    first = lines[0].strip()
    # 先頭行が # 見出しであり、かつセクション名を含む場合に除去
    if _HEADING_RE.match(first):
        stripped_title = _HEADING_RE.sub("", first).strip()
        if stripped_title.lower() in section_name.lower() or section_name.lower() in stripped_title.lower():
            return "".join(lines[1:]).strip()
    return text


class ExplanationService:
    """
    コードベースの説明レポート生成とQ&A応答を担当するサービスクラス。
    各セクションを個別のOllamaコールでストリーミング生成する。
    """

    def __init__(
        self,
        analysis: AnalysisService,
        llm: OllamaClient,
        context: ContextService,
    ) -> None:
        """
        ExplanationServiceを初期化する。

        Args:
            analysis: 分析サービス
            llm: OllamaクライアントLLMバックエンド
            context: コンテキストサービス
        """
        self._analysis = analysis
        self._llm = llm
        self._context = context

        # ── Cache 1: index_json string ──
        # key: (root_str, mtime_float) → serialized JSON string
        # Avoids model_dump() + json.dumps() on every Q&A / report call.
        self._index_json_cache: dict[tuple, str] = {}

        # ── Cache 2: file content ──
        # key: (path_str, mtime_ns, size_bytes, max_chars) → content string
        # Avoids re-reading unchanged files across Q&A calls.
        # Bounded at 300 entries (LRU-ish: dict insertion order).
        self._file_content_cache: dict[tuple, str] = {}

        # ── Cache 4: Q&A response ──
        # Stores full answer strings keyed by a hash of (root, question,
        # pinned_paths, last-3 history turns, index_mtime).
        # Hot layer: in-memory dict; cold layer: .localforge/cache/responses/
        self._response_cache: dict[str, DiskCache] = {}  # root_str → DiskCache
        self._FILE_CACHE_MAX = 300

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _get_index_json(self, root: Path, project_index: "ProjectIndex", include_files: bool = False) -> str:
        """Return serialized project index JSON, using cache to skip model_dump()+json.dumps()."""
        pi_path = root / _LOCALFORGE_DIR / "project_index.json"
        try:
            mtime = pi_path.stat().st_mtime
        except OSError:
            mtime = 0.0
        cache_key = (str(root), mtime, include_files)
        cached = self._index_json_cache.get(cache_key)
        if cached is not None:
            return cached
        index_dict = project_index.model_dump(
            include={"project_name", "summary", "total_files", "indexed_files"}
        )
        if include_files:
            index_dict["files"] = [c.path for c in project_index.file_chunks]
        result = json.dumps(index_dict, ensure_ascii=False)
        # Evict stale entries for the same root (mtime changed)
        self._index_json_cache = {k: v for k, v in self._index_json_cache.items() if k[0] != str(root)}
        self._index_json_cache[cache_key] = result
        return result

    def _read_file_cached(self, path: Path, max_chars: int = -1) -> Optional[str]:
        """Read file content with mtime+size cache. Returns None on read error.
        max_chars=-1 means no truncation (full file)."""
        try:
            stat = path.stat()
            key = (str(path), stat.st_mtime_ns, stat.st_size, max_chars)
        except OSError:
            return None
        cached = self._file_content_cache.get(key)
        if cached is not None:
            return cached
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            if max_chars >= 0 and len(content) > max_chars:
                content = content[:max_chars]
            # LRU eviction: remove oldest entry when over limit
            if len(self._file_content_cache) >= self._FILE_CACHE_MAX:
                self._file_content_cache.pop(next(iter(self._file_content_cache)))
            self._file_content_cache[key] = content
            return content
        except OSError:
            return None

    def _get_response_disk_cache(self, root: Path) -> "DiskCache":
        key = str(root)
        if key not in self._response_cache:
            self._response_cache[key] = DiskCache(
                root / _LOCALFORGE_DIR / "cache" / "responses", max_memory=50
            )
        return self._response_cache[key]

    def _make_response_cache_key(
        self,
        root: Path,
        question: str,
        pinned_paths: Optional[list],
        history: list,
        index_mtime: float,
    ) -> str:
        import hashlib as _hl
        h = _hl.sha256()
        h.update(str(root).encode())
        h.update(question.encode())
        h.update(json.dumps(sorted(pinned_paths or [])).encode())
        # Include last 3 history turns so context changes invalidate cache
        tail = [(m.role, m.content) for m in (history or [])[-3:]]
        h.update(json.dumps(tail).encode())
        h.update(str(round(index_mtime, 3)).encode())
        return h.hexdigest()[:32]

    def invalidate_response_cache(self, root: Path) -> None:
        """Clear response cache for a project (call after build_index)."""
        dc = self._get_response_disk_cache(root)
        dc.clear()
        self._response_cache.pop(str(root), None)

    def _log_async(self, log_path: Path, log_entry: "GenerationLogEntry") -> None:
        """Append a log entry in a background thread — does not block streaming."""
        import threading
        threading.Thread(
            target=self._analysis._index_adapter.append_log_entry,
            args=(log_path, log_entry),
            daemon=True,
        ).start()

    def stream_report(
        self,
        root: Path,
        model: str,
        selected_section_indices: Optional[List[int]] = None,
        resume_from: int = 0,
        language: str = "ja",
    ) -> Generator[dict, None, None]:
        """
        レポートをSSEイベントとしてストリーミング生成する。
        各セクションは別々のOllamaコールで処理し、完了ごとにdiskへ保存する。

        Args:
            root: プロジェクトルート
            model: 使用するOllamaモデル名
            selected_section_indices: 生成するセクションのインデックスリスト（Noneで全セクション）
            resume_from: このインデックス以降のセクションを生成（それ以前は既存内容を再利用）

        Yields:
            SSEペイロード辞書（section, token, progress, done, error）
        """
        getattr(self._llm, "preload_model_async", lambda m: None)(model)

        project_index = self._analysis.load_project_index(root)
        if not project_index:
            yield {"error": "ProjectIndexが見つかりません。先にインデックスを構築してください。"}
            return

        chunks = project_index.file_chunks
        index_json = self._get_index_json(root, project_index, include_files=False)

        # セクション選択フィルタリング
        if selected_section_indices is not None:
            active_indices = set(selected_section_indices)
            sections_to_run = [
                (i, name) for i, name in enumerate(REPORT_SECTIONS)
                if i in active_indices
            ]
        else:
            sections_to_run = list(enumerate(REPORT_SECTIONS))

        total_sections = len(REPORT_SECTIONS)
        completed_sections: List[tuple[str, str]] = []

        # resume_from: 既存レポートから既存セクション内容を取得して再利用する
        if resume_from > 0:
            existing = self._load_existing_sections(root)
            for i, name in enumerate(REPORT_SECTIONS):
                if i < resume_from:
                    existing_content = existing.get(name, "")
                    completed_sections.append((name, existing_content))
                    # スキップ済みセクションをフロントエンドに通知
                    yield {
                        "section": name,
                        "section_idx": i + 1,
                        "section_total": total_sections,
                        "skipped": True,
                    }
                    yield {"token": existing_content}
                    yield {
                        "progress": {
                            "done": i + 1,
                            "total": total_sections,
                            "current_file": name,
                        }
                    }

        # 生成対象セクションのみsemantic searchを並列プリフェッチ
        sections_needing_gen = [
            (i, name) for i, name in sections_to_run if i >= resume_from
        ]

        def _fetch_section_chunks(args: tuple) -> list:
            _, section_name = args
            return self._analysis.get_top_chunks_semantic(chunks, section_name, top_n=6)

        if sections_needing_gen:
            with ThreadPoolExecutor(max_workers=min(len(sections_needing_gen), 4)) as ex:
                fetched_chunks = list(ex.map(_fetch_section_chunks, sections_needing_gen))
            chunk_map = {name: fetched_chunks[j] for j, (_, name) in enumerate(sections_needing_gen)}
        else:
            chunk_map = {}

        for sec_idx, section_name in sections_to_run:
            if sec_idx < resume_from:
                continue  # 既に処理済み

            yield {
                "section": section_name,
                "section_idx": sec_idx + 1,
                "section_total": total_sections,
            }

            relevant_chunks = chunk_map.get(section_name, [])
            # サマリーを先頭3行・最大400文字に拡張してコンテキストを豊かにする
            relevant_summaries = [
                (c.path, "\n".join((c.summary or "").splitlines()[:3])[:400])
                for c in relevant_chunks if c.summary
            ]

            prompt, tokens = self._context.build_report_section_prompt(
                section_name=section_name,
                project_index_json=index_json,
                relevant_summaries=relevant_summaries,
                language=language,
            )

            start_time = time.time()
            section_tokens: List[str] = []
            try:
                for token in self._llm.stream_completion(model, prompt):
                    section_tokens.append(token)
                    yield {"token": token}
            except Exception as exc:
                logger.error("セクション生成エラー [%s]: %s", section_name, exc)
                yield {"token": f"\n[エラー: {exc}]\n"}

            elapsed = (time.time() - start_time) * 1000
            log_entry = GenerationLogEntry(
                mode="explain",
                model=model,
                operation=f"report:{section_name}",
                prompt_tokens_estimated=tokens,
                response_time_ms=elapsed,
                status="completed",
            )
            log_path = root / _LOCALFORGE_DIR / "generation_log.jsonl"
            self._log_async(log_path, log_entry)

            completed_sections.append((section_name, "".join(section_tokens)))

            # セクション完了ごとに差分保存（中断しても失わない）
            is_partial = len(completed_sections) < total_sections
            self._save_report(
                root, completed_sections, project_index.project_name,
                partial=is_partial, total=total_sections,
            )

            yield {
                "progress": {
                    "done": sec_idx + 1,
                    "total": total_sections,
                    "current_file": section_name,
                }
            }

        # 全セクション対象かつ全て完了した場合のみ履歴に保存する
        generated_all = (
            selected_section_indices is None
            and resume_from == 0
            and len(completed_sections) == total_sections
        )
        if generated_all and completed_sections:
            self._save_to_history(root, completed_sections, project_index.project_name, model)

        yield {
            "progress": {
                "done": total_sections,
                "total": total_sections,
                "current_file": "完了",
            }
        }
        yield {"done": True}

    def _save_report(
        self,
        root: Path,
        sections: List[tuple[str, str]],
        project_name: str,
        partial: bool = False,
        total: Optional[int] = None,
    ) -> None:
        """
        レポートをMarkdown形式で .localforge/report.md に保存する。
        partial=True の場合は機械可読なコメントマーカーを埋め込む。
        """
        report_path = root / _LOCALFORGE_DIR / "report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)

        total_count = total or len(sections)
        lines: List[str] = [f"# {project_name} — Codebase Report\n\n"]
        if partial:
            lines.append(f"<!-- localforge:partial:{len(sections)}/{total_count} -->\n\n")

        for name, content in sections:
            lines.append(f"## {name}\n\n")
            lines.append(_strip_leading_heading(content, name))
            lines.append("\n\n---\n\n")

        report_path.write_text("".join(lines), encoding="utf-8")
        logger.info("レポートを保存しました (%s): %s", "部分" if partial else "完了", report_path)

    def _save_to_history(
        self,
        root: Path,
        sections: List[tuple[str, str]],
        project_name: str,
        model: str,
    ) -> None:
        """完成したレポートを .localforge/reports/ に履歴として保存する。"""
        reports_dir = root / _LOCALFORGE_DIR / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        report_file = reports_dir / f"{ts}.md"

        lines: List[str] = [f"# {project_name} — Codebase Report\n\n"]
        for name, content in sections:
            lines.append(f"## {name}\n\n")
            lines.append(_strip_leading_heading(content, name))
            lines.append("\n\n---\n\n")
        report_file.write_text("".join(lines), encoding="utf-8")

        history_path = reports_dir / "history.json"
        history: list = []
        if history_path.exists():
            try:
                history = json.loads(history_path.read_text(encoding="utf-8"))
            except Exception:
                history = []

        history.append({
            "id": ts,
            "filename": report_file.name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "partial": False,
            "sections_done": len(sections),
            "sections_total": len(REPORT_SECTIONS),
            "model": model,
        })
        history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("レポート履歴に保存しました: %s", report_file)

    def _load_existing_sections(self, root: Path) -> dict:
        """
        既存の report.md からセクション名→内容の辞書を読み込む。
        resume_from 時に使用する。
        """
        report_path = root / _LOCALFORGE_DIR / "report.md"
        if not report_path.exists():
            return {}
        try:
            content = report_path.read_text(encoding="utf-8")
        except OSError:
            return {}

        result: dict = {}
        # ## Section Name\n\n ... \n\n---\n\n のパターンで分割
        pattern = _re.compile(r"^## (.+?)$", _re.MULTILINE)
        matches = list(pattern.finditer(content))
        for i, m in enumerate(matches):
            name = m.group(1).strip()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            section_body = content[start:end].strip().rstrip("---").strip()
            result[name] = section_body
        return result

    def get_report_history(self, root: Path) -> list:
        """レポート履歴メタデータのリストを返す（新しい順）。"""
        history_path = root / _LOCALFORGE_DIR / "reports" / "history.json"
        if not history_path.exists():
            return []
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
            return list(reversed(history))
        except Exception:
            return []

    def get_historical_report(self, root: Path, report_id: str) -> Optional[str]:
        """指定IDの履歴レポート内容を返す。存在しない場合はNone。"""
        reports_dir = root / _LOCALFORGE_DIR / "reports"
        # IDはタイムスタンプ文字列 = ファイル名のstem
        for candidate in reports_dir.glob("*.md"):
            if candidate.stem == report_id:
                try:
                    return candidate.read_text(encoding="utf-8")
                except OSError:
                    return None
        return None

    def delete_historical_report(self, root: Path, report_id: str) -> bool:
        """指定IDの履歴レポートを削除してhistory.jsonを更新する。成功したらTrue。"""
        reports_dir = root / _LOCALFORGE_DIR / "reports"
        report_file: Optional[Path] = None
        for candidate in reports_dir.glob("*.md"):
            if candidate.stem == report_id:
                report_file = candidate
                break
        if not report_file or not report_file.exists():
            return False

        report_file.unlink()

        history_path = reports_dir / "history.json"
        if history_path.exists():
            try:
                history = json.loads(history_path.read_text(encoding="utf-8"))
                history = [h for h in history if h.get("id") != report_id]
                history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass
        return True

    def _stream_answer_ultra(
        self,
        root: Path,
        model: str,
        question: str,
        history: List[Message],
        project_index: "ProjectIndex",
        pinned_paths: Optional[List[str]] = None,
    ) -> Generator[dict, None, None]:
        """超高速Q&Aモード: ゼロディスクリード、サマリーのみ、レポートセクションと同等速度。"""
        import threading as _threading
        chunks = project_index.file_chunks

        _preload_event: _threading.Event = getattr(
            self._llm, "preload_model_async", lambda m: _threading.Event()
        )(model)

        yield {"phase": "超高速モード", "detail": "サマリーのみでコンテキストを構築中"}

        # Pinned: use summary field only, no disk reads
        pinned_summaries: List[tuple[str, str]] = []
        if pinned_paths:
            chunk_map = {c.path.replace("\\", "/"): c for c in chunks}
            for p in pinned_paths:
                norm = p.replace("\\", "/").rstrip("/")
                c = chunk_map.get(norm)
                if c and c.summary:
                    pinned_summaries.append((c.path, c.summary))
                else:
                    # folder: collect child summaries
                    prefix = norm + "/"
                    for c2 in chunks:
                        cp = c2.path.replace("\\", "/")
                        if (cp == norm or cp.startswith(prefix)) and c2.summary:
                            pinned_summaries.append((c2.path, c2.summary))

        # RAG top 3 summaries
        top_chunks = self._analysis.get_top_chunks_semantic(chunks, question, top_n=3)
        pinned_set = {p for p, _ in pinned_summaries}
        rag_summaries = [(c.path, c.summary or "") for c in top_chunks if c.path not in pinned_set and c.summary]

        index_json = self._get_index_json(root, project_index, include_files=False)

        prompt, tokens = self._context.build_qa_prompt(
            question=question,
            project_index_json=index_json,
            top_summaries=rag_summaries,
            full_contents=[],
            conversation_history=history[-2:],
            pinned_contents=[(p, s) for p, s in pinned_summaries] if pinned_summaries else None,
            mode="ultra",
        )

        yield {"prompt_preview": prompt[:500], "prompt_tokens": tokens}

        # Wait for the preload to finish before sending the prompt.
        # This eliminates the queue gap where Ollama would otherwise receive
        # the real prompt while still processing the empty preload request.
        if not _preload_event.is_set():
            yield {"phase": "Ollamaモデルロード中", "detail": f"{model} をRAMに読み込み中（コンテキスト構築と並行）"}
            while not _preload_event.wait(timeout=1.0):
                if is_cancelled():
                    yield {"done": True}
                    return
                yield {"status": f"Ollamaがモデルをロード中... ({model})"}

        yield {"status": f"Ollamaが回答を生成中... (推定 {tokens} トークン)"}

        answer_tokens: List[str] = []
        try:
            for token in self._llm.stream_completion(model, prompt, keep_alive="2h"):
                if is_cancelled():
                    break
                answer_tokens.append(token)
                yield {"token": token}
        except Exception as exc:
            logger.error("超高速Q&A生成エラー: %s", exc)
            yield {"error": str(exc)}
            return

        if answer_tokens:
            answer = "".join(answer_tokens)
            _rc = self._get_response_disk_cache(root)
            pi_path = root / _LOCALFORGE_DIR / "project_index.json"
            try:
                _mtime = pi_path.stat().st_mtime
            except OSError:
                _mtime = 0.0
            _key = self._make_response_cache_key(root, question, pinned_paths, history, _mtime)
            _rc.set(_key, answer)

        yield {"done": True}

    def stream_answer(
        self,
        root: Path,
        model: str,
        question: str,
        history: List[Message],
        workspace_roots: Optional[List[tuple[Path, str]]] = None,
        pinned_paths: Optional[List[str]] = None,
        mode: str = "precise",
    ) -> Generator[dict, None, None]:
        """
        Q&A質問への回答をSSEイベントとしてストリーミング生成する。

        パイプライン:
          1. ピン留めファイルがある場合は直接展開 (Phase 1 スキップ)
          2. ピン留めなしの場合のみ LLM ファイル選択 (Phase 1)
          3. セマンティック検索で補完
          4. 依存関係展開 (BFS 5 hop, カスタムインポートのみ)
          5. プロンプト構築 & Ollama ストリーミング

        Yields:
            SSEペイロード辞書（phase, prompt_preview, token, done, error）
        """
        reset_cancel()

        project_index = self._analysis.load_project_index(root)
        if not project_index:
            yield {"error": "ProjectIndexが見つかりません。先にインデックスを構築してください。"}
            return

        # ── Ultra mode: summaries-only path (same structure as a report section) ──
        if mode == "ultra":
            yield from self._stream_answer_ultra(
                root=root, model=model, question=question,
                history=history, project_index=project_index,
                pinned_paths=pinned_paths,
            )
            return

        # Fire model warm-up immediately — phases A-G run in parallel with model loading.
        # Capture the Event so we can wait for it before sending the real prompt,
        # eliminating the Ollama queue gap.
        import threading as _threading
        _model_was_cold = getattr(self._llm, "is_model_loaded", lambda m: None)(model) is False
        if _model_was_cold:
            yield {"status": f"モデルをバックグラウンドでプリロード中... ({model})"}
        _preload_event: _threading.Event = getattr(
            self._llm, "preload_model_async", lambda m: _threading.Event()
        )(model)

        chunks = project_index.file_chunks
        chunk_map = {c.path: c for c in chunks}
        all_summaries = [(c.path, c.summary or "") for c in chunks if c.summary]

        index_json = self._get_index_json(root, project_index, include_files=True)

        # ── Response cache check (exact question + pinned_paths + history) ──
        pi_path = root / _LOCALFORGE_DIR / "project_index.json"
        try:
            _index_mtime = pi_path.stat().st_mtime
        except OSError:
            _index_mtime = 0.0
        _rc_key = self._make_response_cache_key(root, question, pinned_paths, history, _index_mtime)
        _rc = self._get_response_disk_cache(root)
        _cached_answer = _rc.get(_rc_key)
        if _cached_answer is not None:
            yield {"phase": "キャッシュヒット", "detail": "過去の同一質問への回答を再利用します"}
            yield {"status": "キャッシュから回答を返しています..."}
            # Stream cached answer in chunks so the UI behaves identically to live generation
            _CHUNK = 20
            for _i in range(0, len(_cached_answer), _CHUNK):
                yield {"token": _cached_answer[_i:_i + _CHUNK]}
            yield {"phase": "完了", "detail": "キャッシュから回答済み"}
            yield {"done": True}
            return

        # ── Phase A: ピン留めコンテキスト解決 ──
        # ピン留めファイルはユーザーが明示的に選んだもの。全内容を読み込む（文字数上限なし）。
        # トークン予算は build_qa_prompt 内で管理する。
        # mode == "fast" では BFS を省略し直接選択ファイルのみを軽量読み込みする。
        pinned_chunk_contents: List[tuple[str, str]] = []
        pinned_base_chunks: List = []
        _pinned_depth_map: dict = {}
        _direct_pinned_set: set = set()
        if pinned_paths:
            yield {"phase": "ピン留めファイル解決中", "detail": f"{len(pinned_paths)} 件のパスを展開中"}
            yield {"status": f"ピン留めファイルを展開中... ({len(pinned_paths)} 件)"}
            try:
                if mode == "fast":
                    # Fast mode: direct files only, no BFS, 800-char cap
                    pinned_chunks, _pdep, _pinned_depth_map = self._analysis.resolve_pinned_chunks(
                        root, pinned_paths, chunks, max_total=30, max_depth=1
                    )
                else:
                    pinned_chunks, _pdep, _pinned_depth_map = self._analysis.resolve_pinned_chunks(
                        root, pinned_paths, chunks, max_total=200
                    )
                _direct_pinned_set = {p for p, d in _pinned_depth_map.items() if d == 0}
                pinned_base_chunks = pinned_chunks
                yield {"phase": "ピン留めファイル解決中", "detail": f"依存関係を含む {len(pinned_chunks)} ファイルを取得 (直接: {len(_direct_pinned_set)})"}

                def _read_pinned(pc):
                    if mode == "fast":
                        max_chars = 800
                    else:
                        depth = _pinned_depth_map.get(pc.path, 0)
                        if depth <= 1:
                            max_chars = -1  # full file for directly pinned and direct imports
                        elif depth <= 4:
                            max_chars = 6000
                        else:
                            max_chars = 1500
                    content = self._read_file_cached(root / pc.path, max_chars=max_chars)
                    if content is None:
                        logger.warning("ピン留めファイル読み込みエラー: %s", pc.path)
                    return (pc.path, content) if content is not None else None

                if pinned_chunks:
                    with ThreadPoolExecutor(max_workers=min(len(pinned_chunks), 8)) as ex:
                        pinned_chunk_contents = [
                            r for r in ex.map(_read_pinned, pinned_chunks) if r is not None
                        ]
            except Exception as exc:
                logger.warning("ピン留めコンテキスト解決エラー: %s", exc)

        # ── Phase B: ワークスペース他プロジェクトのコンテキスト ──
        # mode == "fast" ではワークスペース展開をスキップする
        workspace_project_data: List[dict] = []
        if workspace_roots and not mode == "fast":
            yield {"phase": "ワークスペース検索中", "detail": f"{len(workspace_roots)} プロジェクト"}
            yield {"status": "ワークスペースプロジェクトを検索中..."}
            _max_ws_chars = self._context.max_qa_file_chars() // 2

            def _load_workspace(ws_args):
                ws_root, ws_name = ws_args
                try:
                    ws_idx = self._analysis.load_project_index(ws_root)
                    if not ws_idx:
                        return None
                    ws_chunks = ws_idx.file_chunks
                    ws_top = self._analysis.get_top_chunks_semantic(ws_chunks, question, top_n=5)
                    ws_exp_chunks, _, _ws_depth = self._analysis.expand_with_dependencies(
                        ws_chunks, [c.path for c in ws_top], max_total=8
                    )
                    ws_summaries = [(c.path, c.summary or "") for c in ws_exp_chunks if c.summary]
                    ws_contents: List[tuple[str, str]] = []
                    for wc in ws_exp_chunks[:3]:
                        fp = ws_root / wc.path
                        if fp.exists():
                            try:
                                content = fp.read_text(encoding="utf-8", errors="replace")
                                ws_contents.append((wc.path, content[:_max_ws_chars]))
                            except OSError:
                                pass
                    return {
                        "name": ws_name,
                        "root": str(ws_root),
                        "summaries": ws_summaries,
                        "contents": ws_contents,
                    }
                except Exception as exc:
                    logger.warning("ワークスペース [%s] Q&Aエラー: %s", ws_name, exc)
                    return None

            with ThreadPoolExecutor(max_workers=min(len(workspace_roots[:3]), 3)) as ex:
                workspace_project_data = [
                    r for r in ex.map(_load_workspace, workspace_roots[:3]) if r is not None
                ]

        # ── Phase C: ファイル選択 ──
        # セマンティック検索（ChromaDB / BM25）に一本化。
        # 以前の GPU 専用 LLM ファイル選択 (generate_sync) は最大 5 分のブロッキングを引き起こすため削除。
        # ベクトル検索 + 依存関係展開 (Phase E) で同等のカバレッジを達成する。
        selected_paths: List[str] = []
        if pinned_base_chunks:
            yield {"phase": "ファイル選択", "detail": f"ピン留め {len(pinned_base_chunks)} ファイルを使用"}
            yield {"status": f"ピン留めファイルを使用中 ({len(pinned_base_chunks)} 件)"}
        else:
            yield {"phase": "ファイル選択", "detail": "セマンティック検索でファイルを選択中"}
            yield {"status": "セマンティック検索でファイルを選択中..."}

        # ── Phase D: セマンティック検索で補完 ──
        # CPU専用では top_n を絞ってコンテキストサイズを抑制する（8000+ トークンを防ぐ）
        # mode == "fast" では常に top_n=3 に絞る
        _is_cpu = not getattr(self._llm, "cuda_available", False)
        _top_n = 3 if mode == "fast" else (5 if _is_cpu else 10)
        if pinned_base_chunks:
            # ピン留めファイルを起点に、不足分をセマンティック検索で補完
            base_chunks = list(pinned_base_chunks)
            seen_paths: set[str] = {c.path for c in base_chunks}
            extra = self._analysis.get_top_chunks_semantic(chunks, question, top_n=_top_n)
            for c in extra:
                if c.path not in seen_paths:
                    base_chunks.append(c)
                    seen_paths.add(c.path)
                    if len(base_chunks) >= 30:
                        break
        elif selected_paths:
            base_chunks = [chunk_map[p] for p in selected_paths if p in chunk_map]
            if len(base_chunks) < _top_n:
                seen_paths = {c.path for c in base_chunks}
                for c in self._analysis.get_top_chunks_semantic(chunks, question, top_n=_top_n):
                    if c.path not in seen_paths:
                        base_chunks.append(c)
                        seen_paths.add(c.path)
                        if len(base_chunks) >= _top_n:
                            break
        else:
            base_chunks = self._analysis.get_top_chunks_semantic(chunks, question, top_n=_top_n)

        # ── Phase E: 依存関係展開 (BFS 10 hop, カスタムインポートのみ) ──
        # mode == "fast" では BFS を完全にスキップして base_chunks をそのまま使う
        # CPU専用: max_total を絞ってプロンプトサイズを抑制する
        if mode == "fast":
            top_chunks = base_chunks
            dep_map: Dict[str, List[str]] = {}
            yield {"phase": "依存関係展開スキップ", "detail": f"高速モード: {len(base_chunks)} ファイルをそのまま使用"}
        else:
            _dep_max = 40 if pinned_base_chunks else (10 if _is_cpu else 20)
            yield {"phase": "依存関係展開中", "detail": f"{len(base_chunks)} ファイルから最大 {_dep_max} ファイルに展開 (10 hop BFS)"}
            yield {"status": f"依存関係を展開中... ({len(base_chunks)} → 最大 {_dep_max} ファイル)"}
            top_chunks, dep_map, _phase_e_depth = self._analysis.expand_with_dependencies(
                chunks,
                [c.path for c in base_chunks],
                max_total=_dep_max,
                max_depth=10,
            )
            yield {"phase": "依存関係展開中", "detail": f"展開後 {len(top_chunks)} ファイル確定"}

        # ── Phase F: ファイル内容読み込み ──
        # CPU専用でピン留めなしの場合はフル内容を省略してサマリーのみ使用する。
        # フル内容は prompt を何千トークンも膨らませ、CPU での prefill を数倍遅くする。
        # mode == "fast" では 800 文字上限で軽量読み込みする。
        _inject_full_content = mode == "fast" or bool(pinned_base_chunks) or not _is_cpu
        full_contents: List[tuple[str, str]] = []
        if _inject_full_content:
            yield {"phase": "ファイル内容読み込み中", "detail": f"{len(top_chunks)} ファイルをディスクから読み込み"}
            yield {"status": f"ファイル内容を読み込み中... ({len(top_chunks)} ファイル)"}
            _max_chars = 800 if mode == "fast" else (self._context.max_qa_file_chars() if not _is_cpu else 3000)
            _pinned_paths_set = {p for p, _ in pinned_chunk_contents}
            _chunks_to_read = [c for c in top_chunks if c.path not in _pinned_paths_set]

            def _read_chunk(chunk):
                content = self._read_file_cached(root / chunk.path, max_chars=_max_chars)
                return (chunk.path, content) if content is not None else None

            if _chunks_to_read:
                with ThreadPoolExecutor(max_workers=min(len(_chunks_to_read), 8)) as ex:
                    full_contents = [
                        r for r in ex.map(_read_chunk, _chunks_to_read) if r is not None
                    ]
        else:
            yield {"phase": "ファイル選択完了", "detail": f"CPU最適化モード: サマリーのみ使用 ({len(top_chunks)} ファイル)"}

        top_summaries = [(c.path, c.summary or "") for c in top_chunks]

        # ── Phase G: プロンプト構築 ──
        yield {"phase": "プロンプト構築中", "detail": "コンテキストを組み立てています"}
        yield {"status": "プロンプトを構築中..."}
        _history_window = history[-3:] if mode == "fast" else history[-10:]
        prompt, tokens = self._context.build_qa_prompt(
            question=question,
            project_index_json=index_json,
            top_summaries=top_summaries,
            full_contents=full_contents,
            conversation_history=_history_window,
            dep_map=dep_map,
            pinned_contents=pinned_chunk_contents or None,
            workspace_projects=workspace_project_data or None,
            pinned_depth_map=_pinned_depth_map if pinned_paths else None,
            direct_pinned_set=_direct_pinned_set if pinned_paths else None,
            mode=mode,
        )

        # プロンプトのプレビューを送信（最初の 1000 文字 + 末尾 200 文字）
        preview_head = prompt[:1000]
        preview_tail = prompt[-200:] if len(prompt) > 1200 else ""
        preview = preview_head + ("\n...[中略]...\n" + preview_tail if preview_tail else "")
        yield {"prompt_preview": preview, "prompt_tokens": tokens}

        # CPU 用 Ollama パラメータ — 固定値にすることでモデル再ロードを防ぐ。
        # 8192 は CPU モードの典型的な Q&A プロンプト（サマリーのみ）を収容するのに十分で、
        # 16384 より prefill が約 2 倍速い。
        _CPU_NUM_CTX = 8192
        _num_ctx: Optional[int] = None
        _num_predict: Optional[int] = None
        if _is_cpu:
            _num_ctx = _CPU_NUM_CTX
            _num_predict = -1

        # Wait for preload to complete before sending the prompt.
        # This guarantees the model is in RAM with no Ollama queue gap.
        # The event is set by the preload thread regardless of success/failure.
        if not _preload_event.is_set():
            yield {"phase": "Ollamaモデルロード中", "detail": f"{model} をRAMに読み込み中（コンテキスト構築と並行）"}
            while not _preload_event.wait(timeout=1.0):
                if is_cancelled():
                    yield {"done": True}
                    return
                yield {"status": f"Ollamaがモデルをロード中... ({model})"}

        yield {"phase": "Ollama生成中", "detail": f"推定 {tokens} トークン送信 → Ollama ({model})" + (f"  [num_ctx={_num_ctx}, num_predict={_num_predict}]" if _is_cpu else "")}
        yield {"status": f"Ollamaが回答を生成中... (推定 {tokens} トークン)"}

        start_time = time.time()
        answer_tokens: List[str] = []
        _first_token_received = False
        try:
            for token in self._llm.stream_completion(
                model,
                prompt,
                num_ctx=_num_ctx,
                num_predict=_num_predict,
                keep_alive="2h",
            ):
                if is_cancelled():
                    break
                if not _first_token_received:
                    _first_token_received = True
                    _load_s = time.time() - start_time
                    if _load_s > 3:
                        yield {"phase": "Ollama生成中", "detail": f"最初のトークン到着 (待機 {_load_s:.1f}s)"}
                        yield {"status": f"Ollamaが回答を生成中... (待機 {_load_s:.1f}s 後に開始)"}
                answer_tokens.append(token)
                yield {"token": token}
        except Exception as exc:
            logger.error("Q&A回答生成エラー: %s", exc)
            yield {"error": str(exc)}
            return

        elapsed = (time.time() - start_time) * 1000
        log_entry = GenerationLogEntry(
            mode="explain",
            model=model,
            operation="qa",
            prompt_tokens_estimated=tokens,
            response_time_ms=elapsed,
            status="completed",
        )
        log_path = root / _LOCALFORGE_DIR / "generation_log.jsonl"
        self._log_async(log_path, log_entry)

        # Cache the full answer for future identical questions (fire-and-forget)
        _full_answer = "".join(answer_tokens)
        if _full_answer:
            import threading as _t
            _t.Thread(
                target=_rc.set, args=(_rc_key, _full_answer), daemon=True
            ).start()

        yield {"phase": "完了", "detail": f"回答生成完了 ({elapsed/1000:.1f}s)"}
        yield {"done": True}

    def append_qa_entry(self, root: Path, question: str, answer: str) -> None:
        """
        Q&Aのやり取りを .localforge/qa_history.md に追記する。
        ファイルが存在しない場合は新規作成する。

        Args:
            root: プロジェクトルート
            question: ユーザーの質問
            answer: アシスタントの回答
        """
        qa_path = root / _LOCALFORGE_DIR / "qa_history.md"
        qa_path.parent.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        entry = f"\n## [{timestamp}]\n\n**Q:** {question}\n\n**A:** {answer.strip()}\n\n---\n"

        if not qa_path.exists():
            qa_path.write_text("# Q&A 履歴\n" + entry, encoding="utf-8")
        else:
            with qa_path.open("a", encoding="utf-8") as fh:
                fh.write(entry)

        logger.debug("Q&Aエントリを保存しました: %s", qa_path)

    def get_summary(self, root: Path) -> Optional[dict]:
        """
        ProjectIndexのサマリーを辞書形式で返す。

        Args:
            root: プロジェクトルート

        Returns:
            サマリー辞書（存在しない場合はNone）
        """
        project_index = self._analysis.load_project_index(root)
        if not project_index:
            return None
        return {
            "project_name": project_index.project_name,
            "summary": project_index.summary,
            "total_files": project_index.total_files,
            "indexed_files": project_index.indexed_files,
            "created_at": project_index.created_at.isoformat(),
            "updated_at": project_index.updated_at.isoformat(),
        }
