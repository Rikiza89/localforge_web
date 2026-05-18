"""
説明サービス — レポート生成・Q&Aオーケストレーターの責務を担う。
11セクションのレポートをSSEでストリーミング生成する。
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Generator, List, Optional

from localforge.application.analysis_service import AnalysisService
from localforge.application.context_service import ContextService
from localforge.domain.models import FileChunk, GenerationLogEntry, Message, ProjectIndex
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

_LOCALFORGE_DIR = ".localforge"

import re as _re
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

    def stream_report(
        self,
        root: Path,
        model: str,
    ) -> Generator[dict, None, None]:
        """
        11セクションのレポートをSSEイベントとしてストリーミング生成する。
        各セクションは別々のOllamaコールで処理する。
        生成完了後に <project>/.localforge/report.md として保存する。

        Args:
            root: プロジェクトルート
            model: 使用するOllamaモデル名

        Yields:
            SSEペイロード辞書（section, token, progress, done, error）
        """
        # Start loading the model immediately — overlaps with index load and parallel
        # semantic searches so the first section's first token arrives sooner.
        getattr(self._llm, "preload_model_async", lambda m: None)(model)

        project_index = self._analysis.load_project_index(root)
        if not project_index:
            yield {"error": "ProjectIndexが見つかりません。先にインデックスを構築してください。"}
            return

        chunks = project_index.file_chunks
        # レポートセクションには summary（構造化テキスト）のみ渡す。
        # ファイルパス一覧は Q&A フェーズ1専用で、レポートでは不要かつ巨大なので除外する。
        index_dict = project_index.model_dump(
            include={"project_name", "summary", "total_files", "indexed_files"}
        )
        index_json = json.dumps(index_dict, ensure_ascii=False)

        total_sections = len(REPORT_SECTIONS)
        completed_sections: List[tuple[str, str]] = []  # (name, content)

        _is_cpu = not getattr(self._llm, "cuda_available", False)
        # 4096 is sufficient for report section prompts (≤1500 tokens each)
        _r_num_ctx: Optional[int] = 4096 if _is_cpu else None
        _r_num_predict: Optional[int] = -1 if _is_cpu else None

        # Pre-fetch all section semantic searches in parallel to avoid blocking on the first section
        def _fetch_section_chunks(section_name: str) -> list:
            return self._analysis.get_top_chunks_semantic(chunks, section_name, top_n=6)

        with ThreadPoolExecutor(max_workers=min(len(REPORT_SECTIONS), 4)) as ex:
            all_relevant_chunks = list(ex.map(_fetch_section_chunks, REPORT_SECTIONS))

        for sec_idx, section_name in enumerate(REPORT_SECTIONS):
            # セクションヘッダーを送信（インデックスと合計数も含める）
            yield {
                "section": section_name,
                "section_idx": sec_idx + 1,
                "section_total": total_sections,
            }

            # セクションに関連するチャンクを選択（事前並列計算済み）
            relevant_chunks = all_relevant_chunks[sec_idx]
            # 各サマリーを1行目のみに絞ってトークンを節約する
            relevant_summaries = [
                (c.path, (c.summary or "").split("\n")[0][:120])
                for c in relevant_chunks if c.summary
            ]

            # プロンプトを構築してストリーミング生成
            prompt, tokens = self._context.build_report_section_prompt(
                section_name=section_name,
                project_index_json=index_json,
                relevant_summaries=relevant_summaries,
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
            self._analysis._index_adapter.append_log_entry(log_path, log_entry)

            completed_sections.append((section_name, "".join(section_tokens)))

            # セクション完了後に進捗を送信（done = 完了済みセクション数）
            yield {
                "progress": {
                    "done": sec_idx + 1,
                    "total": total_sections,
                    "current_file": section_name,
                }
            }

        # レポートをディスクに保存
        self._save_report(root, completed_sections, project_index.project_name)

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
    ) -> None:
        """
        レポートをMarkdown形式で .localforge/report.md に保存する。

        Args:
            root: プロジェクトルート
            sections: [(セクション名, 内容)] のリスト
            project_name: プロジェクト名（ドキュメントタイトル用）
        """
        report_path = root / _LOCALFORGE_DIR / "report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)

        lines: List[str] = [f"# {project_name} — Codebase Report\n\n"]
        for name, content in sections:
            lines.append(f"## {name}\n\n")
            lines.append(_strip_leading_heading(content, name))
            lines.append("\n\n---\n\n")

        report_path.write_text("".join(lines), encoding="utf-8")
        logger.info("レポートを保存しました: %s", report_path)

    def stream_answer(
        self,
        root: Path,
        model: str,
        question: str,
        history: List[Message],
        workspace_roots: Optional[List[tuple[Path, str]]] = None,
        pinned_paths: Optional[List[str]] = None,
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
        project_index = self._analysis.load_project_index(root)
        if not project_index:
            yield {"error": "ProjectIndexが見つかりません。先にインデックスを構築してください。"}
            return

        # Fire model warm-up immediately — phases A-G run in parallel with model loading.
        # By the time context assembly finishes the model is already in RAM (or much
        # further along), so the wait at stream_completion is greatly reduced.
        _model_was_cold = getattr(self._llm, "is_model_loaded", lambda m: None)(model) is False
        if _model_was_cold:
            yield {"status": f"モデルをバックグラウンドでプリロード中... ({model})"}
        getattr(self._llm, "preload_model_async", lambda m: None)(model)

        chunks = project_index.file_chunks
        chunk_map = {c.path: c for c in chunks}
        all_summaries = [(c.path, c.summary or "") for c in chunks if c.summary]

        index_dict = project_index.model_dump(include={"project_name", "summary", "total_files", "indexed_files"})
        index_dict["files"] = [c.path for c in chunks]
        index_json = json.dumps(index_dict, ensure_ascii=False)

        # ── Phase A: ピン留めコンテキスト解決 ──
        # ピン留めファイルはユーザーが明示的に選んだもの。全内容を読み込む（文字数上限なし）。
        # トークン予算は build_qa_prompt 内で管理する。
        pinned_chunk_contents: List[tuple[str, str]] = []
        pinned_base_chunks: List = []
        if pinned_paths:
            yield {"phase": "ピン留めファイル解決中", "detail": f"{len(pinned_paths)} 件のパスを展開中"}
            yield {"status": f"ピン留めファイルを展開中... ({len(pinned_paths)} 件)"}
            try:
                # ピン留めの場合は依存関係をより深く (max_total=40) 展開する
                pinned_chunks, _ = self._analysis.resolve_pinned_chunks(
                    root, pinned_paths, chunks, max_total=40
                )
                pinned_base_chunks = pinned_chunks
                yield {"phase": "ピン留めファイル解決中", "detail": f"依存関係を含む {len(pinned_chunks)} ファイルを取得"}

                def _read_pinned(pc):
                    fp = root / pc.path
                    if fp.exists():
                        try:
                            return (pc.path, fp.read_text(encoding="utf-8", errors="replace"))
                        except OSError as exc:
                            logger.warning("ピン留めファイル読み込みエラー [%s]: %s", pc.path, exc)
                    return None

                with ThreadPoolExecutor(max_workers=min(len(pinned_chunks), 8)) as ex:
                    pinned_chunk_contents = [
                        r for r in ex.map(_read_pinned, pinned_chunks) if r is not None
                    ]
            except Exception as exc:
                logger.warning("ピン留めコンテキスト解決エラー: %s", exc)

        # ── Phase B: ワークスペース他プロジェクトのコンテキスト ──
        workspace_project_data: List[dict] = []
        if workspace_roots:
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
                    ws_exp_chunks, _ = self._analysis.expand_with_dependencies(
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
        _is_cpu = not getattr(self._llm, "cuda_available", False)
        _top_n = 5 if _is_cpu else 10
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

        # ── Phase E: 依存関係展開 (BFS 5 hop, カスタムインポートのみ) ──
        # CPU専用: max_total を絞ってプロンプトサイズを抑制する
        _dep_max = 40 if pinned_base_chunks else (10 if _is_cpu else 20)
        yield {"phase": "依存関係展開中", "detail": f"{len(base_chunks)} ファイルから最大 {_dep_max} ファイルに展開 (5 hop BFS)"}
        yield {"status": f"依存関係を展開中... ({len(base_chunks)} → 最大 {_dep_max} ファイル)"}
        top_chunks, dep_map = self._analysis.expand_with_dependencies(
            chunks,
            [c.path for c in base_chunks],
            max_total=_dep_max,
        )
        yield {"phase": "依存関係展開中", "detail": f"展開後 {len(top_chunks)} ファイル確定"}

        # ── Phase F: ファイル内容読み込み ──
        # CPU専用でピン留めなしの場合はフル内容を省略してサマリーのみ使用する。
        # フル内容は prompt を何千トークンも膨らませ、CPU での prefill を数倍遅くする。
        _inject_full_content = bool(pinned_base_chunks) or not _is_cpu
        full_contents: List[tuple[str, str]] = []
        if _inject_full_content:
            yield {"phase": "ファイル内容読み込み中", "detail": f"{len(top_chunks)} ファイルをディスクから読み込み"}
            yield {"status": f"ファイル内容を読み込み中... ({len(top_chunks)} ファイル)"}
            _max_chars = self._context.max_qa_file_chars() if not _is_cpu else 3000
            _pinned_paths_set = {p for p, _ in pinned_chunk_contents}
            _chunks_to_read = [c for c in top_chunks if c.path not in _pinned_paths_set]

            def _read_chunk(chunk):
                file_path = root / chunk.path
                if file_path.exists():
                    try:
                        content = file_path.read_text(encoding="utf-8", errors="replace")
                        return (chunk.path, content[:_max_chars])
                    except OSError:
                        pass
                return None

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
        prompt, tokens = self._context.build_qa_prompt(
            question=question,
            project_index_json=index_json,
            top_summaries=top_summaries,
            full_contents=full_contents,
            conversation_history=history[-10:],
            dep_map=dep_map,
            pinned_contents=pinned_chunk_contents or None,
            workspace_projects=workspace_project_data or None,
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

        # Check model load status — if still not loaded the preload thread is racing
        # Ollama's queue so the wait here is reduced by however long preprocessing took.
        model_loaded = getattr(self._llm, "is_model_loaded", lambda m: None)(model)
        if model_loaded is False:
            if _model_was_cold:
                yield {"phase": "Ollamaモデルロード中", "detail": f"プリロード中... コンテキスト準備と並行してRAMに読み込んでいます"}
            else:
                yield {"phase": "Ollamaモデルロード中", "detail": f"{model} をRAMに読み込み中"}
            yield {"status": f"Ollamaがモデルをロード中... ({model}) — 最初のトークンが来るまでお待ちください"}
        else:
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
        self._analysis._index_adapter.append_log_entry(log_path, log_entry)

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
