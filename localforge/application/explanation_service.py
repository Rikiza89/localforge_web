"""
説明サービス — レポート生成・Q&Aオーケストレーターの責務を担う。
11セクションのレポートをSSEでストリーミング生成する。
"""

from __future__ import annotations

import json
import logging
import time
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

        for sec_idx, section_name in enumerate(REPORT_SECTIONS):
            # セクションヘッダーを送信（インデックスと合計数も含める）
            yield {
                "section": section_name,
                "section_idx": sec_idx + 1,
                "section_total": total_sections,
            }

            # セクションに関連するチャンクを選択（top_n=6 でプロンプトを抑制）
            relevant_chunks = self._analysis.get_top_chunks_semantic(
                chunks, section_name, top_n=6
            )
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
                for pc in pinned_chunks:
                    fp = root / pc.path
                    if fp.exists():
                        try:
                            # 文字数上限なし — build_qa_prompt のトークン予算に委ねる
                            content = fp.read_text(encoding="utf-8", errors="replace")
                            pinned_chunk_contents.append((pc.path, content))
                        except OSError as exc:
                            logger.warning("ピン留めファイル読み込みエラー [%s]: %s", pc.path, exc)
            except Exception as exc:
                logger.warning("ピン留めコンテキスト解決エラー: %s", exc)

        # ── Phase B: ワークスペース他プロジェクトのコンテキスト ──
        workspace_project_data: List[dict] = []
        if workspace_roots:
            yield {"phase": "ワークスペース検索中", "detail": f"{len(workspace_roots)} プロジェクト"}
            yield {"status": "ワークスペースプロジェクトを検索中..."}
            for ws_root, ws_name in workspace_roots[:3]:
                try:
                    ws_idx = self._analysis.load_project_index(ws_root)
                    if not ws_idx:
                        continue
                    ws_chunks = ws_idx.file_chunks
                    ws_top = self._analysis.get_top_chunks_semantic(ws_chunks, question, top_n=5)
                    ws_exp_chunks, _ = self._analysis.expand_with_dependencies(
                        ws_chunks, [c.path for c in ws_top], max_total=8
                    )
                    ws_summaries = [(c.path, c.summary or "") for c in ws_exp_chunks if c.summary]
                    ws_contents: List[tuple[str, str]] = []
                    _max_ws_chars = self._context.max_qa_file_chars() // 2
                    for wc in ws_exp_chunks[:3]:
                        fp = ws_root / wc.path
                        if fp.exists():
                            try:
                                content = fp.read_text(encoding="utf-8", errors="replace")
                                ws_contents.append((wc.path, content[:_max_ws_chars]))
                            except OSError:
                                pass
                    workspace_project_data.append({
                        "name": ws_name,
                        "root": str(ws_root),
                        "summaries": ws_summaries,
                        "contents": ws_contents,
                    })
                except Exception as exc:
                    logger.warning("ワークスペース [%s] Q&Aエラー: %s", ws_name, exc)

        # ── Phase C: ファイル選択 ──
        # ピン留めファイルが存在する場合は LLM によるファイル選択 (Phase 1) をスキップする。
        # ユーザーが既に関連ファイルを指定しているため、余分な LLM 呼び出しは不要かつ CPU では数分かかる。
        selected_paths: List[str] = []
        if pinned_base_chunks:
            # ピン留めを優先。セマンティック検索で補完のみ行う
            yield {"phase": "ファイル選択", "detail": f"ピン留め {len(pinned_base_chunks)} ファイルを使用 (LLM選択スキップ)"}
            yield {"status": f"ピン留めファイルを使用中 ({len(pinned_base_chunks)} 件) — LLMファイル選択をスキップ"}
        else:
            # ピン留めなし: LLM にどのファイルが必要か選ばせる (Phase 1)
            dep_hints = {c.path: c.imports_resolved for c in chunks if c.imports_resolved}
            yield {"phase": "LLMファイル選択中", "detail": f"{len(all_summaries)} ファイルから関連ファイルを選択"}
            yield {"status": "LLMがファイルを選択中... (CPU では数分かかる場合があります)"}
            try:
                sel_prompt = self._context.build_qa_file_selection_prompt(
                    question, all_summaries, dep_hints=dep_hints
                )
                # CPU での長時間ブロックを防ぐため短めのタイムアウト (120s) を使用
                sel_response = self._llm.generate_sync(model, sel_prompt, read_timeout=120)
                # JSON 配列を堅牢に抽出する
                _start = sel_response.find("[")
                _end = sel_response.rfind("]")
                if _start != -1 and _end != -1:
                    _parsed = json.loads(sel_response[_start:_end + 1])
                    if isinstance(_parsed, list):
                        selected_paths = [p for p in _parsed if isinstance(p, str) and p in chunk_map]
                if selected_paths:
                    yield {"phase": "LLMファイル選択中", "detail": f"{len(selected_paths)} ファイルを選択"}
            except Exception as exc:
                logger.warning("ファイル選択フェーズエラー（フォールバック）: %s", exc)
                yield {"phase": "LLMファイル選択中", "detail": f"エラーのためセマンティック検索にフォールバック: {exc}"}

        # ── Phase D: セマンティック検索で補完 ──
        if pinned_base_chunks:
            # ピン留めファイルを起点に、不足分をセマンティック検索で補完
            base_chunks = list(pinned_base_chunks)
            seen_paths: set[str] = {c.path for c in base_chunks}
            extra = self._analysis.get_top_chunks_semantic(chunks, question, top_n=10)
            for c in extra:
                if c.path not in seen_paths:
                    base_chunks.append(c)
                    seen_paths.add(c.path)
                    if len(base_chunks) >= 30:
                        break
        elif selected_paths:
            base_chunks = [chunk_map[p] for p in selected_paths if p in chunk_map]
            if len(base_chunks) < 10:
                seen_paths = {c.path for c in base_chunks}
                for c in self._analysis.get_top_chunks_semantic(chunks, question, top_n=10):
                    if c.path not in seen_paths:
                        base_chunks.append(c)
                        seen_paths.add(c.path)
                        if len(base_chunks) >= 10:
                            break
        else:
            base_chunks = self._analysis.get_top_chunks_semantic(chunks, question, top_n=10)

        # ── Phase E: 依存関係展開 (BFS 5 hop, カスタムインポートのみ) ──
        # ピン留め時は max_total を大きくして深い依存関係まで取得する
        _dep_max = 40 if pinned_base_chunks else 20
        yield {"phase": "依存関係展開中", "detail": f"{len(base_chunks)} ファイルから最大 {_dep_max} ファイルに展開 (5 hop BFS)"}
        yield {"status": f"依存関係を展開中... ({len(base_chunks)} → 最大 {_dep_max} ファイル)"}
        top_chunks, dep_map = self._analysis.expand_with_dependencies(
            chunks,
            [c.path for c in base_chunks],
            max_total=_dep_max,
        )
        yield {"phase": "依存関係展開中", "detail": f"展開後 {len(top_chunks)} ファイル確定"}

        # ── Phase F: ファイル内容読み込み ──
        yield {"phase": "ファイル内容読み込み中", "detail": f"{len(top_chunks)} ファイルをディスクから読み込み"}
        yield {"status": f"ファイル内容を読み込み中... ({len(top_chunks)} ファイル)"}
        full_contents: List[tuple[str, str]] = []
        for chunk in top_chunks:
            # ピン留めファイルはすでに pinned_chunk_contents に含まれているので重複しない
            if pinned_base_chunks and any(p == chunk.path for p, _ in pinned_chunk_contents):
                continue
            file_path = root / chunk.path
            if file_path.exists():
                try:
                    content = file_path.read_text(encoding="utf-8", errors="replace")
                    full_contents.append((chunk.path, content))
                except OSError:
                    pass

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
        yield {"phase": "Ollama生成中", "detail": f"推定 {tokens} トークン送信 → Ollama ({model})"}
        yield {"status": f"Ollamaが回答を生成中... (推定 {tokens} トークン)"}

        start_time = time.time()
        answer_tokens: List[str] = []
        try:
            for token in self._llm.stream_completion(model, prompt):
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
