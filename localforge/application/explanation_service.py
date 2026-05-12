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
        index_dict = project_index.model_dump(
            include={"project_name", "summary", "total_files", "indexed_files"}
        )
        index_dict["files"] = [c.path for c in chunks]
        index_json = json.dumps(index_dict, ensure_ascii=False)

        total_sections = len(REPORT_SECTIONS)
        completed_sections: List[tuple[str, str]] = []  # (name, content)

        for sec_idx, section_name in enumerate(REPORT_SECTIONS):
            # セクションヘッダーを送信
            yield {"section": section_name}
            yield {
                "progress": {
                    "done": sec_idx,
                    "total": total_sections,
                    "current_file": section_name,
                }
            }

            # セクションに関連するチャンクを選択（セマンティック検索 → キーワードフォールバック）
            relevant_chunks = self._analysis.get_top_chunks_semantic(
                chunks, section_name, top_n=10
            )
            relevant_summaries = [
                (c.path, c.summary or "") for c in relevant_chunks if c.summary
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
            lines.append(content.strip())
            lines.append("\n\n---\n\n")

        report_path.write_text("".join(lines), encoding="utf-8")
        logger.info("レポートを保存しました: %s", report_path)

    def stream_answer(
        self,
        root: Path,
        model: str,
        question: str,
        history: List[Message],
    ) -> Generator[dict, None, None]:
        """
        Q&A質問への回答をSSEイベントとしてストリーミング生成する。

        Args:
            root: プロジェクトルート
            model: 使用するOllamaモデル名
            question: ユーザーの質問
            history: 会話履歴（最大10件）

        Yields:
            SSEペイロード辞書（token, done, error）
        """
        project_index = self._analysis.load_project_index(root)
        if not project_index:
            yield {"error": "ProjectIndexが見つかりません。先にインデックスを構築してください。"}
            return

        chunks = project_index.file_chunks
        chunk_map = {c.path: c for c in chunks}
        all_summaries = [(c.path, c.summary or "") for c in chunks if c.summary]

        # A5: ファイル一覧を含む index_json を組み立てる（LLM がプロジェクト全体を把握できるように）
        index_dict = project_index.model_dump(include={"project_name", "summary", "total_files", "indexed_files"})
        index_dict["files"] = [c.path for c in chunks]
        index_json = json.dumps(index_dict, ensure_ascii=False)

        # A6 フェーズ 1: LLM にどのファイルが必要か選ばせる
        yield {"status": "関連ファイルを分析中..."}
        selected_paths: List[str] = []
        try:
            sel_prompt = self._context.build_qa_file_selection_prompt(question, all_summaries)
            sel_response = self._llm.generate_sync(model, sel_prompt)
            # JSON 配列を抽出（余計なテキストが混入しても壊れないようにする）
            _start = sel_response.find("[")
            _end = sel_response.rfind("]")
            if _start != -1 and _end != -1:
                selected_paths = json.loads(sel_response[_start:_end + 1])
                if not isinstance(selected_paths, list):
                    selected_paths = []
                selected_paths = [p for p in selected_paths if isinstance(p, str) and p in chunk_map]
        except Exception as exc:
            logger.warning("ファイル選択フェーズエラー（フォールバック）: %s", exc)

        # フォールバック: フェーズ 1 が失敗または空の場合はセマンティック検索 top-10 を使用
        if not selected_paths:
            top_chunks = self._analysis.get_top_chunks_semantic(chunks, question, top_n=10)
        else:
            # フェーズ 1 の選択を優先する。不足分(10件未満)のみセマンティック検索で補完する
            top_chunks = [chunk_map[p] for p in selected_paths if p in chunk_map]
            if len(top_chunks) < 10:
                seen: set[str] = {c.path for c in top_chunks}
                for c in self._analysis.get_top_chunks_semantic(chunks, question, top_n=10):
                    if c.path not in seen:
                        top_chunks.append(c)
                        seen.add(c.path)
                        if len(top_chunks) >= 10:
                            break

        # A1 + B8: フルコンテンツ注入
        # - FULL 戦略チャンク（≤200 行）: chunk.content をそのまま使う（ディスク再読み不要）
        # - HYBRID 戦略チャンク（>200 行）: ファイルを再読みして上限文字数まで注入
        _max_chars = self._context.max_qa_file_chars()
        full_contents: List[tuple[str, str]] = []
        for chunk in top_chunks:
            if chunk.strategy.value == "full":
                if chunk.content:
                    full_contents.append((chunk.path, chunk.content[:_max_chars]))
            else:  # hybrid
                file_path = root / chunk.path
                if file_path.exists():
                    try:
                        content = file_path.read_text(encoding="utf-8", errors="replace")
                        full_contents.append((chunk.path, content[:_max_chars]))
                    except OSError:
                        pass

        top_summaries = [(c.path, c.summary or "") for c in top_chunks]

        prompt, tokens = self._context.build_qa_prompt(
            question=question,
            project_index_json=index_json,
            top_summaries=top_summaries,
            full_contents=full_contents,
            conversation_history=history[-10:],
        )

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
