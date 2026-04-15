"""
説明サービス — レポート生成・Q&Aオーケストレーターの責務を担う。
11セクションのレポートをSSEでストリーミング生成する。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator, List, Optional

from localforge.application.analysis_service import AnalysisService
from localforge.application.context_service import ContextService
from localforge.domain.models import FileChunk, Message, ProjectIndex
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

        index_json = project_index.model_dump_json(
            include={"project_name", "summary", "total_files", "indexed_files"}
        )
        chunks = project_index.file_chunks

        total_sections = len(REPORT_SECTIONS)

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

            # セクションに関連するチャンクを選択
            relevant_chunks = self._analysis.get_top_chunks_by_keywords(
                chunks, section_name, top_n=5
            )
            relevant_summaries = [
                (c.path, c.summary or "") for c in relevant_chunks if c.summary
            ]

            # プロンプトを構築してストリーミング生成
            prompt = self._context.build_report_section_prompt(
                section_name=section_name,
                project_index_json=index_json,
                relevant_summaries=relevant_summaries,
            )

            try:
                for token in self._llm.stream_completion(model, prompt):
                    yield {"token": token}
            except Exception as exc:
                logger.error("セクション生成エラー [%s]: %s", section_name, exc)
                yield {"token": f"\n[エラー: {exc}]\n"}

        yield {
            "progress": {
                "done": total_sections,
                "total": total_sections,
                "current_file": "完了",
            }
        }
        yield {"done": True}

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

        index_json = project_index.model_dump_json(
            include={"project_name", "summary", "total_files"}
        )
        chunks = project_index.file_chunks

        # キーワードで上位5件を選択
        top_chunks = self._analysis.get_top_chunks_by_keywords(chunks, question, top_n=5)

        # ハイブリッドファイルのフルコンテンツ（top_5のうちhybridのもの）
        full_contents: List[tuple[str, str]] = []
        for chunk in top_chunks:
            if chunk.strategy.value == "hybrid":
                file_path = root / chunk.path
                if file_path.exists():
                    try:
                        content = file_path.read_text(encoding="utf-8", errors="replace")
                        full_contents.append((chunk.path, content[:3000]))
                    except OSError:
                        pass

        top_summaries = [(c.path, c.summary or "") for c in top_chunks]

        prompt = self._context.build_qa_prompt(
            question=question,
            project_index_json=index_json,
            top_summaries=top_summaries,
            full_contents=full_contents,
            conversation_history=history[-10:],
        )

        try:
            for token in self._llm.stream_completion(model, prompt):
                yield {"token": token}
        except Exception as exc:
            logger.error("Q&A回答生成エラー: %s", exc)
            yield {"error": str(exc)}
            return

        yield {"done": True}

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
