"""
コンテキストサービス — LLMプロンプト組み立てとトークン予算管理の唯一の責務を担う。
すべてのLLMプロンプト構築はこのサービスを通じてのみ行われる。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from localforge.domain.exceptions import TokenBudgetExceededWarning
from localforge.domain.models import FileChunk, GenerationPlan, Message, ProjectIndex

logger = logging.getLogger(__name__)

# トークン推定係数（単語数 × この係数でトークン数を推定）
_WORDS_TO_TOKENS = 1.3
# デフォルトトークン上限
_DEFAULT_TOKEN_LIMIT = 6000


def _estimate_tokens(text: str) -> int:
    """
    テキストのトークン数を単語数から推定する。

    Args:
        text: 推定対象テキスト

    Returns:
        推定トークン数
    """
    return int(len(text.split()) * _WORDS_TO_TOKENS)


class ContextService:
    """
    LLMへのコンテキスト（プロンプト）を組み立てるサービスクラス。
    すべてのプロンプト構築ロジックをここに集約する。
    トークン予算超過時は最古のファイルコンテキストから切り詰める。
    """

    def __init__(self, token_limit: int = _DEFAULT_TOKEN_LIMIT) -> None:
        """
        ContextServiceを初期化する。

        Args:
            token_limit: トークン予算の上限値
        """
        self._token_limit = token_limit

    def update_token_limit(self, limit: int) -> None:
        """トークン上限を更新する。"""
        self._token_limit = limit

    def _guard_budget(self, prompt: str, label: str = "") -> str:
        """
        プロンプトのトークン数を確認し、上限を超えている場合は警告ログを出す。
        超過してもプロンプトはそのまま返す（切り詰めはしない）。

        Args:
            prompt: チェックするプロンプト
            label: ログ用のラベル文字列

        Returns:
            入力プロンプト（変更なし）
        """
        estimated = _estimate_tokens(prompt)
        if estimated > self._token_limit:
            msg = (
                f"トークン予算超過 [{label}]: 推定={estimated}, 上限={self._token_limit}"
            )
            logger.warning(msg)
            # 警告例外はログのみ（処理継続）
        return prompt

    # ------------------------------------------------------------------
    # Generateモード用コンテキスト
    # ------------------------------------------------------------------

    def build_plan_prompt(
        self,
        user_prompt: str,
        folder_name: str,
        file_tree_text: str,
        context_md: str,
        git_log: str,
        file_summaries: Optional[List[tuple[str, str]]] = None,
    ) -> str:
        """
        プロジェクト生成・改善プランのプロンプトを組み立てる。
        既存プロジェクトのインデックスサマリーがある場合はRAGコンテキストとして注入する。

        Args:
            user_prompt: ユーザーの自然言語プロンプト
            folder_name: プロジェクトフォルダ名
            file_tree_text: ファイルツリーのテキスト表現
            context_md: context.mdの内容
            git_log: gitログ（直近5コミット）
            file_summaries: RAG検索で選出した既存ファイルサマリーのリスト（任意）

        Returns:
            組み立てたプロンプト文字列
        """
        parts = [
            f"プロジェクト名: {folder_name}",
        ]
        if file_tree_text.strip():
            parts.append(f"現在のファイル構成:\n{file_tree_text}")

        # E: RAG-based existing file summaries for context-awareness
        if file_summaries:
            used = [(p, s) for p, s in file_summaries if s][:25]
            if used:
                summaries_text = "\n".join(f"- {p}: {s}" for p, s in used)
                parts.append(f"既存ファイルの内容サマリー（関連度順）:\n{summaries_text}")

        if context_md.strip():
            parts.append(f"プロジェクトコンテキスト:\n{context_md}")
        if git_log.strip():
            parts.append(f"最近のgitコミット:\n{git_log}")

        parts.append(
            f"\nユーザーの要求:\n{user_prompt}\n\n"
            "まず、このプランで何をするかを2〜3文のMarkdown形式で簡潔に説明してください"
            "（新規作成ファイル数・修正ファイル数・主な変更点を含む）。\n"
            "その後、以下のJSON形式でプランを出力してください。\n"
            "既存ファイルを改善・修正する場合は action を \"modify\" にして"
            " modification_notes に変更内容を具体的に記述してください。\n"
            "新規ファイルは action を \"create\" にしてください。\n"
            "```json\n"
            "{\n"
            '  "project_name": "プロジェクト名",\n'
            '  "description": "プロジェクトの概要説明",\n'
            '  "files": [\n'
            '    {\n'
            '      "path": "相対ファイルパス",\n'
            '      "description": "このファイルの役割・内容の説明",\n'
            '      "action": "create または modify",\n'
            '      "modification_notes": "修正の場合: 変更内容の詳細（新規ファイルはnull）",\n'
            '      "dependencies": ["依存するファイルパスのリスト"]\n'
            '    }\n'
            '  ]\n'
            "}\n"
            "```"
        )

        prompt = "\n\n".join(parts)
        return self._guard_budget(prompt, "generate_plan")

    def build_file_generation_prompt(
        self,
        target_file: str,
        target_description: str,
        context_md: str,
        plan_json: str,
        dependency_contents: List[tuple[str, str]],
    ) -> str:
        """
        個別ファイル生成のプロンプトを組み立てる。

        Args:
            target_file: 生成対象ファイルの相対パス
            target_description: ファイルの説明
            context_md: context.mdの内容
            plan_json: 承認済みプランのJSON文字列
            dependency_contents: [(依存ファイルパス, 内容)] のリスト

        Returns:
            組み立てたプロンプト文字列
        """
        parts = [
            f"生成対象ファイル: {target_file}",
            f"役割・内容の説明: {target_description}",
        ]

        if context_md.strip():
            parts.append(f"プロジェクトコンテキスト:\n{context_md}")

        if plan_json.strip():
            parts.append(f"プロジェクト全体の計画:\n{plan_json}")

        # トークン予算内で依存ファイルを注入（古いものから切り詰め）
        dep_parts: List[str] = []
        for dep_path, dep_content in dependency_contents:
            dep_parts.append(f"--- {dep_path} ---\n{dep_content}")

        available = self._token_limit - _estimate_tokens("\n\n".join(parts))
        for dep_text in dep_parts:
            if _estimate_tokens(dep_text) < available:
                parts.append(f"依存ファイル:\n{dep_text}")
                available -= _estimate_tokens(dep_text)
            else:
                logger.warning(
                    "トークン予算不足のため依存ファイルを省略: %s", dep_text[:100]
                )
                break

        parts.append(
            f"\n{target_file} の完全なソースコードのみを出力してください。"
            " マークダウンのコードブロック（```）は使用せず、ファイルの内容だけを出力してください。"
        )

        prompt = "\n\n".join(parts)
        return self._guard_budget(prompt, f"generate_file:{target_file}")

    def build_file_edit_prompt(
        self,
        target_file: str,
        modification_notes: str,
        existing_content: str,
        context_md: str,
        plan_json: str,
        dependency_contents: List[tuple[str, str]],
    ) -> str:
        """
        既存ファイル編集のプロンプトを組み立てる。
        既存のファイル内容と変更指示を渡し、LLMに修正済み完全ソースを出力させる。

        Args:
            target_file: 編集対象ファイルの相対パス
            modification_notes: 何をどう変更するかの具体的な説明
            existing_content: 現在のファイル内容
            context_md: context.mdの内容
            plan_json: 承認済みプランのJSON文字列
            dependency_contents: [(依存ファイルパス, 内容)] のリスト

        Returns:
            組み立てたプロンプト文字列
        """
        parts = [
            f"編集対象ファイル: {target_file}",
            f"変更内容: {modification_notes}",
            f"現在のファイル内容:\n{existing_content}",
        ]

        if context_md.strip():
            parts.append(f"プロジェクトコンテキスト:\n{context_md}")
        if plan_json.strip():
            parts.append(f"プロジェクト全体の計画:\n{plan_json}")

        dep_parts: List[str] = []
        for dep_path, dep_content in dependency_contents:
            dep_parts.append(f"--- {dep_path} ---\n{dep_content}")

        available = self._token_limit - _estimate_tokens("\n\n".join(parts))
        for dep_text in dep_parts:
            if _estimate_tokens(dep_text) < available:
                parts.append(f"依存ファイル:\n{dep_text}")
                available -= _estimate_tokens(dep_text)
            else:
                logger.warning(
                    "トークン予算不足のため依存ファイルを省略: %s", dep_text[:100]
                )
                break

        parts.append(
            f"\n上記の変更内容を反映した {target_file} の完全なソースコードを出力してください。"
            " 変更に関係のない部分も含め、ファイル全体を出力してください。"
            " マークダウンのコードブロック（```）は使用せず、ソースコードだけを出力してください。"
        )

        prompt = "\n\n".join(parts)
        return self._guard_budget(prompt, f"edit_file:{target_file}")

    def build_context_update_prompt(
        self,
        previous_context: str,
        new_file_path: str,
        new_file_first_200_lines: str,
    ) -> str:
        """
        context.md更新のプロンプトを組み立てる。

        Args:
            previous_context: 現在のcontext.mdの内容
            new_file_path: 新しく生成したファイルのパス
            new_file_first_200_lines: 新ファイルの最初の200行

        Returns:
            組み立てたプロンプト文字列
        """
        parts = []
        if previous_context.strip():
            parts.append(f"現在のコンテキスト:\n{previous_context}")

        parts.append(
            f"新しく生成されたファイル: {new_file_path}\n"
            f"内容（先頭200行）:\n{new_file_first_200_lines}"
        )
        parts.append(
            "上記の情報を踏まえてプロジェクトのコンテキストメモをマークダウン形式で更新してください。"
            " 変更点・追加内容・依存関係などを簡潔にまとめてください。"
            " 更新後のコンテキスト全文のみを出力してください。"
        )

        prompt = "\n\n".join(parts)
        return self._guard_budget(prompt, "context_update")

    # ------------------------------------------------------------------
    # Explainモード用コンテキスト
    # ------------------------------------------------------------------

    def build_batch_file_summary_prompt(
        self,
        file_chunks: List["FileChunk"],
        content_limit: int = 400,
    ) -> str:
        """
        複数ファイルを一括でサマリー生成するプロンプトを組み立てる。
        1回のLLM呼び出しで複数ファイルのサマリーを取得することで処理を高速化する。

        Args:
            file_chunks: FileChunkのリスト
            content_limit: バッチプロンプト内で使う1ファイルあたりの最大文字数

        Returns:
            組み立てたプロンプト文字列
        """
        sections = []
        for chunk in file_chunks:
            excerpt = chunk.content[:content_limit]
            sections.append(f"FILE: {chunk.path}\n{excerpt}")

        prompt = (
            "各ファイルの役割を1文で要約してください。\n"
            "出力形式: FILE: <パス>\\nSUMMARY: <要約>\n\n"
            + "\n\n".join(sections)
        )
        return self._guard_budget(prompt, "batch_file_summary")

    def build_file_summary_prompt(
        self,
        file_path: str,
        content: str,
        extension: str,
    ) -> str:
        """
        ファイルサマリー生成のプロンプトを組み立てる。

        Args:
            file_path: ファイルの相対パス
            content: ファイルの内容（full or hybrid）
            extension: ファイルの拡張子

        Returns:
            組み立てたプロンプト文字列
        """
        prompt = (
            f"ファイル: {file_path} (拡張子: {extension})\n\n"
            f"{content}\n\n"
            "このファイルの役割、主要なクラス・関数・エクスポート、依存関係を3〜5文の日本語で要約してください。"
            " 要約のみを出力してください。"
        )
        return self._guard_budget(prompt, f"file_summary:{file_path}")

    def build_project_index_prompt(
        self,
        file_summaries: List[tuple[str, str]],
        folder_tree: str,
        root_configs: str,
    ) -> str:
        """
        ProjectIndexマスタードキュメント生成のプロンプトを組み立てる。

        Args:
            file_summaries: [(ファイルパス, サマリー)] のリスト
            folder_tree: ディレクトリツリーのテキスト表現
            root_configs: ルート設定ファイルの内容

        Returns:
            組み立てたプロンプト文字列
        """
        summaries_text = "\n".join(
            f"- {path}: {summary}" for path, summary in file_summaries
        )

        # トークン予算管理：サマリーが多すぎる場合は切り詰め
        if _estimate_tokens(summaries_text) > self._token_limit * 0.6:
            truncated_summaries = file_summaries[: int(len(file_summaries) * 0.6)]
            summaries_text = "\n".join(
                f"- {path}: {summary}" for path, summary in truncated_summaries
            )
            logger.warning("ProjectIndexプロンプト: サマリー数を削減しました")

        parts = [
            f"ディレクトリ構成:\n{folder_tree}",
        ]
        if root_configs.strip():
            parts.append(f"ルート設定ファイル:\n{root_configs}")
        parts.append(f"ファイルサマリー:\n{summaries_text}")
        parts.append(
            "上記の情報を基に、プロジェクト全体の概要を3〜5文の日本語で作成してください。"
            " プロジェクトの目的、主要コンポーネント、技術スタックを含めてください。"
            " 概要のみを出力してください。"
        )

        prompt = "\n\n".join(parts)
        return self._guard_budget(prompt, "project_index")

    def build_report_section_prompt(
        self,
        section_name: str,
        project_index_json: str,
        relevant_summaries: List[tuple[str, str]],
    ) -> str:
        """
        レポートセクション生成のプロンプトを組み立てる。

        Args:
            section_name: セクション名
            project_index_json: ProjectIndexのJSON文字列
            relevant_summaries: [(ファイルパス, サマリー)] のリスト

        Returns:
            組み立てたプロンプト文字列
        """
        summaries_text = "\n".join(
            f"- {path}: {summary}" for path, summary in relevant_summaries
        )

        prompt = (
            f"プロジェクト概要:\n{project_index_json}\n\n"
            f"関連ファイルサマリー:\n{summaries_text}\n\n"
            f"上記の情報を基に、以下のセクションについて詳細な分析を日本語で記述してください。\n"
            f"セクション: {section_name}\n\n"
            f"このセクションの内容のみを出力してください。マークダウン形式で記述してください。"
        )
        return self._guard_budget(prompt, f"report_section:{section_name}")

    def build_qa_prompt(
        self,
        question: str,
        project_index_json: str,
        top_summaries: List[tuple[str, str]],
        full_contents: List[tuple[str, str]],
        conversation_history: List[Message],
    ) -> str:
        """
        Q&A回答のプロンプトを組み立てる。

        Args:
            question: ユーザーの質問
            project_index_json: ProjectIndexのJSON文字列
            top_summaries: 上位5件のファイルサマリー
            full_contents: ハイブリッドファイルの全内容
            conversation_history: 最近10件の会話履歴

        Returns:
            組み立てたプロンプト文字列
        """
        parts = [f"プロジェクト概要:\n{project_index_json}"]

        summaries_text = "\n".join(
            f"- {p}: {s}" for p, s in top_summaries
        )
        if summaries_text:
            parts.append(f"関連ファイルサマリー:\n{summaries_text}")

        # フルコンテンツの注入（予算管理）
        available = self._token_limit - _estimate_tokens("\n\n".join(parts))
        for fc_path, fc_content in full_contents:
            fc_text = f"--- {fc_path} ---\n{fc_content}"
            if _estimate_tokens(fc_text) < available:
                parts.append(f"ファイル内容:\n{fc_text}")
                available -= _estimate_tokens(fc_text)
            else:
                break

        # 会話履歴（最大10件）
        if conversation_history:
            history_text = "\n".join(
                f"{'ユーザー' if m.role == 'user' else 'アシスタント'}: {m.content}"
                for m in conversation_history[-10:]
            )
            parts.append(f"会話履歴:\n{history_text}")

        parts.append(f"質問: {question}\n\n上記のコードベースに基づいて回答してください。")

        prompt = "\n\n".join(parts)
        return self._guard_budget(prompt, "qa")

    # ------------------------------------------------------------------
    # Resumeモード用コンテキスト
    # ------------------------------------------------------------------

    def build_resume_continue_prompt(
        self,
        target_file: str,
        target_description: str,
        context_md: str,
        plan_json: str,
        completed_contents: List[tuple[str, str]],
    ) -> str:
        """
        再開時のファイル生成プロンプトを組み立てる。

        Args:
            target_file: 生成対象ファイル
            target_description: ファイルの説明
            context_md: context.mdの内容
            plan_json: プランのJSON文字列
            completed_contents: 完了済みファイルの内容リスト

        Returns:
            組み立てたプロンプト文字列
        """
        return self.build_file_generation_prompt(
            target_file=target_file,
            target_description=target_description,
            context_md=context_md,
            plan_json=plan_json,
            dependency_contents=completed_contents,
        )

    def build_foreign_resume_qa_prompt(
        self,
        question: str,
        project_index_json: str,
        top_summaries: List[tuple[str, str]],
        conversation_history: List[Message],
    ) -> str:
        """
        外部プロジェクト再開時のQ&AプロンプトをQ&Aプロンプトに委譲して組み立てる。

        Args:
            question: ユーザーの質問
            project_index_json: ProjectIndexのJSON文字列
            top_summaries: 上位5件のファイルサマリー
            conversation_history: 会話履歴

        Returns:
            組み立てたプロンプト文字列
        """
        return self.build_qa_prompt(
            question=question,
            project_index_json=project_index_json,
            top_summaries=top_summaries,
            full_contents=[],
            conversation_history=conversation_history,
        )
