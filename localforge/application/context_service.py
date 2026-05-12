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
# デフォルトトークン上限（現代的なローカルモデルの文脈長に合わせて引き上げ）
_DEFAULT_TOKEN_LIMIT = 12000


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
        model_name: str = "",
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

        # モデル特有の指示の追加
        if "deepseek-r1" in model_name.lower():
            parts.append("思考プロセス（reasoning）を詳しく記述し、最終的な結論を明確にしてください。")
        elif "llama3" in model_name.lower():
            parts.append("簡潔かつ正確な回答を心がけてください。")

        parts.append(
            f"\nユーザーの要求:\n{user_prompt}\n\n"
            "まず、このプランで何をするかを2〜3文のMarkdown形式で簡潔に説明してください"
            "（新規作成ファイル数・修正ファイル数・主な変更点を含む）。\n"
            "その際、プロジェクト全体の一貫性や、技術的な選定理由があればそれも含めてください。\n"
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
        return self._guard_budget(prompt, "generate_plan"), _estimate_tokens(prompt)

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
        return self._guard_budget(prompt, f"generate_file:{target_file}"), _estimate_tokens(prompt)

    def max_chunk_chars(self) -> int:
        """
        ファイルチャンク1つに割り当て可能な最大文字数を返す。
        トークン上限からプロンプトオーバーヘッド分を差し引き、
        4文字≒1トークンで換算する。最小値は3000文字。
        """
        overhead_tokens = 600  # 指示文・メタデータ用に予約
        return max(3000, (self._token_limit - overhead_tokens) * 4)

    def max_qa_file_chars(self) -> int:
        """
        Q&A プロンプト内でファイル1件に割り当て可能な最大文字数を返す。
        トークン上限の半分をファイル内容に充て、4文字≒1トークンで換算する。
        最小値は 3000 文字（6000 トークン上限時の従来値と等価）。
        """
        return max(3000, (self._token_limit // 2) * 4)

    def build_file_diff_prompt(
        self,
        target_file: str,
        modification_notes: str,
        chunk_content: str,
        context_md: str,
        chunk_idx: int = 0,
        total_chunks: int = 1,
    ) -> str:
        """
        ファイル編集差分（SEARCH/REPLACE形式）生成のプロンプトを組み立てる。
        ファイルが大きい場合はチャンク単位で呼び出し、全チャンクの出力を合成して適用する。

        Args:
            target_file: 編集対象ファイルの相対パス
            modification_notes: 変更要求の具体的な説明
            chunk_content: 今回分析するコードチャンク
            context_md: context.mdの内容
            chunk_idx: 現在のチャンクインデックス（0始まり）
            total_chunks: 全チャンク数

        Returns:
            組み立てたプロンプト文字列
        """
        parts = [
            f"ファイル: {target_file}",
            f"変更要求: {modification_notes}",
        ]

        if total_chunks > 1:
            parts.append(
                f"[注意: このファイルは {total_chunks} チャンクに分割されています。"
                f" 現在はチャンク {chunk_idx + 1}/{total_chunks} を分析中です。"
                f" このチャンク内の変更のみを出力してください。]"
            )

        if context_md.strip():
            parts.append(f"プロジェクトコンテキスト:\n{context_md}")

        parts.append(f"現在のコード:\n```\n{chunk_content}\n```")

        parts.append(
            "変更要求を実現するために必要な変更を SEARCH/REPLACE ブロックで出力してください。\n"
            "変更不要な場合は「変更なし」とだけ出力してください。\n\n"
            "出力形式:\n"
            "<<<<<<< SEARCH\n"
            "<変更・削除する既存コードの完全一致テキスト（空白・インデント含む）>\n"
            "=======\n"
            "<置き換える新しいコード（削除の場合は空行のみ）>\n"
            ">>>>>>> REPLACE\n\n"
            "例:\n"
            "<<<<<<< SEARCH\n"
            "def old_func():\n"
            "    print(\"Hello\")\n"
            "=======\n"
            "def new_func():\n"
            "    print(\"Hello World\")\n"
            ">>>>>>> REPLACE\n\n"
            "ルール:\n"
            "- SEARCH ブロックには、既存のコードを **一字一句、空白やインデントも含めて正確に** 記述してください。\n"
            "- SEARCH ブロックは、変更箇所を特定するのに十分な長さ（数行程度）を確保してください。一意に特定できない短い断片は避けてください。\n"
            "- 追加の場合は、挿入箇所の前後の既存コードを SEARCH に含め、REPLACE でそれらを含めた形で新しいコードを記述してください。\n"
            "- 複数の箇所を変更する場合は、複数の SEARCH/REPLACE ブロックを作成してください。\n"
            "- 解説やコメントは含めず、SEARCH/REPLACE ブロックのみを出力してください。\n"
            "- ファイル全体は絶対に出力しないでください。変更が必要な箇所のみを抽出してください。"
        )

        prompt = "\n\n".join(parts)
        return self._guard_budget(
            prompt, f"diff_file:{target_file}[{chunk_idx + 1}/{total_chunks}]"
        ), _estimate_tokens(prompt)

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
        content_limit: int = 800,
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
    ) -> tuple[str, int]:
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
        return self._guard_budget(prompt, f"report_section:{section_name}"), _estimate_tokens(prompt)

    def build_qa_file_selection_prompt(
        self,
        question: str,
        all_summaries: List[tuple[str, str]],
    ) -> str:
        """
        Q&A フェーズ 1: 質問に答えるために必要なファイルを LLM に選ばせるプロンプト。
        LLM はファイルパスの JSON 配列のみを返す。

        Args:
            question: ユーザーの質問
            all_summaries: 全ファイルの (パス, サマリー) リスト

        Returns:
            プロンプト文字列
        """
        # 長すぎる場合は先頭 200 件に絞る（プロンプト爆発防止）
        shown = all_summaries[:200]
        file_list = "\n".join(f"- {p}: {s}" for p, s in shown)
        prompt = (
            "以下のファイル一覧から、この質問に答えるために読む必要があるファイルを選んでください。\n"
            "JSONの配列形式でファイルパスのみを出力してください。例: [\"src/foo.py\", \"lib/bar.py\"]\n"
            "最大10件まで選択できます。不要なファイルは含めないでください。\n\n"
            f"質問: {question}\n\n"
            f"ファイル一覧:\n{file_list}\n\n"
            "選択したファイルのパスをJSONの配列として出力してください:"
        )
        return self._guard_budget(prompt, "qa_file_selection")

    def build_qa_prompt(
        self,
        question: str,
        project_index_json: str,
        top_summaries: List[tuple[str, str]],
        full_contents: List[tuple[str, str]],
        conversation_history: List[Message],
    ) -> tuple[str, int]:
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
        return self._guard_budget(prompt, "qa"), _estimate_tokens(prompt)

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
