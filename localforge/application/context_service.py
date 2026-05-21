"""
コンテキストサービス — LLMプロンプト組み立てとトークン予算管理の唯一の責務を担う。
すべてのLLMプロンプト構築はこのサービスを通じてのみ行われる。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from localforge.domain.exceptions import TokenBudgetExceededWarning
from localforge.domain.models import FileChunk, GenerationPlan, Message, ProjectIndex

logger = logging.getLogger(__name__)

# トークン推定係数（単語数 × この係数でトークン数を推定）
_WORDS_TO_TOKENS = 1.3
# デフォルトトークン上限（現代的なローカルモデルの文脈長に合わせて引き上げ）
_DEFAULT_TOKEN_LIMIT = 131072


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
            logger.debug(
                "トークン予算超過 [%s]: 推定=%d, 上限=%d", label, estimated, self._token_limit
            )
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
        project_index_json: Optional[str] = None,
        pinned_contents: Optional[List[tuple[str, str]]] = None,
        workspace_summaries: Optional[List[tuple[str, str]]] = None,
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
            project_index_json: ProjectIndexのJSON文字列（プロジェクト全体の概要、任意）

        Returns:
            組み立てたプロンプト文字列
        """
        parts = [
            f"プロジェクト名: {folder_name}",
        ]

        # プロジェクト全体の概要を最優先で注入（Resume/Generateの精度向上）
        if project_index_json:
            parts.append(f"プロジェクト全体の概要:\n{project_index_json}")

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

        # ピン留めファイル（ユーザーが選択した既存コード）
        if pinned_contents:
            pin_texts = "\n".join(f"--- {p} ---\n{c[:4000]}" for p, c in pinned_contents[:10])
            parts.append(f"[ピン留めされた既存コード — 必ず参照・統合すること]\n{pin_texts}")

        # ワークスペースプロジェクトのサマリー（外部モジュール情報）
        if workspace_summaries:
            ws_lines = "\n".join(f"- [{name}] {summary}" for name, summary in workspace_summaries[:5])
            parts.append(f"ワークスペース内の他プロジェクト（外部モジュールとして参照可能）:\n{ws_lines}")

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
            "このファイルを分析し、以下の情報を詳細に抽出してください：\n"
            "1. ファイルの全体的な役割と責務\n"
            "2. 定義されている主要なクラス、メソッド、関数のシグネチャと目的\n"
            "3. 外部モジュールや他ファイルとの依存関係（インポート/エクスポート）\n"
            "4. 特筆すべきアルゴリズム、データ構造、または状態管理のロジック\n"
            "これらをマークダウン形式の箇条書きで、詳細かつ具体的に記述してください。"
        )
        # prompt = (
        #     f"ファイル: {file_path} (拡張子: {extension})\n\n"
        #     f"{content}\n\n"
        #     "このファイルの役割、主要なクラス・関数・エクスポート、依存関係を3〜5文の日本語で要約してください。"
        #     " 要約のみを出力してください。"
        # )
        return self._guard_budget(prompt, f"file_summary:{file_path}")

    # セクション別の具体的な分析指示 — 各セクションで何を書くべきかをLLMに明示する
    _SECTION_GUIDANCE: dict = {
        "Project Overview": (
            "Cover: (1) what the project does and its primary goals, (2) intended users or audience, "
            "(3) key technologies and language stack, (4) overall architecture style "
            "(e.g. Clean Architecture, MVC, monolith, microservice), "
            "(5) any notable design philosophy or constraints. Aim for 300-500 words."
        ),
        "Module Map": (
            "Produce a Markdown table listing every major module/package with columns: "
            "Module | Layer (domain/application/infrastructure/interface) | Responsibility. "
            "Then add a short paragraph describing how the layers interact. "
            "Include sub-packages where meaningful."
        ),
        "Entry Points & Startup Flow": (
            "Describe: (1) the main entry point file and function, "
            "(2) the initialization sequence step-by-step (dependency injection, config loading, server start, etc.), "
            "(3) any background threads or processes launched at startup, "
            "(4) how the app handles shutdown/cleanup. Use a numbered list for the sequence."
        ),
        "Data Flow": (
            "Trace how data moves through the system: "
            "(1) primary input sources (HTTP request, file, user input), "
            "(2) processing pipeline (parsing, validation, transformation, business logic), "
            "(3) storage or output (DB write, file write, HTTP response). "
            "Give at least one concrete end-to-end example with actual function/method names."
        ),
        "Key Interfaces & Contracts": (
            "Document: (1) all Protocol/interface/abstract class definitions with their method signatures, "
            "(2) major Pydantic models or data schemas, "
            "(3) public API contracts (REST endpoints, event schemas). "
            "Use code blocks for signatures where helpful."
        ),
        "External Dependencies": (
            "Produce a Markdown table: Library | Version (if known) | Purpose | Risk/Notes. "
            "Group by category (LLM/AI, web framework, storage, utilities). "
            "Flag any dependencies with known security concerns, heavy resource use, or limited maintenance."
        ),
        "Configuration": (
            "List: (1) all configuration files and their formats, "
            "(2) all environment variables the app reads, "
            "(3) the config loading order and override mechanism, "
            "(4) required vs optional settings and their defaults. "
            "Use a table for env vars: Name | Default | Description."
        ),
        "Test Coverage": (
            "Describe: (1) testing strategy (unit, integration, e2e), "
            "(2) test framework(s) used, "
            "(3) rough coverage estimate if determinable from the codebase, "
            "(4) what is well-tested vs what is untested or difficult to test, "
            "(5) any mocking or fixture patterns used. "
            "Be specific about gaps."
        ),
        "Notable Patterns & Design Decisions": (
            "Identify: (1) architectural patterns used (Repository, Factory, Strategy, etc.) with examples, "
            "(2) specific design decisions that stand out and the likely reasoning behind them, "
            "(3) any trade-offs that are evident in the code. "
            "Reference actual file/class names to ground each point."
        ),
        "Potential Issues & Technical Debt": (
            "List concrete issues sorted by estimated severity (High/Medium/Low): "
            "for each item state the issue, where it lives (file/function), and a suggested fix direction. "
            "Include: error handling gaps, performance hotspots, security concerns, "
            "missing validation, TODO/FIXME comments, and coupling problems."
        ),
        "Project Health & Code Quality Analysis": (
            "Assess: (1) code readability and consistency, "
            "(2) documentation quality (docstrings, comments, README), "
            "(3) modularity and separation of concerns (1-5 score with justification), "
            "(4) test confidence level, "
            "(5) overall maintainability rating with a brief rationale. "
            "Be candid — include both strengths and weaknesses."
        ),
        "How to Extend This Project": (
            "Provide a practical guide for the three most common extension scenarios "
            "(e.g. adding a new API endpoint, adding a new LLM operation, adding a new storage adapter). "
            "For each: list the exact files to create/modify, describe the minimal changes needed, "
            "and note any pitfalls or conventions to follow. Use numbered steps."
        ),
    }

    def build_report_section_prompt(
        self,
        section_name: str,
        project_index_json: str,
        relevant_summaries: List[tuple[str, str]],
        language: str = "ja",
    ) -> tuple[str, int]:
        """
        レポートセクション生成のプロンプトを組み立てる。
        セクション別の具体的な指示を注入してレポートの深度と一貫性を高める。

        Args:
            section_name: セクション名
            project_index_json: ProjectIndexのJSON文字列
            relevant_summaries: [(ファイルパス, サマリー)] のリスト（各最大400文字）
            language: 出力言語 ("ja" = 日本語, "en" = 英語)

        Returns:
            (プロンプト文字列, 推定トークン数)
        """
        summaries_text = "\n".join(
            f"- {path}:\n  {summary}" for path, summary in relevant_summaries
        )

        section_guidance = self._SECTION_GUIDANCE.get(
            section_name,
            "Provide a detailed, structured analysis of this section using markdown formatting."
        )

        if language == "en":
            lang_instruction = "Write the report section in English."
        else:
            lang_instruction = "レポートセクションを日本語で記述してください。"

        prompt = (
            f"You are analyzing a software project. {lang_instruction}\n\n"
            f"Project metadata:\n{project_index_json}\n\n"
            f"Most relevant file summaries:\n{summaries_text}\n\n"
            f"Section to write: **{section_name}**\n\n"
            f"Specific guidance for this section:\n{section_guidance}\n\n"
            f"Output rules:\n"
            f"- Do NOT output the section title/heading — content only.\n"
            f"- Use Markdown (### subheadings, tables, code blocks, bullet lists as appropriate).\n"
            f"- Be specific: reference actual file names, class names, and function names from the summaries.\n"
            f"- Output only the content for this section, nothing else."
        )
        return self._guard_budget(prompt, f"report_section:{section_name}"), _estimate_tokens(prompt)

    def build_qa_file_selection_prompt(
        self,
        question: str,
        all_summaries: List[tuple[str, str]],
        dep_hints: Optional[Dict[str, List[str]]] = None,
    ) -> str:
        """
        Q&A フェーズ 1: 質問に答えるために必要なファイルを LLM に選ばせるプロンプト。
        LLM はファイルパスの JSON 配列のみを返す。
        dep_hints が指定された場合、各ファイルのインポート先を併記して選択精度を高める。

        Args:
            question: ユーザーの質問
            all_summaries: 全ファイルの (パス, サマリー) リスト
            dep_hints: ファイルパス → インポート先パスのリスト（省略可）

        Returns:
            プロンプト文字列
        """
        shown = all_summaries[:200]
        lines: List[str] = []
        for p, s in shown:
            line = f"- {p}: {s}"
            if dep_hints and p in dep_hints:
                deps_str = ", ".join(dep_hints[p][:5])
                line += f" [imports: {deps_str}]"
            lines.append(line)
        file_list = "\n".join(lines)
        prompt = (
            "以下のファイル一覧から、この質問に答えるために読む必要があるファイルを選んでください。\n"
            "JSONの配列形式でファイルパスのみを出力してください。例: [\"src/foo.py\", \"lib/bar.py\"]\n"
            "最大10件まで選択できます。[imports: ...] はそのファイルが依存するプロジェクト内ファイルを示します。\n\n"
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
        dep_map: Optional[Dict[str, List[str]]] = None,
        pinned_contents: Optional[List[tuple[str, str]]] = None,
        workspace_projects: Optional[List[Dict[str, object]]] = None,
    ) -> tuple[str, int]:
        """
        Q&A回答のプロンプトを組み立てる。

        Args:
            question: ユーザーの質問
            project_index_json: ProjectIndexのJSON文字列
            top_summaries: 上位ファイルのサマリー
            full_contents: ファイルの全内容
            conversation_history: 最近10件の会話履歴
            dep_map: ファイルパス → インポート先パスのリスト（依存関係マップ）

        Returns:
            (プロンプト文字列, 推定トークン数)
        """
        # 固定コスト部分: 質問・概要・依存関係マップ・指示文
        instructions = (
            f"質問: {question}\n\n"
            "あなたはシニアソフトウェアアーキテクトとして、上記のコードベースに基づいて極めて詳細かつ網羅的に回答してください。\n"
            "以下のルールに厳密に従ってください：\n"
            "1. 抽象的な説明を避け、具体的なクラス名、関数名、変数名、データ構造を明記すること。\n"
            "2. 情報を引用する際は、必ず正確なファイルパス（例: `src/main.py`）を併記すること。上記の依存関係マップに記載されたパスのみを使用し、推測しないこと。\n"
            "3. 対象コンポーネントが他のどのファイル・モジュールに依存しているか、また何から依存されているか（依存関係）を依存関係マップに基づいて説明すること。\n"
            "4. ロジックのフローや実行順序が存在する場合は、ステップ・バイ・ステップで分解して解説すること。\n"
            "5. 提供されたコンテキスト内に情報が不足している場合は、推測で補わず「〇〇の詳細はコンテキストにありません」と正直に述べること。\n"
        )

        parts = [f"プロジェクト概要:\n{project_index_json}"]

        # 依存関係マップ — ファイル間の実際のインポート関係をLLMに提示する
        if dep_map:
            imported_by: Dict[str, List[str]] = {}
            for path, deps in dep_map.items():
                for dep in deps:
                    if dep not in imported_by:
                        imported_by[dep] = []
                    if path not in imported_by[dep]:
                        imported_by[dep].append(path)

            dep_lines: List[str] = []
            all_dep_paths = set(dep_map.keys()) | set(imported_by.keys())
            for path in sorted(all_dep_paths):
                fwd = dep_map.get(path, [])
                rev = imported_by.get(path, [])
                parts_line: List[str] = []
                if fwd:
                    parts_line.append(f"imports → {', '.join(fwd)}")
                if rev:
                    parts_line.append(f"imported by ← {', '.join(rev)}")
                if parts_line:
                    dep_lines.append(f"  {path}: {' | '.join(parts_line)}")

            if dep_lines:
                parts.append(
                    "ファイル依存関係マップ（実際のimport文から解決済み）:\n"
                    + "\n".join(dep_lines)
                )

        summaries_text = "\n".join(f"- {p}: {s}" for p, s in top_summaries)
        if summaries_text:
            parts.append(f"関連ファイルサマリー:\n{summaries_text}")

        # ワークスペース内の他プロジェクトのコンテキスト
        if workspace_projects:
            for wp in workspace_projects[:3]:
                wp_name = wp.get("name", "")
                wp_summaries = wp.get("summaries", [])
                wp_contents = wp.get("contents", [])
                if wp_summaries or wp_contents:
                    header = f"[ワークスペース: {wp_name}]"
                    lines: List[str] = [header]
                    for p, s in (wp_summaries or [])[:5]:
                        lines.append(f"  - {p}: {s}")
                    for p, c in (wp_contents or [])[:3]:
                        lines.append(f"  --- {p} ---\n  {c[:2000]}")
                    parts.append("\n".join(lines))

        # 会話履歴 — O(n) トリミング。各メッセージのトークン数を事前計算し差分更新する。
        _instr_tokens = _estimate_tokens(instructions)
        _parts_tokens = _estimate_tokens("\n\n".join(parts))
        if conversation_history:
            history_msgs = list(conversation_history[-10:])
            _max_history_tokens = max((self._token_limit - _parts_tokens - _instr_tokens - 4096) // 4, 500)
            _msg_tokens = [
                _estimate_tokens(f"{'ユーザー' if m.role == 'user' else 'アシスタント'}: {m.content}")
                for m in history_msgs
            ]
            _total_history = sum(_msg_tokens)
            while history_msgs and _total_history > _max_history_tokens:
                _total_history -= _msg_tokens.pop(0)
                history_msgs.pop(0)
            if history_msgs:
                history_text = "\n".join(
                    f"{'ユーザー' if m.role == 'user' else 'アシスタント'}: {m.content}"
                    for m in history_msgs
                )
                parts.append(f"会話履歴:\n{history_text}")
                _parts_tokens += _estimate_tokens(f"会話履歴:\n{history_text}") + 2

        # 残りの予算を計算してファイルコンテンツを注入（ピン留め優先）
        _fixed_cost = _parts_tokens + _instr_tokens + 200
        available = max(self._token_limit - _fixed_cost, 0)

        # ピン留めファイルを優先して注入（予算内）
        if pinned_contents:
            pinned_added: List[tuple[str, str]] = []
            for p, c in pinned_contents:
                fc_text = f"--- {p} ---\n{c}"
                cost = _estimate_tokens(fc_text)
                if cost <= available:
                    pinned_added.append((p, c))
                    available -= cost
                else:
                    # ファイルが大きすぎる場合は先頭部分のみ注入して警告
                    _chars_budget = max(available * 3, 500)
                    pinned_added.append((p, c[:_chars_budget]))
                    available = 0
                    logger.warning(
                        "ピン留めファイル [%s] がトークン予算を超過。先頭 %d 文字のみ注入。",
                        p, _chars_budget,
                    )
                    break
            if pinned_added:
                pin_texts = "\n".join(f"--- {p} ---\n{c}" for p, c in pinned_added)
                parts.append(f"[ピン留めコンテキスト — ユーザーが選択したファイル]\n{pin_texts}")

        # 残予算でその他ファイルを注入
        for fc_path, fc_content in full_contents:
            if available <= 0:
                break
            fc_text = f"--- {fc_path} ---\n{fc_content}"
            cost = _estimate_tokens(fc_text)
            if cost <= available:
                parts.append(f"ファイル内容:\n{fc_text}")
                available -= cost
            else:
                # 残予算に収まる分だけ先頭を注入
                _chars = max(available * 3, 200)
                parts.append(f"ファイル内容:\n--- {fc_path} ---\n{fc_content[:_chars]}\n...[予算超過により切り詰め]")
                available = 0
                break

        parts.append(instructions)

        prompt = "\n\n".join(parts)
        estimated = _estimate_tokens(prompt)
        if estimated > self._token_limit:
            logger.warning(
                "Q&Aプロンプトがトークン予算を超過: 推定=%d, 上限=%d", estimated, self._token_limit
            )
        return self._guard_budget(prompt, "qa"), estimated

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
