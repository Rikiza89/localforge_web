"""
プロジェクトサービス — プロジェクト管理・モード判定・状態管理の責務を担う。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from localforge.domain.models import (
    FileNode,
    GenerationLogEntry,
    GenerationPlan,
    Project,
    ProjectConfig,
    ProjectMode,
    ResumeState,
)
from localforge.infrastructure.filesystem_adapter import FileSystemAdapter
from localforge.infrastructure.git_adapter import GitAdapter
from localforge.infrastructure.index_adapter import IndexAdapter

logger = logging.getLogger(__name__)

# .localforgeディレクトリ名
_LOCALFORGE_DIR = ".localforge"
_CONFIG_FILE = "config.json"
_CONTEXT_FILE = "context.md"
_INDEX_JSONL = "index.jsonl"
_GENERATION_LOG = "generation_log.jsonl"
_PROJECT_INDEX = "project_index.json"
_APP_LOG = "app.log"


class ProjectService:
    """
    プロジェクトの開閉・モード判定・設定管理を担うサービスクラス。
    """

    def __init__(
        self,
        fs: FileSystemAdapter,
        git: GitAdapter,
        index: IndexAdapter,
    ) -> None:
        """
        ProjectServiceを初期化する。

        Args:
            fs: ファイルシステムアダプター
            git: gitアダプター
            index: インデックスアダプター
        """
        self._fs = fs
        self._git = git
        self._index = index
        self._current_project: Optional[Project] = None

    @property
    def current_project(self) -> Optional[Project]:
        """現在アクティブなプロジェクトを返す。"""
        return self._current_project

    def open_project(self, root: Path) -> Project:
        """
        プロジェクトフォルダを開いてモードを判定する。

        Args:
            root: プロジェクトのルートディレクトリ

        Returns:
            開いたProjectオブジェクト
        """
        mode = self.detect_project_mode(root)
        config = self._load_or_create_config(root, mode)
        file_tree = self._fs.build_file_tree(root)

        resume_state: Optional[ResumeState] = None
        if mode == ProjectMode.RESUME:
            resume_state = self._build_resume_state(root)

        project = Project(
            root=root,
            mode=mode,
            config=config,
            file_tree=file_tree,
            resume_state=resume_state,
        )
        self._current_project = project
        logger.info("プロジェクト開始: %s (mode=%s)", root, mode.value)
        return project

    def detect_project_mode(self, root: Path) -> ProjectMode:
        """
        フォルダ構成を分析してプロジェクトモードを判定する。
        判定順序:
          1. .localforge/config.json + 不完全なgeneration_log → RESUME (LocalForge)
          2. .localforge/index.jsonl + コードファイル → RESUME (外部プロジェクト)
          3. コードファイルあり、.localforgeなし → EXPLAIN
          4. それ以外 → GENERATE

        Args:
            root: 判定するディレクトリ

        Returns:
            判定されたProjectMode
        """
        lf_dir = root / _LOCALFORGE_DIR
        config_path = lf_dir / _CONFIG_FILE
        gen_log_path = lf_dir / _GENERATION_LOG
        index_path = lf_dir / _INDEX_JSONL

        # 1. LocalForgeプロジェクトで未完了エントリあり → RESUME
        if config_path.exists():
            log_entries = self._index.load_log_entries(gen_log_path)
            incomplete = [e for e in log_entries if e.status == "pending"]
            if incomplete:
                logger.debug("RESUME判定: 未完了エントリ %d件", len(incomplete))
                return ProjectMode.RESUME

        # 2. index.jsonl が存在してコードファイルあり → RESUME
        if index_path.exists() and self._fs.has_code_files(root):
            logger.debug("RESUME判定: 既存インデックスあり")
            return ProjectMode.RESUME

        # 3. コードファイルあり、.localforgeなし → EXPLAIN
        if not lf_dir.exists() and self._fs.has_code_files(root):
            logger.debug("EXPLAIN判定: コードファイルあり")
            return ProjectMode.EXPLAIN

        # コードファイルがあっても.localforgeがない場合もEXPLAINに
        if self._fs.has_code_files(root) and not (lf_dir / _INDEX_JSONL).exists():
            logger.debug("EXPLAIN判定: インデックス未作成")
            return ProjectMode.EXPLAIN

        # 4. デフォルト → GENERATE
        logger.debug("GENERATE判定: デフォルト")
        return ProjectMode.GENERATE

    def _load_or_create_config(
        self, root: Path, mode: ProjectMode
    ) -> ProjectConfig:
        """
        config.jsonを読み込む。存在しない場合は新規作成する。

        Args:
            root: プロジェクトルート
            mode: 検出したプロジェクトモード

        Returns:
            ProjectConfig
        """
        config_path = root / _LOCALFORGE_DIR / _CONFIG_FILE
        if config_path.exists():
            try:
                raw = config_path.read_text(encoding="utf-8")
                return ProjectConfig.model_validate_json(raw)
            except (ValueError, json.JSONDecodeError) as exc:
                logger.warning("config.jsonパースエラー（デフォルト使用）: %s", exc)

        config = ProjectConfig(
            project_name=root.name,
            mode=mode,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        self.save_config(root, config)
        return config

    def save_config(self, root: Path, config: ProjectConfig) -> None:
        """
        config.jsonを保存する。

        Args:
            root: プロジェクトルート
            config: 保存するProjectConfig
        """
        config_path = root / _LOCALFORGE_DIR / _CONFIG_FILE
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config.updated_at = datetime.utcnow()
        config_path.write_text(config.model_dump_json(indent=2), encoding="utf-8")

    def get_context_md(self, root: Path) -> str:
        """
        context.mdの内容を返す。存在しない場合は空文字列を返す。

        Args:
            root: プロジェクトルート

        Returns:
            context.mdの内容
        """
        context_path = root / _LOCALFORGE_DIR / _CONTEXT_FILE
        if context_path.exists():
            return context_path.read_text(encoding="utf-8")
        return ""

    def save_context_md(self, root: Path, content: str) -> None:
        """
        context.mdを保存する。

        Args:
            root: プロジェクトルート
            content: 保存する内容
        """
        context_path = root / _LOCALFORGE_DIR / _CONTEXT_FILE
        context_path.parent.mkdir(parents=True, exist_ok=True)
        context_path.write_text(content, encoding="utf-8")

    def save_generation_plan(self, root: Path, plan: GenerationPlan) -> None:
        """
        GenerationPlanを.localforge/plan.jsonとして保存する。

        Args:
            root: プロジェクトルート
            plan: 保存するプラン
        """
        plan_path = root / _LOCALFORGE_DIR / "plan.json"
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")

    def load_generation_plan(self, root: Path) -> Optional[GenerationPlan]:
        """
        .localforge/plan.jsonからGenerationPlanを読み込む。

        Args:
            root: プロジェクトルート

        Returns:
            GenerationPlan（存在しない場合はNone）
        """
        plan_path = root / _LOCALFORGE_DIR / "plan.json"
        if not plan_path.exists():
            return None
        try:
            return GenerationPlan.model_validate_json(
                plan_path.read_text(encoding="utf-8")
            )
        except (ValueError, json.JSONDecodeError) as exc:
            logger.error("プランパースエラー: %s", exc)
            return None

    def log_operation(
        self,
        root: Path,
        entry: GenerationLogEntry,
    ) -> None:
        """
        操作ログをgeneration_log.jsonlに記録する。

        Args:
            root: プロジェクトルート
            entry: 記録するログエントリ
        """
        log_path = root / _LOCALFORGE_DIR / _GENERATION_LOG
        self._index.append_log_entry(log_path, entry)

    def update_log_entry_status(
        self, root: Path, file_path: str, status: str
    ) -> None:
        """
        generation_log.jsonl内の特定ファイルのステータスを更新する。

        Args:
            root: プロジェクトルート
            file_path: 対象ファイルパス
            status: 新しいステータス（"completed" など）
        """
        log_path = root / _LOCALFORGE_DIR / _GENERATION_LOG
        entries = self._index.load_log_entries(log_path)
        updated = []
        for e in entries:
            if e.file_path == file_path and e.status == "pending":
                e.status = status
            updated.append(e)

        # 全エントリを書き直す
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as fh:
            for e in updated:
                fh.write(e.model_dump_json() + "\n")

    def get_file_tree(self, root: Path) -> List[FileNode]:
        """
        現在のファイルツリーを構築して返す。

        Args:
            root: プロジェクトルート

        Returns:
            FileNodeのリスト
        """
        return self._fs.build_file_tree(root)

    def get_project_status(self) -> Dict:
        """
        現在のプロジェクト状態を辞書形式で返す。

        Returns:
            ステータス辞書
        """
        if not self._current_project:
            return {"mode": None, "root": None, "model": None, "git_branch": None}

        p = self._current_project
        branch = self._git.get_current_branch(p.root)
        return {
            "mode": p.mode.value,
            "root": str(p.root),
            "model": p.config.model,
            "git_branch": branch or None,
        }

    def _build_resume_state(self, root: Path) -> ResumeState:
        """
        再開状態を構築する内部メソッド。

        Args:
            root: プロジェクトルート

        Returns:
            ResumeState
        """
        lf_dir = root / _LOCALFORGE_DIR
        is_localforge = (lf_dir / _CONFIG_FILE).exists()
        plan = self.load_generation_plan(root)

        completed: List[str] = []
        pending: List[str] = []

        if plan:
            log_entries = self._index.load_log_entries(
                lf_dir / _GENERATION_LOG
            )
            completed_paths = {
                e.file_path for e in log_entries if e.status == "completed"
            }
            for pf in plan.files:
                if pf.path in completed_paths:
                    completed.append(pf.path)
                else:
                    pending.append(pf.path)
        else:
            # 外部プロジェクト: すべてのコードファイルを完了済みとして扱う
            code_files = self._fs.list_code_files(root)
            completed = [str(f.relative_to(root)) for f in code_files]

        # 最終コミットメッセージを取得
        git_log = self._git.get_log(root, max_entries=1)
        last_commit = git_log[0]["message"] if git_log else None

        return ResumeState(
            project_root=str(root),
            mode=ProjectMode.RESUME,
            plan=plan,
            completed_files=completed,
            pending_files=pending,
            last_commit_message=last_commit,
            is_localforge_project=is_localforge,
        )

    def set_model(self, root: Path, model: str) -> None:
        """
        使用するモデルを変更してconfig.jsonに保存する。

        Args:
            root: プロジェクトルート
            model: 新しいモデル名
        """
        if self._current_project:
            self._current_project.config.model = model
            self.save_config(root, self._current_project.config)
