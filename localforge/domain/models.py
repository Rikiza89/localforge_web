"""
ドメインモデル定義 — LocalForgeアプリケーションで使用するすべてのデータモデルを定義する。
Pydanticを使用して型安全性とバリデーションを確保する。
"""

from __future__ import annotations

import enum
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 列挙型
# ---------------------------------------------------------------------------

class ProjectMode(str, enum.Enum):
    """プロジェクトモードの列挙型。"""
    GENERATE = "generate"
    RESUME = "resume"
    EXPLAIN = "explain"


class FileStatus(str, enum.Enum):
    """ファイル生成状態の列挙型。"""
    PENDING = "pending"
    GENERATED = "generated"
    MODIFIED = "modified"
    INDEXED = "indexed"
    UNINDEXED = "unindexed"


class ChunkStrategy(str, enum.Enum):
    """ファイル読み込み戦略の列挙型。"""
    FULL = "full"
    HYBRID = "hybrid"


# ---------------------------------------------------------------------------
# ファイルノード（ファイルツリー表示用）
# ---------------------------------------------------------------------------

class FileNode(BaseModel):
    """ファイルツリーの1ノードを表すモデル。"""
    name: str
    path: str
    is_dir: bool
    status: FileStatus = FileStatus.UNINDEXED
    children: List["FileNode"] = Field(default_factory=list)
    size: Optional[int] = None
    modified_at: Optional[float] = None

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# 生成プラン
# ---------------------------------------------------------------------------

class PlannedFile(BaseModel):
    """生成プランに含まれる個別ファイルのモデル。"""
    path: str
    description: str
    dependencies: List[str] = Field(default_factory=list)


class GenerationPlan(BaseModel):
    """AI生成によるプロジェクト構成プランのモデル。"""
    project_name: str
    description: str
    files: List[PlannedFile]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    approved: bool = False


# ---------------------------------------------------------------------------
# ファイルチャンク（インデックス用）
# ---------------------------------------------------------------------------

class FileChunk(BaseModel):
    """インデックス構築に使用するファイルコンテンツチャンクのモデル。"""
    path: str
    content: str
    strategy: ChunkStrategy
    size: int
    mtime: float
    summary: Optional[str] = None
    language: Optional[str] = None
    indexed_at: Optional[datetime] = None


# ---------------------------------------------------------------------------
# ProjectIndex（マスタードキュメント）
# ---------------------------------------------------------------------------

class ProjectIndex(BaseModel):
    """プロジェクト全体のインデックスを表すマスタードキュメント。"""
    project_root: str
    project_name: str
    summary: str
    file_chunks: List[FileChunk] = Field(default_factory=list)
    total_files: int = 0
    indexed_files: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# 説明レポート
# ---------------------------------------------------------------------------

class ReportSection(BaseModel):
    """説明レポートの1セクションを表すモデル。"""
    name: str
    content: str


class ExplanationReport(BaseModel):
    """コードベース分析によって生成される説明レポートのモデル。"""
    project_root: str
    sections: List[ReportSection] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    complete: bool = False


# ---------------------------------------------------------------------------
# チャット履歴（Q&A用）
# ---------------------------------------------------------------------------

class Message(BaseModel):
    """Q&Aチャットの1メッセージを表すモデル。"""
    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# 再開状態
# ---------------------------------------------------------------------------

class ResumeState(BaseModel):
    """プロジェクト再開時の状態を表すモデル。"""
    project_root: str
    mode: ProjectMode
    plan: Optional[GenerationPlan] = None
    completed_files: List[str] = Field(default_factory=list)
    pending_files: List[str] = Field(default_factory=list)
    last_commit_message: Optional[str] = None
    is_localforge_project: bool = False


# ---------------------------------------------------------------------------
# プロジェクト設定（config.json）
# ---------------------------------------------------------------------------

class ProjectConfig(BaseModel):
    """プロジェクトごとの設定ファイル（.localforge/config.json）のモデル。"""
    project_name: str = ""
    mode: ProjectMode = ProjectMode.GENERATE
    model: str = ""
    token_limit: int = 6000
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# 生成ログエントリ
# ---------------------------------------------------------------------------

class GenerationLogEntry(BaseModel):
    """生成ログ（generation_log.jsonl）の1エントリを表すモデル。"""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    mode: str
    model: str
    operation: str
    prompt_tokens_estimated: int = 0
    response_time_ms: Optional[float] = None
    file_path: Optional[str] = None
    status: str = "pending"


# ---------------------------------------------------------------------------
# プロジェクト全体状態
# ---------------------------------------------------------------------------

class Project(BaseModel):
    """アクティブなプロジェクトの全体状態を表すモデル。"""
    root: Path
    mode: ProjectMode
    config: ProjectConfig = Field(default_factory=ProjectConfig)
    file_tree: List[FileNode] = Field(default_factory=list)
    resume_state: Optional[ResumeState] = None
    index: Optional[ProjectIndex] = None

    model_config = {"arbitrary_types_allowed": True}
