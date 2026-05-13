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
# シンボル（tree-sitter AST 抽出）
# ---------------------------------------------------------------------------

class Symbol(BaseModel):
    """ファイルから抽出されたコードシンボル（関数・クラス・インポートなど）。"""
    kind: Literal["function", "method", "class", "import"]
    name: str
    signature: str = ""
    docstring: str = ""
    line_start: int = 0
    line_end: int = 0
    parent: Optional[str] = None  # メソッドの場合はクラス名


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
    action: Literal["create", "modify"] = "create"
    modification_notes: Optional[str] = None


class GenerationPlan(BaseModel):
    """AI生成によるプロジェクト構成プランのモデル。"""
    project_name: str
    description: str
    files: List[PlannedFile]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    approved: bool = False


# ---------------------------------------------------------------------------
# ワークスペース（複数プロジェクト管理）
# ---------------------------------------------------------------------------

class WorkspaceEntry(BaseModel):
    """ワークスペースに属するプロジェクトのエントリ。"""
    root: str               # 絶対パス
    name: str               # フォルダ名
    auto: bool = False      # True = .localforgeスキャンで自動検出


class WorkspaceState(BaseModel):
    """ワークスペースの状態（.localforge/workspace.json に保存）。"""
    manual_entries: List["WorkspaceEntry"] = Field(default_factory=list)


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
    symbols: List["Symbol"] = Field(default_factory=list)
    imports_resolved: List[str] = Field(default_factory=list)


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
    token_limit: int = 12000
    # Ollama が使用する CPU スレッド数（None = Ollama デフォルト）
    num_thread: Optional[int] = None
    # False にするとインデックス時の RAG 埋め込みフェーズをスキップする（CPU 専用機向け）
    enable_rag: bool = True
    # LLM プロバイダー: "ollama" または "huggingface"
    llm_provider: str = "ollama"
    # HuggingFace GGUF モデルファイルの絶対パス
    hf_model_path: str = ""
    # コンテキストピン留めされたパス（プロジェクト相対、ファイルまたはフォルダ）
    context_pinned: List[str] = Field(default_factory=list)
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
