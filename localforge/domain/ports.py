"""
ポートインターフェース定義 — クリーンアーキテクチャのポート層。
インフラストラクチャ層の実装が準拠すべきProtocolクラスを定義する。
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator, Iterable, List, Optional, Protocol, Tuple

from localforge.domain.models import FileChunk, FileNode, GenerationLogEntry, ProjectIndex


class LLMPort(Protocol):
    """LLMバックエンドとの通信を抽象化するポートインターフェース。"""

    def stream_completion(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """
        LLMへのプロンプトを送信し、テキストチャンクをストリーミングで生成する。

        Args:
            model: 使用するOllamaモデル名
            prompt: ユーザープロンプト
            system: システムプロンプト（省略可能）

        Yields:
            テキストチャンク（文字列）
        """
        ...

    def list_models(self) -> List[str]:
        """
        Ollamaで利用可能なモデルの一覧を返す。

        Returns:
            モデル名のリスト
        """
        ...

    def is_available(self) -> bool:
        """
        Ollamaサーバーが起動していてアクセス可能かどうかを確認する。

        Returns:
            接続可能であればTrue
        """
        ...


class FileSystemPort(Protocol):
    """ファイルシステム操作を抽象化するポートインターフェース。"""

    def read_text(self, path: Path) -> str:
        """
        テキストファイルを読み込む。

        Args:
            path: ファイルの絶対パス

        Returns:
            ファイルの内容（文字列）
        """
        ...

    def write_text(self, path: Path, content: str) -> None:
        """
        テキストファイルを書き込む（親ディレクトリは自動作成）。

        Args:
            path: ファイルの絶対パス
            content: 書き込むテキスト内容
        """
        ...

    def list_files(
        self,
        root: Path,
        extensions: Optional[Iterable[str]] = None,
        ignore_dirs: Optional[Iterable[str]] = None,
    ) -> List[Path]:
        """
        ディレクトリ以下のファイル一覧を返す。

        Args:
            root: 検索対象のルートディレクトリ
            extensions: フィルタする拡張子のリスト（省略時はすべて）
            ignore_dirs: 除外するディレクトリ名のリスト

        Returns:
            ファイルパスのリスト
        """
        ...

    def build_file_tree(self, root: Path) -> List[FileNode]:
        """
        ディレクトリのファイルツリーを構築する。

        Args:
            root: ツリーのルートディレクトリ

        Returns:
            FileNodeのリスト（階層構造）
        """
        ...

    def get_mtime_size(self, path: Path) -> Tuple[float, int]:
        """
        ファイルの最終更新時刻とサイズを返す。

        Args:
            path: ファイルの絶対パス

        Returns:
            (mtime, size) のタプル
        """
        ...

    def exists(self, path: Path) -> bool:
        """
        パスが存在するかどうかを確認する。

        Args:
            path: 確認するパス

        Returns:
            存在すればTrue
        """
        ...


class GitPort(Protocol):
    """git操作を抽象化するポートインターフェース。"""

    def init(self, path: Path) -> None:
        """
        指定パスでgit initを実行する。

        Args:
            path: 初期化するディレクトリ
        """
        ...

    def commit_all(self, path: Path, message: str) -> str:
        """
        すべての変更をステージングしてコミットする。

        Args:
            path: リポジトリのルートディレクトリ
            message: コミットメッセージ

        Returns:
            コミットハッシュ
        """
        ...

    def get_log(self, path: Path, max_entries: int = 20) -> List[dict]:
        """
        gitコミットログを返す。

        Args:
            path: リポジトリのルートディレクトリ
            max_entries: 最大取得件数

        Returns:
            コミット情報のリスト
        """
        ...

    def get_diff(self, path: Path) -> str:
        """
        git diff HEADの結果を返す。

        Args:
            path: リポジトリのルートディレクトリ

        Returns:
            diff出力テキスト
        """
        ...

    def get_status(self, path: Path) -> str:
        """
        git status --shortの結果を返す。

        Args:
            path: リポジトリのルートディレクトリ

        Returns:
            ステータス出力テキスト
        """
        ...

    def get_current_branch(self, path: Path) -> str:
        """
        現在のgitブランチ名を返す。

        Args:
            path: リポジトリのルートディレクトリ

        Returns:
            ブランチ名
        """
        ...


class VectorIndexPort(Protocol):
    """ベクトルインデックスの永続化・検索を抽象化するポートインターフェース。"""

    def init_collection(self, project_root: Path) -> None:
        """プロジェクトルートに対応するコレクションを初期化する。"""
        ...

    def is_initialized(self) -> bool:
        """コレクションが初期化済みかどうかを返す。"""
        ...

    def collection_exists(self, project_root: Path) -> bool:
        """ディスク上にコレクションが存在するかどうかを返す。"""
        ...

    def upsert_chunk(self, chunk: "FileChunk") -> bool:
        """FileChunkをベクトルインデックスに追加または更新する。"""
        ...

    def needs_reembedding(self, chunk: "FileChunk") -> bool:
        """チャンクの再埋め込みが必要かどうかを返す。"""
        ...

    def migrate_from_chunks(self, chunks: "List[FileChunk]") -> int:
        """既存チャンクリストからベクトルインデックスへ一括移行する。"""
        ...

    def get_top_chunks_semantic(
        self,
        all_chunks: "List[FileChunk]",
        query: str,
        top_n: int = 5,
    ) -> "List[FileChunk]":
        """クエリに意味的に近い上位N件のFileChunkを返す。"""
        ...


class IndexPort(Protocol):
    """ProjectIndexの永続化を抽象化するポートインターフェース。"""

    def save_chunks(self, path: Path, chunks: List[FileChunk]) -> None:
        """
        FileChunkのリストをJSONL形式で保存する。

        Args:
            path: 保存先ファイルパス（index.jsonl）
            chunks: 保存するFileChunkのリスト
        """
        ...

    def load_chunks(self, path: Path) -> List[FileChunk]:
        """
        JSONL形式のFileChunkを読み込む。

        Args:
            path: 読み込み元ファイルパス（index.jsonl）

        Returns:
            FileChunkのリスト
        """
        ...

    def save_index(self, path: Path, index: ProjectIndex) -> None:
        """
        ProjectIndexをJSONファイルとして保存する。

        Args:
            path: 保存先ファイルパス（project_index.json）
            index: 保存するProjectIndex
        """
        ...

    def load_index(self, path: Path) -> Optional[ProjectIndex]:
        """
        JSONファイルからProjectIndexを読み込む。

        Args:
            path: 読み込み元ファイルパス（project_index.json）

        Returns:
            ProjectIndex（ファイルが存在しない場合はNone）
        """
        ...

    def append_log_entry(self, path: Path, entry: GenerationLogEntry) -> None:
        """
        生成ログエントリをJSONL形式で追記する。

        Args:
            path: ログファイルパス（generation_log.jsonl）
            entry: 追記するログエントリ
        """
        ...

    def load_log_entries(self, path: Path) -> List[GenerationLogEntry]:
        """
        生成ログエントリをすべて読み込む。

        Args:
            path: ログファイルパス（generation_log.jsonl）

        Returns:
            GenerationLogEntryのリスト
        """
        ...
