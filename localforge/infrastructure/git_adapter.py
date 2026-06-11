"""
gitアダプター — GitPython + subprocess fallbackを使用したgit操作ラッパー。
GitPortインターフェースを実装する。
"""

from __future__ import annotations

import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from localforge.domain.exceptions import GitOperationError

logger = logging.getLogger(__name__)


class GitAdapter:
    """
    git操作を提供するアダプタークラス。
    GitPythonが利用可能な場合はそちらを優先し、
    不可能な場合はsubprocessにフォールバックする。
    """

    def _run(self, args: List[str], cwd: Path) -> str:
        """
        gitコマンドをsubprocessで実行する内部メソッド。

        Args:
            args: gitコマンドの引数リスト（"git"を除く）
            cwd: 実行ディレクトリ

        Returns:
            コマンドの標準出力

        Raises:
            GitOperationError: コマンドが失敗した場合
        """
        cmd = ["git"] + args
        try:
            result = subprocess.run(
                cmd,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                check=True,
                encoding="utf-8",
                errors="replace",
            )
            return result.stdout
        except subprocess.CalledProcessError as exc:
            raise GitOperationError(
                f"gitコマンド失敗: {' '.join(cmd)}\n{exc.stderr}"
            ) from exc
        except FileNotFoundError as exc:
            raise GitOperationError("gitコマンドが見つかりません") from exc

    def _run_no_raise(self, args: List[str], cwd: Path, default: str = "") -> str:
        """
        エラー時にデフォルト値を返すgitコマンド実行内部メソッド。

        Args:
            args: gitコマンドの引数リスト
            cwd: 実行ディレクトリ
            default: エラー時のデフォルト値

        Returns:
            コマンドの標準出力またはデフォルト値
        """
        try:
            return self._run(args, cwd)
        except GitOperationError:
            return default

    def init(self, path: Path) -> None:
        """
        指定パスでgit initを実行する。

        Args:
            path: 初期化するディレクトリ

        Raises:
            GitOperationError: git initが失敗した場合
        """
        try:
            self._run(["init"], path)
            logger.info("git init完了: %s", path)
            # LocalForgeの自動コミット用にGPG署名を無効化する
            self._run_no_raise(["config", "commit.gpgsign", "false"], path)
            # .gitignoreを作成する
            # .localforge/ 全体を除外する: インデックス（ファイル内容のサマリー）、
            # 生成ログ、キャッシュ等が誤ってリポジトリにコミットされ情報漏洩するのを防ぐ
            gitignore_path = path / ".gitignore"
            if not gitignore_path.exists():
                gitignore_path.write_text(
                    ".localforge/\n__pycache__/\n*.pyc\n.venv/\nvenv/\n*.bak\n",
                    encoding="utf-8",
                )
        except GitOperationError as exc:
            raise GitOperationError(f"git init失敗: {exc}") from exc

    def commit_all(self, path: Path, message: str) -> str:
        """
        すべての変更をステージングしてコミットする。

        Args:
            path: リポジトリのルートディレクトリ
            message: コミットメッセージ

        Returns:
            コミットハッシュ（取得できない場合は空文字列）

        Raises:
            GitOperationError: コミットが失敗した場合
        """
        try:
            # ユーザー設定がない場合のデフォルト設定
            self._run_no_raise(
                ["config", "user.email", "localforge@local"], path
            )
            self._run_no_raise(
                ["config", "user.name", "LocalForge"], path
            )
            # すべての変更をステージング
            self._run(["add", "-A"], path)
            # コミット（変更なしの場合はエラーになるが無視）
            # LocalForgeの自動コミットではGPG署名を無効化する（CI/CD・自動化環境対応）
            try:
                self._run(["-c", "commit.gpgsign=false", "commit", "-m", message], path)
            except GitOperationError as exc:
                if "nothing to commit" in str(exc) or "nothing added to commit" in str(exc):
                    logger.debug("コミット対象なし: %s", path)
                    return ""
                raise
            # コミットハッシュを取得
            commit_hash = self._run_no_raise(
                ["rev-parse", "--short", "HEAD"], path
            ).strip()
            logger.info("コミット完了: %s (%s)", message[:50], commit_hash)
            return commit_hash
        except GitOperationError:
            raise

    def get_log(self, path: Path, max_entries: int = 20) -> List[dict]:
        """
        gitコミットログを返す。

        Args:
            path: リポジトリのルートディレクトリ
            max_entries: 最大取得件数

        Returns:
            コミット情報のリスト（hash, message, author, date）
        """
        if not self._is_git_repo(path):
            return []

        try:
            output = self._run(
                [
                    "log",
                    f"-{max_entries}",
                    "--pretty=format:%H|%s|%an|%ai",
                    "--no-merges",
                ],
                path,
            )
        except GitOperationError:
            return []

        entries: List[dict] = []
        for line in output.strip().splitlines():
            parts = line.split("|", 3)
            if len(parts) == 4:
                entries.append({
                    "hash": parts[0][:8],
                    "message": parts[1],
                    "author": parts[2],
                    "date": parts[3],
                })
        return entries

    def get_diff(self, path: Path) -> str:
        """
        git diff HEADの結果を返す。

        Args:
            path: リポジトリのルートディレクトリ

        Returns:
            diff出力テキスト
        """
        if not self._is_git_repo(path):
            return ""
        return self._run_no_raise(["diff", "HEAD"], path)

    def get_status(self, path: Path) -> str:
        """
        git status --shortの結果を返す。

        Args:
            path: リポジトリのルートディレクトリ

        Returns:
            ステータス出力テキスト
        """
        if not self._is_git_repo(path):
            return ""
        return self._run_no_raise(["status", "--short"], path)

    def get_current_branch(self, path: Path) -> str:
        """
        現在のgitブランチ名を返す。

        Args:
            path: リポジトリのルートディレクトリ

        Returns:
            ブランチ名（git管理外の場合は空文字列）
        """
        if not self._is_git_repo(path):
            return ""
        return self._run_no_raise(
            ["rev-parse", "--abbrev-ref", "HEAD"], path, default=""
        ).strip()

    def _is_git_repo(self, path: Path) -> bool:
        """
        指定パスがgitリポジトリかどうかを確認する内部メソッド。

        Args:
            path: 確認するディレクトリ

        Returns:
            gitリポジトリであればTrue
        """
        git_dir = path / ".git"
        return git_dir.is_dir()

    # ------------------------------------------------------------------
    # チェックポイント・ブランチ操作
    # ------------------------------------------------------------------

    def create_checkpoint(self, path: Path, label: str = "") -> str:
        """
        現在の状態をコミットしてチェックポイントタグを作成する。
        未コミット変更がある場合はまず自動コミットする。

        Args:
            path: リポジトリルート
            label: チェックポイントラベル（プラン名など）

        Returns:
            チェックポイントのコミットハッシュ（失敗時は空文字列）
        """
        if not self._is_git_repo(path):
            return ""
        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        tag_label = label or "batch"
        msg = f"[LocalForge checkpoint] pre-{tag_label} @ {ts}"
        try:
            # 未コミット変更があればコミットしておく
            status = self._run_no_raise(["status", "--short"], path).strip()
            if status:
                self.commit_all(path, msg)
            commit_hash = self._run_no_raise(
                ["rev-parse", "--short", "HEAD"], path
            ).strip()
            # 軽量タグでチェックポイントをマーク
            tag_name = f"localforge-cp-{ts}"
            self._run_no_raise(["tag", tag_name], path)
            logger.info("チェックポイント作成: %s (%s)", msg, commit_hash)
            return commit_hash
        except Exception as exc:
            logger.warning("チェックポイント作成失敗: %s", exc)
            return ""

    def rollback_to_checkpoint(self, path: Path, commit_hash: str) -> bool:
        """
        指定コミットハッシュに hard reset する。

        Args:
            path: リポジトリルート
            commit_hash: ロールバック先のコミットハッシュ

        Returns:
            成功時 True
        """
        if not self._is_git_repo(path) or not commit_hash:
            return False
        try:
            self._run(["reset", "--hard", commit_hash], path)
            logger.info("ロールバック完了: %s", commit_hash)
            return True
        except GitOperationError as exc:
            logger.error("ロールバック失敗: %s", exc)
            return False

    def get_last_checkpoint(self, path: Path) -> Optional[dict]:
        """
        最新の LocalForge チェックポイントタグを返す。

        Returns:
            {"hash": ..., "tag": ..., "message": ...} または None
        """
        if not self._is_git_repo(path):
            return None
        try:
            # localforge-cp-* タグのうち最新を取得
            output = self._run_no_raise(
                ["tag", "--list", "localforge-cp-*", "--sort=-creatordate"],
                path,
            ).strip()
            if not output:
                return None
            latest_tag = output.splitlines()[0].strip()
            commit_hash = self._run_no_raise(
                ["rev-list", "-n", "1", latest_tag], path
            ).strip()[:8]
            msg = self._run_no_raise(
                ["log", "-1", "--pretty=%s", latest_tag], path
            ).strip()
            return {"hash": commit_hash, "tag": latest_tag, "message": msg}
        except Exception:
            return None

    def create_and_switch_branch(self, path: Path, branch_name: str) -> str:
        """
        新しいブランチを作成してスイッチする。

        Args:
            path: リポジトリルート
            branch_name: 作成するブランチ名

        Returns:
            ブランチ名（失敗時は空文字列）
        """
        if not self._is_git_repo(path):
            return ""
        try:
            self._run(["checkout", "-b", branch_name], path)
            logger.info("ブランチ作成・スイッチ: %s", branch_name)
            return branch_name
        except GitOperationError as exc:
            logger.error("ブランチ作成失敗 [%s]: %s", branch_name, exc)
            return ""

    def merge_branch(self, path: Path, source_branch: str, target_branch: str = "main") -> bool:
        """
        source_branch を target_branch にマージする。

        Args:
            path: リポジトリルート
            source_branch: マージ元ブランチ
            target_branch: マージ先ブランチ

        Returns:
            成功時 True
        """
        if not self._is_git_repo(path):
            return False
        try:
            current = self.get_current_branch(path)
            if current != target_branch:
                self._run(["checkout", target_branch], path)
            self._run(["merge", "--no-ff", source_branch], path)
            logger.info("マージ完了: %s → %s", source_branch, target_branch)
            return True
        except GitOperationError as exc:
            logger.error("マージ失敗 [%s → %s]: %s", source_branch, target_branch, exc)
            return False
