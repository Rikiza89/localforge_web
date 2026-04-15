"""
カスタム例外クラス群 — LocalForgeアプリケーション全体で使用するドメイン例外を定義する。
"""


class LocalForgeError(Exception):
    """LocalForgeアプリケーションの基底例外クラス。"""


class OllamaConnectionError(LocalForgeError):
    """OllamaサーバーへのHTTP接続に失敗した場合の例外。"""


class OllamaModelNotFoundError(LocalForgeError):
    """指定されたOllamaモデルが見つからない場合の例外。"""


class PlanParseError(LocalForgeError):
    """生成プランのJSONパースに失敗した場合の例外。"""


class FileWriteError(LocalForgeError):
    """ファイルの書き込みに失敗した場合の例外。"""


class GitOperationError(LocalForgeError):
    """git操作に失敗した場合の例外。"""


class ContextUpdateError(LocalForgeError):
    """コンテキスト（context.md）の更新に失敗した場合の例外。"""


class IndexBuildError(LocalForgeError):
    """ProjectIndexの構築に失敗した場合の例外。"""


class TokenBudgetExceededWarning(LocalForgeError):
    """
    トークン予算を超過した場合の警告例外。
    処理は継続されるが、プロンプトは切り詰められる。
    """


class ResumeStateCorruptError(LocalForgeError):
    """再開状態ファイルが破損または不整合な場合の例外。"""
