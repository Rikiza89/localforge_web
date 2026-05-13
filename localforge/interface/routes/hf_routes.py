"""
HuggingFace ルート — /api/hf/* エンドポイントの定義。
モデルカタログ・ダウンロード・手動案内・プロバイダー切替を提供する。
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from flask import Blueprint, Response, current_app, jsonify, request, stream_with_context

from localforge.infrastructure import hf_model_manager as mgr
from localforge.infrastructure.hf_model_manager import MODELS_DIR  # used in load route 404 message

logger = logging.getLogger(__name__)

bp = Blueprint("hf", __name__, url_prefix="/api/hf")

# ダウンロード中かどうかを追跡するグローバルフラグ（1 ダウンロードのみ同時実行）
_download_lock = threading.Lock()
_active_download: dict = {}   # {"model_id": str, "cancel": bool}


def _get_router():
    return current_app.config["llm"]


def _get_project_svc():
    return current_app.config["project_service"]


# ---------------------------------------------------------------------------
# モデルカタログ
# ---------------------------------------------------------------------------

@bp.route("/models", methods=["GET"])
def list_models():
    """
    カタログモデル一覧（ダウンロード状態付き）とローカルモデル一覧を返す。

    Response JSON:
        catalog: カタログモデルのリスト（downloaded フラグ付き）
        local:   ローカルで見つかった GGUF ファイルのリスト
        active_provider: 現在のプロバイダー ("ollama" | "huggingface")
        loaded_model: HF クライアントが現在ロードしているモデルのパス（空文字 = なし）
    """
    router = _get_router()
    catalog = mgr.get_catalog_with_status()
    local = mgr.scan_local_models()
    loaded = router.hf.get_loaded_model_path()

    return jsonify({
        "catalog": catalog,
        "local": local,
        "active_provider": router.active_provider,
        "loaded_model": loaded,
        "models_dir": str(MODELS_DIR).replace("\\", "/"),
    })


# ---------------------------------------------------------------------------
# モデルダウンロード（SSE ストリーム）
# ---------------------------------------------------------------------------

@bp.route("/download", methods=["GET"])
def download_model():
    """
    HuggingFace Hub からモデルをダウンロードし、進行状況を SSE で配信する。

    Query params (どちらか一方を指定):
        model_id  (str): カタログの model ID
        repo_id   (str): 任意の HF repo ID（ライブ検索結果用）
        filename  (str): repo_id 指定時は必須

    SSE events:
        status: {"status": str}
        done:   {"done": true, "path": str}
        error:  {"error": str, "proxy_error": bool, "instructions": dict|null}
    """
    model_id  = request.args.get("model_id",  "").strip()
    repo_id   = request.args.get("repo_id",   "").strip()
    filename  = request.args.get("filename",  "").strip()

    # resolve to repo_id + filename regardless of input form
    if model_id:
        model_info = mgr.get_catalog_model(model_id)
        if not model_info:
            return jsonify({"error": f"カタログに存在しないモデル ID: {model_id}"}), 404
        repo_id  = model_info["repo_id"]
        filename = model_info["filename"]
        display  = model_info["name"]
    elif repo_id and filename:
        display = f"{repo_id.split('/')[-1]} / {filename}"
    else:
        return jsonify({"error": "model_id または repo_id+filename を指定してください"}), 400

    def _generate():
        def _sse(event: str, data: dict) -> str:
            return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

        yield _sse("status", {"status": f"{display} のダウンロードを開始します..."})

        try:
            path = mgr.download_file(repo_id, filename)
            yield _sse("done", {"done": True, "path": path})
        except RuntimeError as exc:
            exc_str = str(exc)
            is_proxy = "プロキシ" in exc_str or "ネットワーク" in exc_str
            instructions = None
            if is_proxy:
                try:
                    instructions = mgr.get_manual_instructions_for_file(
                        repo_id, filename, display
                    )
                except Exception:
                    pass
            yield _sse("error", {
                "error": exc_str,
                "proxy_error": is_proxy,
                "instructions": instructions,
            })
        except Exception as exc:
            yield _sse("error", {"error": str(exc), "proxy_error": False, "instructions": None})

    return Response(
        stream_with_context(_generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# ライブ検索
# ---------------------------------------------------------------------------

@bp.route("/search", methods=["GET"])
def search_models():
    """
    HuggingFace API で GGUF モデルをライブ検索する。

    Query params:
        q        (str, optional): 検索クエリ（空 = 人気順トップ）
        limit    (int, optional): 最大件数（デフォルト 20、最大 50）

    Response JSON:
        models: モデルリスト
        query:  実行したクエリ
        online: True（API 到達成功）
    """
    query = request.args.get("q", "").strip()
    limit = min(int(request.args.get("limit", 20)), 50)

    try:
        results = mgr.search_hf_live(query=query, limit=limit)
        return jsonify({"models": results, "query": query, "online": True})
    except RuntimeError as exc:
        logger.warning("HuggingFace API 検索失敗: %s", exc)
        # 200 を返す — フロントエンドは online:false を見てエラー表示する
        return jsonify({
            "models": [],
            "query": query,
            "online": False,
            "error": str(exc),
        })


@bp.route("/model-files", methods=["GET"])
def get_model_files():
    """
    指定リポジトリの GGUF ファイル一覧をサイズ付きで返す。

    Query params:
        repo_id      (str): HuggingFace repo ID
        max_size_gb  (float, optional): フィルタ上限 GB（デフォルト 19）

    Response JSON:
        files: ファイルリスト（サイズ昇順）
        repo_id: 対象 repo ID
    """
    repo_id     = request.args.get("repo_id", "").strip()
    max_size_gb = float(request.args.get("max_size_gb", 19.0))

    if not repo_id:
        return jsonify({"error": "repo_id が指定されていません"}), 400

    try:
        files = mgr.get_hf_model_files(repo_id, max_size_gb=max_size_gb)
        return jsonify({"files": files, "repo_id": repo_id})
    except RuntimeError as exc:
        logger.warning("HuggingFace model-files 取得失敗 (%s): %s", repo_id, exc)
        return jsonify({"error": str(exc), "files": [], "repo_id": repo_id})


@bp.route("/file-instructions", methods=["POST"])
def get_file_instructions():
    """
    ライブ検索ファイル向け手動ダウンロード手順を返す。

    Request JSON:
        repo_id    (str): HF repo ID
        filename   (str): GGUF ファイル名
        model_name (str, optional): 表示名
        size_gb    (float, optional): ファイルサイズ GB
    """
    data       = request.get_json(silent=True) or {}
    repo_id    = data.get("repo_id",    "").strip()
    filename   = data.get("filename",   "").strip()
    model_name = data.get("model_name", "").strip()
    size_gb    = data.get("size_gb")

    if not repo_id or not filename:
        return jsonify({"error": "repo_id と filename が必要です"}), 400

    try:
        inst = mgr.get_manual_instructions_for_file(repo_id, filename, model_name, size_gb)
        return jsonify({"instructions": inst})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# 手動ダウンロード案内
# ---------------------------------------------------------------------------

@bp.route("/instructions", methods=["POST"])
def get_instructions():
    """
    手動ダウンロード用の手順を返す。
    バックエンドが保存先フォルダを事前作成する。

    Request JSON:
        model_id (str): カタログの model ID

    Response JSON:
        instructions: 手順データ（url, wget_cmd, curl_cmd, steps, dest_dir, dest_path）
    """
    data = request.get_json(silent=True) or {}
    model_id = data.get("model_id", "").strip()
    if not model_id:
        return jsonify({"error": "model_id が指定されていません"}), 400

    try:
        instructions = mgr.get_manual_instructions(model_id)
        return jsonify({"instructions": instructions})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404


# ---------------------------------------------------------------------------
# モデルロード
# ---------------------------------------------------------------------------

@bp.route("/load", methods=["POST"])
def load_model():
    """
    ローカルの GGUF ファイルを HuggingFace クライアントにロードする。
    ロード前に現在のプロバイダーを huggingface に切り替える。

    Request JSON:
        path (str): GGUF ファイルの絶対パス
        n_ctx (int, optional): コンテキスト長（デフォルト 8192）
        n_threads (int, optional): CPU スレッド数（デフォルト 自動）

    Response JSON:
        loaded: True
        path: ロードされたモデルのパス
    """
    data = request.get_json(silent=True) or {}
    model_path = data.get("path", "").strip()
    n_ctx = data.get("n_ctx", 8192)
    n_threads = data.get("n_threads", 0)

    if not model_path:
        return jsonify({"error": "path が指定されていません"}), 400

    path = Path(model_path).resolve()

    logger.info(
        "HF load リクエスト: raw=%r  resolved=%s  exists=%s  MODELS_DIR=%s",
        model_path, path, path.exists(), MODELS_DIR.resolve(),
    )

    if path.suffix.lower() != ".gguf":
        logger.warning("HF load 拒否: GGUF 以外の拡張子 %s", path.suffix)
        return jsonify({"error": "GGUF ファイルのみサポートしています"}), 400

    if not path.is_file():
        models_dir_display = str(MODELS_DIR).replace("\\", "/")
        logger.warning(
            "HF load 失敗: ファイルが存在しない  path=%s  models_dir=%s  "
            "models_dir_exists=%s  models_dir_contents=%s",
            path,
            MODELS_DIR.resolve(),
            MODELS_DIR.exists(),
            [str(p) for p in MODELS_DIR.rglob("*.gguf")] if MODELS_DIR.exists() else [],
        )
        return jsonify({
            "error": (
                f"ファイルが見つかりません: {path}\n"
                f"モデルを {models_dir_display} に配置してください"
            )
        }), 404

    router = _get_router()

    try:
        # プロバイダーを HF に切替（Ollama モデルをアンロード）
        router.switch_provider("huggingface")
        # GGUF ファイルをロード
        router.hf.load_model(str(path), n_ctx=n_ctx, n_threads=n_threads)

        # プロジェクト設定に保存
        project_svc = _get_project_svc()
        project = project_svc.current_project
        if project:
            project.config.llm_provider = "huggingface"
            project.config.hf_model_path = str(path)
            project_svc.save_config(project.root, project.config)

        return jsonify({"loaded": True, "path": str(path)})

    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500
    except Exception as exc:
        logger.error("HF モデルロードエラー: %s", exc)
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# スキャン（ローカルモデル一覧を再取得）
# ---------------------------------------------------------------------------

@bp.route("/scan", methods=["GET"])
def scan_models():
    """
    ~/.localforge/models/ を再スキャンしてローカル GGUF ファイル一覧を返す。

    Response JSON:
        local: ローカルモデルのリスト
    """
    local = mgr.scan_local_models()
    return jsonify({"local": local})


# ---------------------------------------------------------------------------
# プロバイダー切替
# ---------------------------------------------------------------------------

@bp.route("/provider", methods=["POST"])
def switch_provider():
    """
    LLM プロバイダーを切り替える。
    切替前に現在のプロバイダーのモデルをアンロードする。

    Request JSON:
        provider (str): "ollama" または "huggingface"

    Response JSON:
        provider: 切り替え後のプロバイダー名
    """
    data = request.get_json(silent=True) or {}
    provider = data.get("provider", "").strip()

    if provider not in ("ollama", "huggingface"):
        return jsonify({"error": "provider は 'ollama' または 'huggingface' を指定してください"}), 400

    router = _get_router()
    try:
        router.switch_provider(provider)

        # プロジェクト設定に保存
        project_svc = _get_project_svc()
        project = project_svc.current_project
        if project:
            project.config.llm_provider = provider
            project_svc.save_config(project.root, project.config)

        return jsonify({"provider": router.active_provider})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# アンロード
# ---------------------------------------------------------------------------

@bp.route("/unload", methods=["POST"])
def unload_hf_model():
    """
    HuggingFace クライアントのモデルをアンロードしてメモリを解放する。

    Response JSON:
        unloaded: True
    """
    router = _get_router()
    router.hf.unload_model()
    return jsonify({"unloaded": True})


# ---------------------------------------------------------------------------
# ステータス
# ---------------------------------------------------------------------------

@bp.route("/status", methods=["GET"])
def hf_status():
    """
    HuggingFace クライアントの現在の状態を返す。

    Response JSON:
        available:     llama-cpp-python がインストールされているか
        loaded:        モデルがロード済みか
        loaded_model:  ロード済みモデルのパス（空文字 = なし）
        active_provider: 現在のプロバイダー
    """
    router = _get_router()
    return jsonify({
        "available": router.hf.is_available(),
        "loaded": router.hf.is_model_loaded(),
        "loaded_model": router.hf.get_loaded_model_path(),
        "active_provider": router.active_provider,
    })
