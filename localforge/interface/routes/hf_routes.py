"""
HuggingFace ルート — /api/hf/* エンドポイント。
safetensors 形式モデルのカタログ・ダウンロード・ロード・プロバイダー切替を提供する。
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from flask import Blueprint, Response, current_app, jsonify, request, stream_with_context

from localforge.infrastructure import hf_model_manager as mgr
from localforge.infrastructure.hf_model_manager import MODELS_DIR

logger = logging.getLogger(__name__)

bp = Blueprint("hf", __name__, url_prefix="/api/hf")


def _get_router():
    return current_app.config["llm"]


def _get_project_svc():
    return current_app.config["project_service"]


# ---------------------------------------------------------------------------
# モデルカタログ
# ---------------------------------------------------------------------------

@bp.route("/models", methods=["GET"])
def list_models():
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
    HuggingFace Hub からモデルをダウンロードし、ファイルごとに進行状況を SSE で配信する。

    Query params:
        model_id (str): カタログの model ID
        repo_id  (str): 任意の HF repo ID（ライブ検索用）
    """
    model_id = request.args.get("model_id", "").strip()
    repo_id  = request.args.get("repo_id",  "").strip()

    if model_id:
        model_info = mgr.get_catalog_model(model_id)
        if not model_info:
            return jsonify({"error": f"カタログに存在しないモデル ID: {model_id}"}), 404
        repo_id  = model_info["repo_id"]
        display  = model_info["name"]
    elif repo_id:
        display = repo_id.split("/")[-1]
    else:
        return jsonify({"error": "model_id または repo_id を指定してください"}), 400

    def _generate():
        def _sse(event: str, data: dict) -> str:
            return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

        yield _sse("status", {"status": f"{display} のファイルリストを取得中..."})

        # ダウンロード対象ファイルリストを取得
        try:
            files = mgr.get_repo_download_files(repo_id)
        except RuntimeError as exc:
            exc_str = str(exc)
            is_proxy = "プロキシ" in exc_str or "ネットワーク" in exc_str or "接続" in exc_str
            inst = None
            if is_proxy:
                try:
                    inst = mgr.get_manual_instructions(repo_id, display)
                except Exception:
                    pass
            yield _sse("error", {"error": exc_str, "proxy_error": is_proxy, "instructions": inst})
            return
        except Exception as exc:
            yield _sse("error", {"error": str(exc), "proxy_error": False, "instructions": None})
            return

        if not files:
            yield _sse("error", {
                "error": f"ダウンロード対象ファイルが見つかりません: {repo_id}",
                "proxy_error": False, "instructions": None,
            })
            return

        total_bytes = sum(f.get("size", 0) for f in files)
        total_files = len(files)
        done_bytes = 0

        yield _sse("status", {"status": f"{total_files} ファイル（合計 {_fmt_gb(total_bytes)} GB）をダウンロード中..."})

        for i, file_info in enumerate(files):
            fname = file_info["filename"]
            yield _sse("status", {"status": f"({i+1}/{total_files}) {fname}..."})

            try:
                mgr.download_file(repo_id, fname)
            except RuntimeError as exc:
                exc_str = str(exc)
                is_proxy = "プロキシ" in exc_str or "ネットワーク" in exc_str
                inst = None
                if is_proxy:
                    try:
                        inst = mgr.get_manual_instructions(repo_id, display)
                    except Exception:
                        pass
                yield _sse("error", {"error": exc_str, "proxy_error": is_proxy, "instructions": inst})
                return
            except Exception as exc:
                yield _sse("error", {"error": str(exc), "proxy_error": False, "instructions": None})
                return

            done_bytes += file_info.get("size", 0)
            if total_bytes:
                yield _sse("progress", {"done": done_bytes, "total": total_bytes})

        dest_dir = str(mgr.repo_dest_dir(repo_id).resolve())
        yield _sse("done", {"done": True, "path": dest_dir})

    return Response(
        stream_with_context(_generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _fmt_gb(size_bytes: int) -> str:
    if not size_bytes:
        return "?"
    return f"{size_bytes / (1024 ** 3):.1f}"


# ---------------------------------------------------------------------------
# ライブ検索
# ---------------------------------------------------------------------------

@bp.route("/search", methods=["GET"])
def search_models():
    query = request.args.get("q", "").strip()
    limit = min(int(request.args.get("limit", 20)), 50)

    try:
        results = mgr.search_hf_live(query=query, limit=limit)
        return jsonify({"models": results, "query": query, "online": True})
    except RuntimeError as exc:
        logger.warning("HuggingFace API 検索失敗: %s", exc)
        return jsonify({"models": [], "query": query, "online": False, "error": str(exc)})


# ---------------------------------------------------------------------------
# 手動ダウンロード案内
# ---------------------------------------------------------------------------

@bp.route("/instructions", methods=["POST"])
def get_instructions():
    """
    手動ダウンロード用の手順を返す（カタログモデルおよびライブ検索モデル共通）。

    Request JSON:
        model_id   (str, optional): カタログの model ID
        repo_id    (str, optional): 任意の HF repo ID
        model_name (str, optional): 表示名
    """
    data       = request.get_json(silent=True) or {}
    model_id   = data.get("model_id",   "").strip()
    repo_id    = data.get("repo_id",    "").strip()
    model_name = data.get("model_name", "").strip()

    if model_id:
        model_info = mgr.get_catalog_model(model_id)
        if not model_info:
            return jsonify({"error": f"カタログに存在しないモデル ID: {model_id}"}), 404
        repo_id    = model_info["repo_id"]
        model_name = model_name or model_info["name"]
    elif not repo_id:
        return jsonify({"error": "model_id または repo_id が必要です"}), 400

    try:
        inst = mgr.get_manual_instructions(repo_id, model_name)
        return jsonify({"instructions": inst})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# モデルロード
# ---------------------------------------------------------------------------

@bp.route("/load", methods=["POST"])
def load_model():
    """
    ローカルの safetensors モデルディレクトリを HuggingFace クライアントにロードする。

    Request JSON:
        path      (str): モデルディレクトリの絶対パス（config.json を含む）
        n_threads (int, optional): CPU スレッド数（デフォルト 自動）
    """
    data       = request.get_json(silent=True) or {}
    model_path = data.get("path", "").strip()
    n_threads  = data.get("n_threads", 0)

    if not model_path:
        return jsonify({"error": "path が指定されていません"}), 400

    path = Path(model_path).resolve()

    logger.info(
        "HF load リクエスト: raw=%r  resolved=%s  exists=%s  MODELS_DIR=%s",
        model_path, path, path.exists(), MODELS_DIR.resolve(),
    )

    if not path.exists():
        models_dir_display = str(MODELS_DIR).replace("\\", "/")
        return jsonify({
            "error": (
                f"ディレクトリが見つかりません: {path}\n"
                f"モデルを {models_dir_display} に配置してください"
            )
        }), 404

    if not (path / "config.json").is_file():
        return jsonify({
            "error": f"config.json が見つかりません。有効な HuggingFace モデルディレクトリを指定してください: {path}"
        }), 400

    router = _get_router()

    try:
        router.switch_provider("huggingface")
        router.hf.load_model(str(path), n_threads=n_threads)

        project_svc = _get_project_svc()
        project = project_svc.current_project
        if project:
            project.config.llm_provider = "huggingface"
            project.config.hf_model_path = str(path)
            project_svc.save_config(project.root, project.config)

        return jsonify({"loaded": True, "path": str(path)})

    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        logger.error("HF モデルロードエラー: %s", exc)
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# スキャン
# ---------------------------------------------------------------------------

@bp.route("/scan", methods=["GET"])
def scan_models():
    local = mgr.scan_local_models()
    return jsonify({"local": local})


# ---------------------------------------------------------------------------
# プロバイダー切替
# ---------------------------------------------------------------------------

@bp.route("/provider", methods=["POST"])
def switch_provider():
    data     = request.get_json(silent=True) or {}
    provider = data.get("provider", "").strip()

    if provider not in ("ollama", "huggingface"):
        return jsonify({"error": "provider は 'ollama' または 'huggingface' を指定してください"}), 400

    router = _get_router()
    try:
        router.switch_provider(provider)

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
    router = _get_router()
    router.hf.unload_model()
    return jsonify({"unloaded": True})


# ---------------------------------------------------------------------------
# ステータス
# ---------------------------------------------------------------------------

@bp.route("/status", methods=["GET"])
def hf_status():
    router = _get_router()
    return jsonify({
        "available":      router.hf.is_available(),
        "loaded":         router.hf.is_model_loaded(),
        "loaded_model":   router.hf.get_loaded_model_path(),
        "active_provider": router.active_provider,
    })
