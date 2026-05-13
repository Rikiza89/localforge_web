"""
HuggingFace モデル管理 — safetensors 形式モデルのカタログ・ダウンロード・スキャン。
モデルはリポジトリ単位でディレクトリに保存される（GGUF 単一ファイル形式ではない）。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# app.log と同じ .localforge/ フォルダ内にモデルを保存する
_APP_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = _APP_ROOT / ".localforge" / "models"

_HF_API_BASE = "https://huggingface.co/api"
_HF_API_TIMEOUT = 15

# SSL 検証無効化フラグ（自己署名証明書プロキシ環境向け自動フォールバック）
_ssl_verify_disabled = False


def _disable_ssl_verify() -> None:
    global _ssl_verify_disabled
    if _ssl_verify_disabled:
        return
    try:
        import ssl, urllib3
        ssl._create_default_https_context = ssl._create_unverified_context
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        # configure_http_backend は huggingface_hub >= 0.20 で利用可能
        try:
            import requests as _req
            from huggingface_hub import configure_http_backend

            def _insecure_session() -> _req.Session:
                s = _req.Session()
                s.verify = False
                return s

            configure_http_backend(backend_factory=_insecure_session)
        except ImportError:
            pass  # ssl._create_default_https_context で代替

        _ssl_verify_disabled = True
        logger.warning("SSL 証明書の検証を無効化しました（自己署名証明書プロキシを検出）。")
    except Exception as e:
        logger.warning("SSL 無効化の設定に失敗しました: %s", e)


def _is_ssl_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(kw in msg for kw in [
        "certificate verify failed", "ssl", "cert", "certificate_verify_failed",
    ])


# ---------------------------------------------------------------------------
# キュレート済みモデルカタログ — CPU 専用・safetensors 形式
# RAM 使用量は float16 ロード時のおおよその値（ダウンロードサイズとほぼ同等）
# ---------------------------------------------------------------------------

HF_MODEL_CATALOG: List[Dict] = [

    # -----------------------------------------------------------------------
    # Ultra-light — ~0.3–2.2 GB download / RAM
    # -----------------------------------------------------------------------
    {
        "id": "smollm2-135m",
        "name": "SmolLM2 135M Instruct",
        "repo_id": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "size_gb": 0.3,
        "description": "HuggingFace の超軽量モデル。テスト・デモ向け。",
        "recommended_for": "超高速テスト・デモ",
        "tags": ["fast", "small"],
    },
    {
        "id": "smollm2-360m",
        "name": "SmolLM2 360M Instruct",
        "repo_id": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "size_gb": 0.7,
        "description": "HuggingFace の小型高性能モデル。",
        "recommended_for": "超軽量・高速応答",
        "tags": ["fast", "small"],
    },
    {
        "id": "qwen2.5-0.5b",
        "name": "Qwen2.5 0.5B Instruct",
        "repo_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "size_gb": 1.0,
        "description": "Alibaba の最小 Qwen2.5。CPU でも高速。",
        "recommended_for": "超高速応答・最小メモリ",
        "tags": ["fast", "small"],
    },
    {
        "id": "tinyllama-1.1b",
        "name": "TinyLlama 1.1B Chat",
        "repo_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "size_gb": 2.2,
        "description": "コンパクトな Llama 系チャットモデル。",
        "recommended_for": "軽量・汎用チャット",
        "tags": ["fast", "small"],
    },

    # -----------------------------------------------------------------------
    # Light — ~3–4 GB download / RAM
    # -----------------------------------------------------------------------
    {
        "id": "qwen2.5-1.5b",
        "name": "Qwen2.5 1.5B Instruct",
        "repo_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "size_gb": 3.1,
        "description": "バランスの取れた小型汎用モデル。",
        "recommended_for": "軽量・高品質応答",
        "tags": ["balanced", "recommended"],
    },
    {
        "id": "qwen2.5-coder-1.5b",
        "name": "Qwen2.5 Coder 1.5B Instruct",
        "repo_id": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "size_gb": 3.1,
        "description": "コード特化 1.5B モデル。軽量で実用的。",
        "recommended_for": "軽量コード生成・補完",
        "tags": ["code", "recommended"],
    },
    {
        "id": "smollm2-1.7b",
        "name": "SmolLM2 1.7B Instruct",
        "repo_id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "size_gb": 3.4,
        "description": "HuggingFace の 1.7B 高性能モデル。",
        "recommended_for": "汎用・高速応答",
        "tags": ["balanced"],
    },

    # -----------------------------------------------------------------------
    # Medium — ~6–8 GB download / RAM
    # -----------------------------------------------------------------------
    {
        "id": "qwen2.5-3b",
        "name": "Qwen2.5 3B Instruct",
        "repo_id": "Qwen/Qwen2.5-3B-Instruct",
        "size_gb": 6.2,
        "description": "品質と速度のバランスが優れた 3B モデル。",
        "recommended_for": "汎用目的・バランス重視",
        "tags": ["balanced", "recommended"],
    },
    {
        "id": "qwen2.5-coder-3b",
        "name": "Qwen2.5 Coder 3B Instruct",
        "repo_id": "Qwen/Qwen2.5-Coder-3B-Instruct",
        "size_gb": 6.2,
        "description": "コード特化 3B。実用的なコード生成に最適。",
        "recommended_for": "コード生成・デバッグ",
        "tags": ["code", "recommended"],
    },
    {
        "id": "phi-3.5-mini",
        "name": "Phi-3.5 Mini Instruct (3.8B)",
        "repo_id": "microsoft/Phi-3.5-mini-instruct",
        "size_gb": 7.6,
        "description": "Microsoft の Phi-3.5 Mini。サイズ対性能比が優秀。",
        "recommended_for": "推論・数学・コード",
        "tags": ["reasoning", "code"],
    },

    # -----------------------------------------------------------------------
    # Large — ~15 GB download / RAM（32GB RAM 推奨）
    # -----------------------------------------------------------------------
    {
        "id": "qwen2.5-7b",
        "name": "Qwen2.5 7B Instruct",
        "repo_id": "Qwen/Qwen2.5-7B-Instruct",
        "size_gb": 15.2,
        "description": "高品質 7B 汎用モデル。CPU では低速だが高精度。",
        "recommended_for": "高精度汎用・32GB RAM 推奨",
        "tags": ["balanced", "large"],
    },
    {
        "id": "qwen2.5-coder-7b",
        "name": "Qwen2.5 Coder 7B Instruct",
        "repo_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "size_gb": 15.2,
        "description": "コード特化 7B モデル。OSS 最高クラスの生成品質。",
        "recommended_for": "高精度コード生成・32GB RAM 推奨",
        "tags": ["code", "large"],
    },
]


# ---------------------------------------------------------------------------
# パスユーティリティ
# ---------------------------------------------------------------------------

def repo_dest_dir(repo_id: str) -> Path:
    """
    リポジトリの保存ディレクトリパスを返す。
    スラッシュを "--" に置換して OS セーフなディレクトリ名にする。
    例: Qwen/Qwen2.5-1.5B-Instruct → .localforge/models/Qwen--Qwen2.5-1.5B-Instruct
    """
    return MODELS_DIR / repo_id.replace("/", "--")


def get_catalog_model(model_id: str) -> Optional[Dict]:
    for m in HF_MODEL_CATALOG:
        if m["id"] == model_id:
            return m
    return None


# ---------------------------------------------------------------------------
# ローカルモデルスキャン
# ---------------------------------------------------------------------------

def scan_local_models() -> List[Dict]:
    """
    MODELS_DIR 以下の HuggingFace モデルディレクトリ一覧を返す。
    config.json が存在するディレクトリのみ有効なモデルとして認識する。
    """
    result = []
    if not MODELS_DIR.exists():
        return result

    for model_dir in sorted(MODELS_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        if not (model_dir / "config.json").is_file():
            continue

        model_dir = model_dir.resolve()
        total_bytes = sum(
            f.stat().st_size for f in model_dir.rglob("*") if f.is_file()
        )
        size_gb = round(total_bytes / (1024 ** 3), 2)

        # ディレクトリ名からカタログエントリを逆引き
        dir_name = model_dir.name  # e.g. "Qwen--Qwen2.5-1.5B-Instruct"
        catalog_entry = next(
            (m for m in HF_MODEL_CATALOG if repo_dest_dir(m["repo_id"]).name == dir_name),
            None,
        )

        result.append({
            "id": dir_name,
            "name": catalog_entry["name"] if catalog_entry else dir_name.replace("--", "/"),
            "path": str(model_dir),
            "size_gb": size_gb,
            "description": catalog_entry["description"] if catalog_entry else "ローカルモデル",
            "recommended_for": catalog_entry.get("recommended_for", "") if catalog_entry else "",
            "tags": catalog_entry.get("tags", []) if catalog_entry else [],
            "in_catalog": catalog_entry is not None,
        })

    return result


def get_catalog_with_status() -> List[Dict]:
    """カタログモデルにダウンロード済みステータスを付与して返す。"""
    result = []
    for model in HF_MODEL_CATALOG:
        dest_dir = repo_dest_dir(model["repo_id"])
        downloaded = (dest_dir / "config.json").is_file()
        entry = dict(model)
        entry["downloaded"] = downloaded
        entry["local_path"] = str(dest_dir.resolve()) if downloaded else ""
        entry["dest_dir"] = str(dest_dir)
        result.append(entry)
    return result


# ---------------------------------------------------------------------------
# ダウンロード対象ファイルリスト取得
# ---------------------------------------------------------------------------

def get_repo_download_files(repo_id: str) -> List[Dict]:
    """
    HuggingFace API でリポジトリのファイル一覧を取得し、
    ダウンロードすべきファイル（safetensors + 設定ファイル）のリストを返す。
    pytorch_model*.bin は safetensors が存在する場合はスキップする。
    """
    import requests

    def _fetch(verify=True):
        return requests.get(
            f"{_HF_API_BASE}/models/{repo_id}",
            timeout=_HF_API_TIMEOUT,
            verify=verify,
        )

    try:
        resp = _fetch()
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        if _is_ssl_error(exc):
            _disable_ssl_verify()
            try:
                resp = _fetch(verify=False)
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc2:
                raise RuntimeError(f"ファイルリストの取得に失敗しました ({repo_id}): {exc2}") from exc2
        else:
            raise RuntimeError(f"ファイルリストの取得に失敗しました ({repo_id}): {exc}") from exc

    siblings = data.get("siblings", [])
    filenames = [s.get("rfilename", "") for s in siblings]
    sizes: Dict[str, int] = {s.get("rfilename", ""): s.get("size") or 0 for s in siblings}

    has_safetensors = any(f.endswith(".safetensors") for f in filenames)

    result = []
    for fname in filenames:
        lower = fname.lower()
        # 不要なフォーマットをスキップ
        if any(lower.endswith(e) for e in [".h5", ".msgpack", ".ot", ".gguf"]):
            continue
        if any(s in lower for s in ["flax_model", "tf_model", "rust_model", "openvino", "onnx"]):
            continue
        # safetensors が存在する場合は pytorch_model*.bin をスキップ
        if has_safetensors and lower.endswith(".bin") and "pytorch_model" in lower:
            continue
        result.append({"filename": fname, "size": sizes.get(fname, 0)})

    return result


# ---------------------------------------------------------------------------
# ダウンロード（単一ファイル）
# ---------------------------------------------------------------------------

def download_file(repo_id: str, filename: str) -> str:
    """
    repo_id + filename を指定して 1 ファイルをダウンロードする。
    SSL エラーの場合は検証を無効化して自動再試行する。

    Returns:
        ダウンロードされたファイルの絶対パス
    """
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError
    except ImportError as exc:
        raise FileNotFoundError(
            f"huggingface-hub がインストールされていません。詳細: {exc}"
        ) from exc

    dest_dir = repo_dest_dir(repo_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename

    if dest_path.is_file():
        logger.debug("既にダウンロード済み: %s", dest_path)
        return str(dest_path)

    logger.info("ダウンロード中: %s / %s → %s", repo_id, filename, dest_dir)

    def _do():
        return hf_hub_download(
            repo_id=repo_id, filename=filename, local_dir=str(dest_dir),
        )

    try:
        path = _do()
        return str(path)
    except (HfHubHTTPError, RepositoryNotFoundError, Exception) as exc:
        if _is_ssl_error(exc):
            logger.warning("SSL エラーを検出。SSL 検証を無効化して再試行します。")
            _disable_ssl_verify()
            try:
                path = _do()
                return str(path)
            except Exception as exc2:
                exc = exc2
        exc_str = str(exc)
        proxy_kws = ["ProxyError", "ConnectionError", "SSLError", "Timeout",
                     "proxy", "403", "407", "firewall"]
        if any(kw.lower() in exc_str.lower() for kw in proxy_kws):
            raise RuntimeError("プロキシまたはネットワーク制限によりブロックされました。") from exc
        raise RuntimeError(f"ダウンロードに失敗しました: {exc_str}") from exc


# ---------------------------------------------------------------------------
# 手動ダウンロード手順
# ---------------------------------------------------------------------------

def get_manual_instructions(repo_id: str, model_name: str = "") -> Dict:
    """
    手動ダウンロード用の詳細手順を返す。保存先ディレクトリを事前に作成する。
    """
    dest_dir = repo_dest_dir(repo_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = model_name or repo_id.split("/")[-1]
    dest_display = str(dest_dir).replace("\\", "/")

    # 手動ダウンロード用スクリプトをフォルダに書き出す（SSL バイパス込み）
    script_path = dest_dir / "download.py"
    script_path.write_text(
        '# Manual download script — run with: venv/Scripts/python download.py\n'
        'import ssl, urllib3\n'
        'ssl._create_default_https_context = ssl._create_unverified_context\n'
        'urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)\n\n'
        'from huggingface_hub import snapshot_download\n\n'
        f'snapshot_download(\n'
        f'    "{repo_id}",\n'
        f'    local_dir=r"{dest_display}",\n'
        f'    ignore_patterns=["*.h5", "*.msgpack", "flax_*", "tf_*", "rust_*"],\n'
        f')\n'
        f'print("Done:", r"{dest_display}")\n',
        encoding="utf-8",
    )

    return {
        "model_name":  name,
        "repo_id":     repo_id,
        "dest_dir":    str(dest_dir),
        "dest_display": dest_display,
        "cli_cmd":     f'huggingface-cli download {repo_id} --local-dir "{dest_display}"',
        "python_cmd":  (
            f'from huggingface_hub import snapshot_download\n'
            f'snapshot_download("{repo_id}", local_dir=r"{dest_display}")'
        ),
        "python_oneliner": (
            f'from huggingface_hub import snapshot_download; '
            f'snapshot_download("{repo_id}", local_dir=r"{dest_display}")'
        ),
        "steps": [
            f"1. Python でダウンロード（推奨・venv から実行）:",
            f"   venv\\Scripts\\python -c \"from huggingface_hub import snapshot_download; snapshot_download('{repo_id}', local_dir=r'{dest_display}')\"",
            f"",
            f"2. または huggingface-cli（venv を有効化している場合）:",
            f"   huggingface-cli download {repo_id} --local-dir \"{dest_display}\"",
            f"",
            f"3. ダウンロード先（自動作成済み）:",
            f"   {dest_display}",
            f"",
            f"4. ダウンロード完了後、「ローカルモデル」タブで「↻ 再スキャン」をクリックしてロードしてください。",
        ],
    }


# ---------------------------------------------------------------------------
# ライブ検索
# ---------------------------------------------------------------------------

def search_hf_live(query: str = "", limit: int = 20) -> List[Dict]:
    """HuggingFace API で safetensors 形式のモデルをライブ検索する。"""
    import requests

    params: Dict = {
        "filter": "safetensors",
        "sort": "downloads",
        "direction": "-1",
        "limit": min(limit, 50),
        "full": "false",
    }
    if query:
        params["search"] = query

    def _fetch(verify=True):
        return requests.get(
            f"{_HF_API_BASE}/models", params=params,
            timeout=_HF_API_TIMEOUT, verify=verify,
        )

    try:
        resp = _fetch()
        resp.raise_for_status()
    except Exception as exc:
        if _is_ssl_error(exc):
            _disable_ssl_verify()
            try:
                resp = _fetch(verify=False)
                resp.raise_for_status()
            except Exception as exc2:
                raise RuntimeError(f"HuggingFace API に接続できません: {exc2}") from exc2
        else:
            raise RuntimeError(f"HuggingFace API に接続できません: {exc}") from exc

    result = []
    for m in resp.json():
        repo_id = m.get("id", "")
        dest = repo_dest_dir(repo_id)
        result.append({
            "repo_id":       repo_id,
            "name":          repo_id.split("/")[-1],
            "downloads":     m.get("downloads", 0),
            "likes":         m.get("likes", 0),
            "tags":          m.get("tags", []),
            "last_modified": m.get("lastModified", ""),
            "downloaded":    (dest / "config.json").is_file(),
            "local_path":    str(dest.resolve()) if (dest / "config.json").is_file() else "",
        })
    return result
