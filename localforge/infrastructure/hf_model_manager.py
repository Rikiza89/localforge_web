"""
HuggingFace モデル管理 — キュレート済みモデルカタログ、ダウンロード、手動ダウンロード案内。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)

# ローカルモデル保存ディレクトリ
MODELS_DIR = Path.home() / ".localforge" / "models"

# HuggingFace ダイレクトダウンロード URL テンプレート
_HF_DOWNLOAD_URL = "https://huggingface.co/{repo_id}/resolve/main/{filename}"

# ---------------------------------------------------------------------------
# キュレート済みモデルカタログ — 最大 19 GB、CPU 専用 32 GB RAM 環境向け
# すべて GGUF 形式。サイズに応じて Q4_K_M / Q3_K_M / Q2_K を使用。
# ---------------------------------------------------------------------------

HF_MODEL_CATALOG: List[Dict] = [

    # -----------------------------------------------------------------------
    # Ultra-light — ~0.7–1.6 GB
    # -----------------------------------------------------------------------
    {
        "id": "llama-3.2-1b",
        "name": "Llama 3.2 1B Instruct",
        "repo_id": "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "filename": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "size_gb": 0.7,
        "description": "Meta の最小 Llama モデル。超高速、最小メモリ。",
        "recommended_for": "超高速応答・最小メモリ",
        "tags": ["fast", "small"],
    },
    {
        "id": "gemma-2-2b",
        "name": "Gemma 2 2B Instruct",
        "repo_id": "bartowski/gemma-2-2b-it-GGUF",
        "filename": "gemma-2-2b-it-Q4_K_M.gguf",
        "size_gb": 1.6,
        "description": "Google の Gemma 2 2B。小型ながら洗練された応答品質。",
        "recommended_for": "軽量・高品質小型モデル",
        "tags": ["fast", "small"],
    },

    # -----------------------------------------------------------------------
    # Light — ~2–3 GB
    # -----------------------------------------------------------------------
    {
        "id": "llama-3.2-3b",
        "name": "Llama 3.2 3B Instruct",
        "repo_id": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "filename": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "size_gb": 2.0,
        "description": "Meta の Llama 3.2 3B。高速かつ実用的な品質。",
        "recommended_for": "軽いタスク・低メモリ使用",
        "tags": ["fast", "small"],
    },
    {
        "id": "phi-3.5-mini",
        "name": "Phi-3.5 Mini Instruct",
        "repo_id": "bartowski/Phi-3.5-mini-instruct-GGUF",
        "filename": "Phi-3.5-mini-instruct-Q4_K_M.gguf",
        "size_gb": 2.2,
        "description": "Microsoft の Phi-3.5 Mini。サイズに対して驚異的な性能。",
        "recommended_for": "低メモリ・高速応答",
        "tags": ["fast", "small"],
    },

    # -----------------------------------------------------------------------
    # Medium-small — ~4–5 GB
    # -----------------------------------------------------------------------
    {
        "id": "mistral-7b",
        "name": "Mistral 7B Instruct v0.3",
        "repo_id": "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
        "filename": "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
        "size_gb": 4.4,
        "description": "Mistral の主力 7B 命令モデル。安定した汎用性能。",
        "recommended_for": "汎用目的・安定した性能",
        "tags": ["balanced"],
    },
    {
        "id": "qwen2.5-coder-7b",
        "name": "Qwen2.5 Coder 7B Instruct",
        "repo_id": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        "filename": "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        "size_gb": 4.4,
        "description": "Alibaba のコード特化 7B モデル。プログラミングに強い。",
        "recommended_for": "コード生成・プログラミング特化",
        "tags": ["code", "recommended"],
    },
    {
        "id": "llama-3.1-8b",
        "name": "Llama 3.1 8B Instruct",
        "repo_id": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "filename": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "size_gb": 4.9,
        "description": "Meta の Llama 3.1 8B。汎用性が高く高品質。最も安定した選択肢。",
        "recommended_for": "汎用コードタスク・バランス重視",
        "tags": ["balanced", "recommended"],
    },
    {
        "id": "deepseek-r1-distill-llama-8b",
        "name": "DeepSeek-R1 Distill Llama 8B",
        "repo_id": "bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF",
        "filename": "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf",
        "size_gb": 4.9,
        "description": "DeepSeek-R1 の推論能力を 8B モデルに蒸留。論理・コード推論に優秀。",
        "recommended_for": "推論・論理タスク・コードデバッグ",
        "tags": ["reasoning", "code"],
    },

    # -----------------------------------------------------------------------
    # Medium — ~5–8 GB
    # -----------------------------------------------------------------------
    {
        "id": "gemma-2-9b",
        "name": "Gemma 2 9B Instruct",
        "repo_id": "bartowski/gemma-2-9b-it-GGUF",
        "filename": "gemma-2-9b-it-Q4_K_M.gguf",
        "size_gb": 5.4,
        "description": "Google の Gemma 2 9B。文書理解・要約タスクに優秀。",
        "recommended_for": "文書理解・要約・汎用",
        "tags": ["balanced"],
    },
    {
        "id": "mistral-nemo-12b",
        "name": "Mistral Nemo 12B Instruct",
        "repo_id": "bartowski/Mistral-Nemo-Instruct-2407-GGUF",
        "filename": "Mistral-Nemo-Instruct-2407-Q4_K_M.gguf",
        "size_gb": 7.2,
        "description": "Mistral と NVIDIA 共同開発の 12B モデル。長いコンテキストに強い。",
        "recommended_for": "長文コンテキスト・汎用高精度",
        "tags": ["balanced"],
    },

    # -----------------------------------------------------------------------
    # Medium-large — ~8–10 GB
    # -----------------------------------------------------------------------
    {
        "id": "phi-4-14b",
        "name": "Phi-4 14B Instruct",
        "repo_id": "bartowski/phi-4-GGUF",
        "filename": "phi-4-Q4_K_M.gguf",
        "size_gb": 8.5,
        "description": "Microsoft の最新 Phi-4 14B。数学・推論・コードで最高クラスの性能。",
        "recommended_for": "数学・推論・複雑なコード",
        "tags": ["reasoning", "code", "recommended"],
    },
    {
        "id": "qwen2.5-14b",
        "name": "Qwen2.5 14B Instruct",
        "repo_id": "bartowski/Qwen2.5-14B-Instruct-GGUF",
        "filename": "Qwen2.5-14B-Instruct-Q4_K_M.gguf",
        "size_gb": 8.8,
        "description": "Alibaba の Qwen2.5 14B。多言語対応で汎用性が非常に高い。",
        "recommended_for": "多言語・汎用高精度",
        "tags": ["balanced", "recommended"],
    },
    {
        "id": "qwen2.5-coder-14b",
        "name": "Qwen2.5 Coder 14B Instruct",
        "repo_id": "bartowski/Qwen2.5-Coder-14B-Instruct-GGUF",
        "filename": "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf",
        "size_gb": 8.8,
        "description": "Qwen2.5 のコード特化 14B 版。大規模コードベースの理解・生成に最適。",
        "recommended_for": "大規模コード生成・コードレビュー",
        "tags": ["code", "recommended"],
    },
    {
        "id": "deepseek-r1-distill-qwen-14b",
        "name": "DeepSeek-R1 Distill Qwen 14B",
        "repo_id": "bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF",
        "filename": "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
        "size_gb": 8.8,
        "description": "DeepSeek-R1 の推論能力を 14B モデルに蒸留。Chain-of-Thought 思考。",
        "recommended_for": "高度な推論・複雑なアルゴリズム",
        "tags": ["reasoning", "code"],
    },
    {
        "id": "deepseek-coder-v2-lite",
        "name": "DeepSeek Coder V2 Lite Instruct",
        "repo_id": "bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF",
        "filename": "DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf",
        "size_gb": 9.0,
        "description": "DeepSeek のコード特化 MoE モデル（16B 実効 2.4B）。コード品質最高クラス。",
        "recommended_for": "複雑なコード生成・バグ修正",
        "tags": ["code"],
    },

    # -----------------------------------------------------------------------
    # Large — ~11–15 GB
    # -----------------------------------------------------------------------
    {
        "id": "gemma-2-27b",
        "name": "Gemma 2 27B Instruct",
        "repo_id": "bartowski/gemma-2-27b-it-GGUF",
        "filename": "gemma-2-27b-it-Q3_K_M.gguf",
        "size_gb": 11.8,
        "description": "Google の Gemma 2 最大版（Q3_K_M）。高品質な自然言語理解。",
        "recommended_for": "高品質な文書生成・複雑な指示理解",
        "tags": ["balanced"],
    },
    {
        "id": "mistral-small-22b",
        "name": "Mistral Small 22B Instruct",
        "repo_id": "bartowski/Mistral-Small-Instruct-2409-GGUF",
        "filename": "Mistral-Small-Instruct-2409-Q4_K_M.gguf",
        "size_gb": 13.0,
        "description": "Mistral の 22B モデル。Mistral 7B より大幅に賢く、コスト効率も高い。",
        "recommended_for": "高精度汎用・コード・分析",
        "tags": ["balanced", "recommended"],
    },
    {
        "id": "qwen2.5-32b",
        "name": "Qwen2.5 32B Instruct",
        "repo_id": "bartowski/Qwen2.5-32B-Instruct-GGUF",
        "filename": "Qwen2.5-32B-Instruct-Q3_K_M.gguf",
        "size_gb": 14.4,
        "description": "Alibaba の Qwen2.5 32B（Q3_K_M）。多言語・複雑な推論に非常に強い。",
        "recommended_for": "最高精度の汎用タスク・多言語",
        "tags": ["balanced"],
    },
    {
        "id": "qwen2.5-coder-32b",
        "name": "Qwen2.5 Coder 32B Instruct",
        "repo_id": "bartowski/Qwen2.5-Coder-32B-Instruct-GGUF",
        "filename": "Qwen2.5-Coder-32B-Instruct-Q3_K_M.gguf",
        "size_gb": 14.4,
        "description": "Qwen2.5 のコード特化 32B 版（Q3_K_M）。OSS 最高クラスのコード生成能力。",
        "recommended_for": "OSS 最高水準のコード生成",
        "tags": ["code", "recommended"],
    },

    # -----------------------------------------------------------------------
    # Very large — ~15–19 GB
    # -----------------------------------------------------------------------
    {
        "id": "mixtral-8x7b",
        "name": "Mixtral 8x7B Instruct v0.1",
        "repo_id": "bartowski/Mixtral-8x7B-Instruct-v0.1-GGUF",
        "filename": "Mixtral-8x7B-Instruct-v0.1-Q2_K.gguf",
        "size_gb": 15.6,
        "description": "Mistral の Mixture-of-Experts 47B モデル（Q2_K）。実行時 13B 相当のメモリ使用。",
        "recommended_for": "MoE 高速推論・多様な専門タスク",
        "tags": ["balanced"],
    },
    {
        "id": "llama-3.3-70b",
        "name": "Llama 3.3 70B Instruct",
        "repo_id": "bartowski/Llama-3.3-70B-Instruct-GGUF",
        "filename": "Llama-3.3-70B-Instruct-Q2_K.gguf",
        "size_gb": 18.5,
        "description": "Meta の最新 Llama 3.3 70B（Q2_K）。OSS 最高峰の汎用性能。32GB RAM で動作可能。",
        "recommended_for": "最高品質の汎用タスク・32GB RAM 必須",
        "tags": ["balanced", "large"],
    },
]


def get_model_dest_dir(model_id: str) -> Path:
    """モデルの保存ディレクトリパスを返す。"""
    return MODELS_DIR / model_id


def get_model_dest_path(model_id: str, filename: str) -> Path:
    """モデルファイルの保存パスを返す。"""
    return get_model_dest_dir(model_id) / filename


def get_download_url(repo_id: str, filename: str) -> str:
    """HuggingFace ダイレクトダウンロード URL を返す。"""
    return _HF_DOWNLOAD_URL.format(repo_id=repo_id, filename=filename)


def get_catalog_model(model_id: str) -> Optional[Dict]:
    """カタログから指定 ID のモデル情報を返す。"""
    for m in HF_MODEL_CATALOG:
        if m["id"] == model_id:
            return m
    return None


def scan_local_models() -> List[Dict]:
    """~/.localforge/models/ 以下の GGUF ファイルを検索してリストを返す。"""
    result = []
    if not MODELS_DIR.exists():
        return result

    for gguf_path in sorted(MODELS_DIR.rglob("*.gguf")):
        size_bytes = gguf_path.stat().st_size
        size_gb = round(size_bytes / (1024 ** 3), 2)
        model_id = gguf_path.parent.name

        # カタログと照合してメタデータを補完
        catalog_entry = get_catalog_model(model_id)
        result.append({
            "id": model_id,
            "name": catalog_entry["name"] if catalog_entry else gguf_path.stem,
            "path": str(gguf_path),
            "filename": gguf_path.name,
            "size_gb": size_gb,
            "description": catalog_entry["description"] if catalog_entry else "ローカルモデル",
            "recommended_for": catalog_entry.get("recommended_for", "") if catalog_entry else "",
            "tags": catalog_entry.get("tags", []) if catalog_entry else [],
            "in_catalog": catalog_entry is not None,
        })

    return result


def get_catalog_with_status() -> List[Dict]:
    """カタログモデルにダウンロード済みかどうかのステータスを付与して返す。"""
    local = {m["path"]: m for m in scan_local_models()}
    local_by_filename = {}
    for m in scan_local_models():
        local_by_filename[m["filename"]] = m

    result = []
    for model in HF_MODEL_CATALOG:
        dest_path = get_model_dest_path(model["id"], model["filename"])
        downloaded = dest_path.is_file()
        entry = dict(model)
        entry["downloaded"] = downloaded
        entry["local_path"] = str(dest_path) if downloaded else ""
        entry["download_url"] = get_download_url(model["repo_id"], model["filename"])
        entry["dest_dir"] = str(get_model_dest_dir(model["id"]))
        result.append(entry)

    return result


def download_model(
    model_id: str,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> str:
    """
    HuggingFace Hub 経由でモデルをダウンロードする。

    Args:
        model_id:    カタログの model ID
        progress_cb: (downloaded_bytes, total_bytes) を受け取るコールバック

    Returns:
        ダウンロードされたファイルの絶対パス

    Raises:
        ValueError:       モデル ID がカタログに存在しない
        RuntimeError:     ネットワーク/プロキシエラー（手動ダウンロードを促す）
        FileNotFoundError: huggingface-hub が未インストール
    """
    model = get_catalog_model(model_id)
    if model is None:
        raise ValueError(f"カタログに存在しないモデル ID です: {model_id}")

    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError
    except ImportError as exc:
        raise FileNotFoundError(
            "huggingface-hub がインストールされていません。"
            f" `pip install huggingface-hub` を実行してください。詳細: {exc}"
        ) from exc

    dest_dir = get_model_dest_dir(model_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / model["filename"]

    if dest_path.is_file():
        logger.info("モデルは既にダウンロード済みです: %s", dest_path)
        return str(dest_path)

    logger.info(
        "HuggingFace Hub からダウンロード中: %s/%s → %s",
        model["repo_id"], model["filename"], dest_dir,
    )

    try:
        downloaded_path = hf_hub_download(
            repo_id=model["repo_id"],
            filename=model["filename"],
            local_dir=str(dest_dir),
            local_dir_use_symlinks=False,
        )
        logger.info("ダウンロード完了: %s", downloaded_path)
        return str(downloaded_path)

    except (HfHubHTTPError, RepositoryNotFoundError, Exception) as exc:
        exc_str = str(exc)
        # プロキシ/ネットワークエラーの判定
        proxy_keywords = ["ProxyError", "ConnectionError", "SSLError", "Timeout",
                          "proxy", "403", "407", "firewall"]
        is_proxy = any(kw.lower() in exc_str.lower() for kw in proxy_keywords)
        hint = (
            "プロキシまたはネットワーク制限によりダウンロードをブロックされました。"
            if is_proxy else
            f"ダウンロードに失敗しました: {exc_str}"
        )
        raise RuntimeError(hint) from exc


def get_manual_instructions(model_id: str) -> Dict:
    """
    手動ダウンロード用の詳細手順を返す。
    バックエンドが保存先フォルダを事前に作成する。

    Returns:
        {
          "model_name": str,
          "download_url": str,
          "dest_dir": str,
          "dest_path": str,
          "filename": str,
          "wget_cmd": str,
          "curl_cmd": str,
          "steps": List[str],
        }
    """
    model = get_catalog_model(model_id)
    if model is None:
        raise ValueError(f"カタログに存在しないモデル ID です: {model_id}")

    dest_dir = get_model_dest_dir(model_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / model["filename"]
    url = get_download_url(model["repo_id"], model["filename"])

    return {
        "model_name": model["name"],
        "download_url": url,
        "dest_dir": str(dest_dir),
        "dest_path": str(dest_path),
        "filename": model["filename"],
        "size_gb": model["size_gb"],
        "wget_cmd": f'wget -O "{dest_path}" "{url}"',
        "curl_cmd": f'curl -L -o "{dest_path}" "{url}"',
        "steps": [
            f'1. 以下の URL をブラウザでアクセスしてファイルをダウンロードしてください:',
            f'   {url}',
            f'2. または、ターミナルで以下のコマンドを実行してください:',
            f'   wget -O "{dest_path}" "{url}"',
            f'   （curl の場合）',
            f'   curl -L -o "{dest_path}" "{url}"',
            f'3. ダウンロードしたファイルを以下のフォルダに配置してください:',
            f'   {dest_dir}',
            f'4. ファイル名が "{model["filename"]}" であることを確認してください。',
            f'5. アプリに戻り「ファイルを確認する」ボタンをクリックしてください。',
        ],
    }
