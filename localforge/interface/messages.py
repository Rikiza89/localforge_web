"""
Server-side i18n message map for error/status strings returned by API routes.
Keys are stable identifiers; values are dicts keyed by language code.
"""

from __future__ import annotations

_MESSAGES: dict[str, dict[str, str]] = {
    "no_project": {
        "en": "No project is open",
        "ja": "プロジェクトが開かれていません",
        "it": "Nessun progetto aperto",
    },
    "no_folder_selected": {
        "en": "No folder was selected",
        "ja": "フォルダが選択されませんでした",
        "it": "Nessuna cartella selezionata",
    },
    "invalid_path": {
        "en": "The specified path is not a directory",
        "ja": "指定されたパスはディレクトリではありません",
        "it": "Il percorso specificato non è una cartella",
    },
    "no_model": {
        "en": "No model selected. Please select a model in the UI",
        "ja": "モデルが選択されていません。UIでモデルを選択してください",
        "it": "Nessun modello selezionato. Seleziona un modello nell'interfaccia",
    },
    "no_prompt": {
        "en": "No prompt specified",
        "ja": "プロンプトが指定されていません",
        "it": "Nessun prompt specificato",
    },
    "no_plan": {
        "en": "No approved plan found",
        "ja": "承認済みプランが見つかりません",
        "it": "Nessun piano approvato trovato",
    },
    "no_question": {
        "en": "No question specified",
        "ja": "質問が指定されていません",
        "it": "Nessuna domanda specificata",
    },
    "no_model_name": {
        "en": "No model name specified",
        "ja": "モデル名が指定されていません",
        "it": "Nome modello non specificato",
    },
    "no_path": {
        "en": "No path specified",
        "ja": "パスが指定されていません",
        "it": "Nessun percorso specificato",
    },
    "no_content": {
        "en": "No content specified",
        "ja": "コンテンツが指定されていません",
        "it": "Nessun contenuto specificato",
    },
    "access_denied": {
        "en": "Access outside project root is forbidden",
        "ja": "プロジェクトルート外へのアクセスは禁止されています",
        "it": "Accesso fuori dalla root del progetto non consentito",
    },
    "file_not_found": {
        "en": "File not found",
        "ja": "ファイルが見つかりません",
        "it": "File non trovato",
    },
    "invalid_language": {
        "en": "Invalid language code. Use: en, ja, it",
        "ja": "無効な言語コードです。使用可能: en, ja, it",
        "it": "Codice lingua non valido. Usa: en, ja, it",
    },
    "no_index": {
        "en": "No index found. Build the index first.",
        "ja": "インデックスが見つかりません。先にインデックスを構築してください。",
        "it": "Nessun indice trovato. Costruisci prima l'indice.",
    },
    "ollama_unavailable": {
        "en": "Cannot connect to Ollama server. Make sure Ollama is running.",
        "ja": "Ollamaサーバーに接続できません。Ollamaが起動しているか確認してください。",
        "it": "Impossibile connettersi al server Ollama. Assicurati che Ollama sia in esecuzione.",
    },
    "paths_must_be_list": {
        "en": "paths must be a list",
        "ja": "pathsはリストである必要があります",
        "it": "paths deve essere una lista",
    },
    "num_thread_invalid": {
        "en": "num_thread must be an integer >= 1",
        "ja": "num_threadは1以上の整数を指定してください",
        "it": "num_thread deve essere un intero >= 1",
    },
    "report_not_found": {
        "en": "The specified report was not found",
        "ja": "指定されたレポートが見つかりません",
        "it": "Il report specificato non è stato trovato",
    },
}

_SUPPORTED = {"en", "ja", "it"}


def msg(language: str, key: str, **kwargs: object) -> str:
    """Return the translated message for key in the given language."""
    lang = language if language in _SUPPORTED else "en"
    text = _MESSAGES.get(key, {}).get(lang) or _MESSAGES.get(key, {}).get("en") or key
    if kwargs:
        try:
            text = text.format(**kwargs)
        except (KeyError, ValueError):
            pass
    return text
