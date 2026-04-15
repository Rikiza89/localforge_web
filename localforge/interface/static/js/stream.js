/**
 * stream.js — EventSourceラッパー・ストリーミングヘルパー
 * SSEストリームを受信して各ペイロードタイプに応じた処理を行う。
 */

"use strict";

/**
 * SSEストリームを開始する。
 * @param {string} url - SSEエンドポイントURL
 * @param {HTMLElement|null} outputEl - トークンを追記する出力要素（nullも可）
 * @param {Object} handlers - イベントハンドラーのマップ
 *   - onToken(token): テキストトークン受信時
 *   - onSection(name): セクションヘッダー受信時
 *   - onFileWritten(path): ファイル書き込み完了時
 *   - onProgress(done, total, currentFile): 進捗更新時
 *   - onDone(): ストリーム完了時
 *   - onError(message): エラー受信時
 * @returns {EventSource} 開いたEventSourceインスタンス
 */
function startStream(url, outputEl, handlers) {
  const es = new EventSource(url);

  es.onmessage = (event) => {
    let data;
    try {
      data = JSON.parse(event.data);
    } catch (e) {
      console.warn("SSEデータのJSONパースエラー:", event.data);
      return;
    }

    if (data.done) {
      es.close();
      if (handlers.onDone) handlers.onDone();
      return;
    }

    if (data.error) {
      es.close();
      if (handlers.onError) handlers.onError(data.error);
      return;
    }

    if (data.token !== undefined) {
      if (outputEl) {
        outputEl.textContent += data.token;
        outputEl.scrollTop = outputEl.scrollHeight;
      }
      if (handlers.onToken) handlers.onToken(data.token);
      return;
    }

    if (data.section !== undefined) {
      if (handlers.onSection) handlers.onSection(data.section);
      return;
    }

    if (data.file_written !== undefined) {
      if (handlers.onFileWritten) handlers.onFileWritten(data.file_written);
      return;
    }

    if (data.progress !== undefined) {
      const { done, total, current_file } = data.progress;
      if (handlers.onProgress) handlers.onProgress(done, total, current_file || "");
      return;
    }

    if (data.status !== undefined) {
      updateStatusBar(data.status);
      return;
    }
  };

  es.onerror = () => {
    es.close();
    if (handlers.onError) handlers.onError("SSE接続エラーが発生しました。");
  };

  return es;
}

/**
 * POSTリクエストでSSEストリームを開始する（EventSourceはGETのみのため代替実装）。
 * フロントエンドからPOSTペイロードを送信してSSEを受け取るためにfetch+ReadableStreamを使用する。
 *
 * @param {string} url - SSEエンドポイントURL
 * @param {Object} body - POSTリクエストのボディ
 * @param {HTMLElement|null} outputEl - トークンを追記する出力要素
 * @param {Object} handlers - イベントハンドラーのマップ（startStreamと同じ）
 * @returns {Promise<void>}
 */
async function startPostStream(url, body, outputEl, handlers) {
  let aborted = false;
  const controller = new AbortController();

  // キャンセル関数を返す
  const cancel = () => {
    aborted = true;
    controller.abort();
  };

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal,
    });

    if (!response.ok) {
      const errText = await response.text();
      if (handlers.onError) handlers.onError(`HTTP ${response.status}: ${errText}`);
      return cancel;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (!aborted) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const rawData = line.slice(6).trim();
        if (!rawData) continue;

        let data;
        try {
          data = JSON.parse(rawData);
        } catch (e) {
          console.warn("SSEデータのJSONパースエラー:", rawData);
          continue;
        }

        if (data.done) {
          if (handlers.onDone) handlers.onDone();
          return cancel;
        }
        if (data.error) {
          if (handlers.onError) handlers.onError(data.error);
          return cancel;
        }
        if (data.token !== undefined) {
          if (outputEl) {
            outputEl.textContent += data.token;
            outputEl.scrollTop = outputEl.scrollHeight;
          }
          if (handlers.onToken) handlers.onToken(data.token);
        }
        if (data.section !== undefined && handlers.onSection) {
          handlers.onSection(data.section);
        }
        if (data.file_written !== undefined && handlers.onFileWritten) {
          handlers.onFileWritten(data.file_written);
        }
        if (data.progress !== undefined && handlers.onProgress) {
          const { done: d, total, current_file } = data.progress;
          handlers.onProgress(d, total, current_file || "");
        }
      }
    }
  } catch (err) {
    if (!aborted && handlers.onError) {
      handlers.onError(`ストリームエラー: ${err.message}`);
    }
  }

  return cancel;
}

/**
 * ステータスバーのメッセージを更新する。
 * @param {string} message - 表示するメッセージ
 */
function updateStatusBar(message) {
  const el = document.getElementById("status-message");
  if (el) el.textContent = message;
}

/**
 * アラートバナーを表示する。
 * @param {string} message - 表示するメッセージ
 * @param {"error"|"success"|"warning"} type - アラートのタイプ
 * @param {number} [timeout=5000] - 自動消去のタイムアウト（ミリ秒、0で無効）
 */
function showAlert(message, type = "error", timeout = 5000) {
  const container = document.getElementById("alert-container");
  if (!container) return;

  const alert = document.createElement("div");
  alert.className = `alert alert-${type}`;
  alert.innerHTML = `
    <span class="alert-message">${escapeHtml(message)}</span>
    <button class="alert-close" title="閉じる">✕</button>
  `;

  const closeBtn = alert.querySelector(".alert-close");
  closeBtn.addEventListener("click", () => alert.remove());

  container.appendChild(alert);

  if (timeout > 0) {
    setTimeout(() => {
      if (alert.parentNode) alert.remove();
    }, timeout);
  }
}

/**
 * HTMLエスケープを行う。
 * @param {string} str - エスケープする文字列
 * @returns {string} エスケープ済み文字列
 */
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/**
 * APIエンドポイントにJSONリクエストを送信する。
 * @param {string} url - エンドポイントURL
 * @param {string} method - HTTPメソッド
 * @param {Object} [body] - リクエストボディ
 * @returns {Promise<Object>} レスポンスのJSONオブジェクト
 */
async function apiRequest(url, method = "GET", body = null) {
  const opts = {
    method,
    headers: { "Content-Type": "application/json" },
  };
  if (body) opts.body = JSON.stringify(body);

  const response = await fetch(url, opts);
  const json = await response.json();

  if (!response.ok) {
    throw new Error(json.message || `HTTP ${response.status}`);
  }

  return json;
}
