/**
 * stream.js — EventSourceラッパー・ストリーミングヘルパー
 * SSEストリームを受信して各ペイロードタイプに応じた処理を行う。
 */

"use strict";

// =========================================================================
// Ollamaライブ出力パネル
// =========================================================================

/**
 * スクロール位置がボトムから threshold px 以内かどうかを返す。
 * @param {HTMLElement} el
 * @param {number} threshold
 * @returns {boolean}
 */
function _isNearBottom(el, threshold = 80) {
  return el.scrollHeight - el.scrollTop - el.clientHeight <= threshold;
}

/**
 * ユーザーが既にボトム付近にいる場合のみスクロールする。
 * 手動でスクロールアップ中の場合は何もしない。
 * @param {HTMLElement} el
 */
function _autoScroll(el) {
  // スクロール可能な場合のみ実行
  if (el.scrollHeight > el.clientHeight) {
    if (_isNearBottom(el)) {
      el.scrollTop = el.scrollHeight;
    }
  }
}

const OllamaPanel = (() => {
  let _inThinkBlock = false;
  let _thinkingVisible = false;
  let _normalBuf = "";
  let _thinkBuf = "";

  function _normalEl() { return document.getElementById("ollama-output-normal"); }
  function _thinkingEl() { return document.getElementById("ollama-output-thinking"); }
  function _thinkingContent() { return document.getElementById("ollama-thinking-content"); }
  function _panel() { return document.getElementById("ollama-panel"); }

  function appendToken(token) {
    const panel = _panel();
    if (!panel) return;

    panel.classList.add("streaming");

    // <think>...</think> タグ検出
    let remaining = token;
    while (remaining.length > 0) {
      if (!_inThinkBlock) {
        const thinkStart = remaining.indexOf("<think>");
        if (thinkStart === -1) {
          _normalBuf += remaining;
          const el = _normalEl();
          if (el) { el.innerHTML = _renderMd(_normalBuf); _autoScroll(el); }
          break;
        }
        const before = remaining.slice(0, thinkStart);
        if (before) {
          _normalBuf += before;
          const el = _normalEl();
          if (el) { el.innerHTML = _renderMd(_normalBuf); _autoScroll(el); }
        }
        _inThinkBlock = true;
        remaining = remaining.slice(thinkStart + "<think>".length);
      } else {
        const thinkEnd = remaining.indexOf("</think>");
        if (thinkEnd === -1) {
          _thinkBuf += remaining;
          const el = _thinkingContent();
          if (el) { el.innerHTML = _renderMd(_thinkBuf); _autoScroll(el.parentElement); }
          break;
        }
        const thinkText = remaining.slice(0, thinkEnd);
        if (thinkText) {
          _thinkBuf += thinkText;
          const el = _thinkingContent();
          if (el) { el.innerHTML = _renderMd(_thinkBuf); _autoScroll(el.parentElement); }
        }
        _inThinkBlock = false;
        remaining = remaining.slice(thinkEnd + "</think>".length);
      }
    }
  }

  function markDone() {
    const panel = _panel();
    if (panel) panel.classList.remove("streaming");
    _inThinkBlock = false;
  }

  function clear() {
    _normalBuf = "";
    _thinkBuf = "";
    const normal = _normalEl();
    const thinking = _thinkingContent();
    if (normal) normal.innerHTML = "";
    if (thinking) thinking.innerHTML = "";
    _inThinkBlock = false;
    markDone();
  }

  function toggleThinking() {
    _thinkingVisible = !_thinkingVisible;
    const el = _thinkingEl();
    const btn = document.getElementById("ollama-thinking-toggle");
    if (el) el.classList.toggle("hidden", !_thinkingVisible);
    if (btn) btn.textContent = _thinkingVisible ? "思考を隠す" : "思考を表示";
  }

  function init() {
    const toggleBtn = document.getElementById("ollama-panel-toggle");
    const panel = _panel();
    if (toggleBtn && panel) {
      toggleBtn.addEventListener("click", () => {
        panel.classList.toggle("collapsed");
      });
    }
    const thinkingBtn = document.getElementById("ollama-thinking-toggle");
    if (thinkingBtn) thinkingBtn.addEventListener("click", toggleThinking);
    const clearBtn = document.getElementById("ollama-clear-btn");
    if (clearBtn) clearBtn.addEventListener("click", clear);
  }

  return { appendToken, markDone, clear, init };
})();

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
  let es = null;
  let _closed = false;
  let _idleTimer = null;
  // 5分 — モデルロード時間を考慮した余裕のあるタイムアウト。
  // スレッドベースのハートビート（15秒間隔）が正常に機能すれば実際には発火しない。
  const IDLE_TIMEOUT = 300000;

  function _resetIdle() {
    if (_idleTimer) clearTimeout(_idleTimer);
    _idleTimer = setTimeout(() => {
      if (!_closed) {
        console.warn("SSEアイドルタイムアウト — 再接続します");
        es.close();
        es = _open();
      }
    }, IDLE_TIMEOUT);
  }

  function _dispatch(data) {
    _resetIdle();

    if (data.heartbeat) return;

    if (data.raw_token !== undefined) {
      OllamaPanel.appendToken(data.raw_token);
      return;
    }

    if (data.done) {
      _closed = true;
      if (_idleTimer) clearTimeout(_idleTimer);
      es.close();
      OllamaPanel.markDone();
      if (handlers.onDone) handlers.onDone();
      return;
    }

    if (data.error) {
      _closed = true;
      if (_idleTimer) clearTimeout(_idleTimer);
      es.close();
      OllamaPanel.markDone();
      if (handlers.onError) handlers.onError(data.error);
      return;
    }

    if (data.token !== undefined) {
      if (outputEl) {
        outputEl.textContent += data.token;
        _autoScroll(outputEl);
      }
      if (handlers.onToken) handlers.onToken(data.token);
      return;
    }

    if (data.section !== undefined) {
      if (handlers.onSection) handlers.onSection(data.section, data.section_idx, data.section_total);
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

    if (data.checkpoint !== undefined) {
      if (handlers.onCheckpoint) handlers.onCheckpoint(data.checkpoint);
      return;
    }

    if (data.warning !== undefined) {
      if (handlers.onWarning) handlers.onWarning(data.warning);
      return;
    }
  }

  function _open() {
    const source = new EventSource(url);
    source.onmessage = (event) => {
      let data;
      try { data = JSON.parse(event.data); }
      catch (e) { console.warn("SSEデータのJSONパースエラー:", event.data); return; }
      _dispatch(data);
    };
    source.onerror = () => {
      source.close();
      if (!_closed && handlers.onError) handlers.onError("SSE接続エラーが発生しました。");
    };
    _resetIdle();
    return source;
  }

  es = _open();
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

        if (data.heartbeat) continue;

        if (data.raw_token !== undefined) {
          OllamaPanel.appendToken(data.raw_token);
          continue;
        }

        if (data.done) {
          OllamaPanel.markDone();
          if (handlers.onDone) handlers.onDone();
          return cancel;
        }
        if (data.error) {
          OllamaPanel.markDone();
          if (handlers.onError) handlers.onError(data.error);
          return cancel;
        }
        if (data.token !== undefined) {
          if (outputEl) {
            outputEl.textContent += data.token;
            _autoScroll(outputEl);
          }
          if (handlers.onToken) handlers.onToken(data.token);
        }
        if (data.section !== undefined && handlers.onSection) {
          handlers.onSection(data.section, data.section_idx, data.section_total);
        }
        if (data.file_written !== undefined && handlers.onFileWritten) {
          handlers.onFileWritten(data.file_written);
        }
        if (data.progress !== undefined && handlers.onProgress) {
          const { done: d, total, current_file } = data.progress;
          handlers.onProgress(d, total, current_file || "");
        }
        if (data.status !== undefined) {
          updateStatusBar(data.status);
        }
        if (data.checkpoint !== undefined && handlers.onCheckpoint) {
          handlers.onCheckpoint(data.checkpoint);
        }
        if (data.warning !== undefined && handlers.onWarning) {
          handlers.onWarning(data.warning);
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
 * MarkdownテキストをHTMLにレンダリングする。
 * marked.jsが利用可能な場合は変換し、なければエスケープしてpreで囲む。
 * @param {string} text
 * @returns {string} HTML文字列
 */
function _renderMd(text) {
  if (typeof marked !== "undefined") {
    return marked.parse(text || "");
  }
  return "<pre>" + escapeHtml(text || "") + "</pre>";
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
    throw new Error(json.error || json.message || `HTTP ${response.status}`);
  }

  return json;
}
