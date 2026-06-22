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
  // rAF debounce state — avoids re-rendering full markdown on every token
  let _normalDirty = false;
  let _thinkDirty = false;
  let _rafPending = false;

  // インクリメンタルレンダリング状態:
  // バッファ全体を毎フレーム marked.parse() すると長いストリームで O(n²) になる。
  // 確定領域（最後の段落境界まで、コードフェンス外）は一度だけパースして HTML を
  // キャッシュし、末尾の未確定領域のみ毎フレーム再パースする。
  // markDone() で全体を一度フルパースして最終整合を取る。
  const _incr = {
    normal: { stableLen: 0, stableHtml: "" },
    think: { stableLen: 0, stableHtml: "" },
  };

  function _renderIncremental(buf, state) {
    const boundary = buf.lastIndexOf("\n\n");
    if (boundary + 2 > state.stableLen && boundary >= 0) {
      const candidate = buf.slice(0, boundary + 2);
      // 開いたコードフェンス内では確定境界を進めない（分割パースで壊れるため）
      const fenceCount = (candidate.match(/```/g) || []).length;
      if (fenceCount % 2 === 0) {
        state.stableHtml += _renderMd(buf.slice(state.stableLen, boundary + 2));
        state.stableLen = boundary + 2;
      }
    }
    return state.stableHtml + _renderMd(buf.slice(state.stableLen));
  }

  function _normalEl() { return document.getElementById("ollama-output-normal"); }
  function _thinkingEl() { return document.getElementById("ollama-output-thinking"); }
  function _thinkingContent() { return document.getElementById("ollama-thinking-content"); }
  function _panel() { return document.getElementById("ollama-panel"); }

  function _flushDirty(final) {
    if (_normalDirty) {
      _normalDirty = false;
      const el = _normalEl();
      if (el) {
        el.innerHTML = final
          ? _renderMd(_normalBuf)
          : _renderIncremental(_normalBuf, _incr.normal);
        _autoScroll(el);
      }
    }
    if (_thinkDirty) {
      _thinkDirty = false;
      const el = _thinkingContent();
      if (el) {
        el.innerHTML = final
          ? _renderMd(_thinkBuf)
          : _renderIncremental(_thinkBuf, _incr.think);
        _autoScroll(el.parentElement);
      }
    }
  }

  function _scheduleFlush() {
    if (_rafPending) return;
    _rafPending = true;
    requestAnimationFrame(() => {
      _rafPending = false;
      _flushDirty(false);
    });
  }

  function _flushNow() {
    _rafPending = false;
    // 最終フラッシュ: 全体を一度フルパースして分割パースの境界ズレを解消する
    _normalDirty = _normalBuf.length > 0;
    _thinkDirty = _thinkBuf.length > 0;
    _flushDirty(true);
  }

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
          _normalDirty = true;
          break;
        }
        const before = remaining.slice(0, thinkStart);
        if (before) {
          _normalBuf += before;
          _normalDirty = true;
        }
        _inThinkBlock = true;
        remaining = remaining.slice(thinkStart + "<think>".length);
      } else {
        const thinkEnd = remaining.indexOf("</think>");
        if (thinkEnd === -1) {
          _thinkBuf += remaining;
          _thinkDirty = true;
          break;
        }
        const thinkText = remaining.slice(0, thinkEnd);
        if (thinkText) {
          _thinkBuf += thinkText;
          _thinkDirty = true;
        }
        _inThinkBlock = false;
        remaining = remaining.slice(thinkEnd + "</think>".length);
      }
    }
    if (_normalDirty || _thinkDirty) _scheduleFlush();
  }

  function markDone() {
    _flushNow();
    const panel = _panel();
    if (panel) panel.classList.remove("streaming");
    _inThinkBlock = false;
  }

  function clear() {
    _normalBuf = "";
    _thinkBuf = "";
    _normalDirty = false;
    _thinkDirty = false;
    _rafPending = false;
    _incr.normal.stableLen = 0;
    _incr.normal.stableHtml = "";
    _incr.think.stableLen = 0;
    _incr.think.stableHtml = "";
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

  function _switchInnerTab(tabName) {
    document.querySelectorAll(".ollama-inner-tab").forEach(btn => {
      btn.classList.toggle("active", btn.dataset.innerTab === tabName);
    });
    document.querySelectorAll(".ollama-inner-content").forEach(content => {
      const id = content.id.replace("inner-content-", "");
      content.classList.toggle("hidden", id !== tabName);
    });
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

    // 内タブ切り替え
    document.querySelectorAll(".ollama-inner-tab").forEach(btn => {
      btn.addEventListener("click", () => _switchInnerTab(btn.dataset.innerTab));
    });
  }

  return { appendToken, markDone, clear, init };
})();

// =========================================================================
// プロセスログ — フェーズタイムラインとプロンプトプレビュー
// =========================================================================

const ProcessLog = (() => {
  let _phaseEntries = [];
  let _activeEntry = null;

  function _listEl() { return document.getElementById("process-phase-list"); }
  function _badgeEl() { return document.getElementById("process-log-badge"); }
  function _previewDetails() { return document.getElementById("prompt-preview-details"); }
  function _previewContent() { return document.getElementById("prompt-preview-content"); }
  function _tokenBadge() { return document.getElementById("prompt-token-badge"); }

  function _now() {
    const d = new Date();
    return `${String(d.getHours()).padStart(2,"0")}:${String(d.getMinutes()).padStart(2,"0")}:${String(d.getSeconds()).padStart(2,"0")}`;
  }

  function _renderEntry(entry) {
    const el = document.createElement("div");
    el.className = "process-phase-entry phase-active";
    el.dataset.phaseId = entry.id;

    const iconEl = document.createElement("span");
    iconEl.className = "phase-icon spinning";
    iconEl.textContent = "⟳";

    const body = document.createElement("div");
    body.className = "phase-body";

    const name = document.createElement("div");
    name.className = "phase-name";
    name.textContent = entry.name;

    const detail = document.createElement("div");
    detail.className = "phase-detail";
    detail.textContent = entry.detail || "";

    const timeEl = document.createElement("span");
    timeEl.className = "phase-time";
    timeEl.textContent = entry.time;

    body.appendChild(name);
    body.appendChild(detail);
    el.appendChild(iconEl);
    el.appendChild(body);
    el.appendChild(timeEl);
    return el;
  }

  function addPhase(phaseName, detail) {
    const list = _listEl();
    if (!list) return;

    // 前のアクティブエントリを完了状態にする
    if (_activeEntry) {
      const prevEl = list.querySelector(`[data-phase-id="${_activeEntry.id}"]`);
      if (prevEl) {
        prevEl.classList.remove("phase-active");
        prevEl.classList.add("phase-done");
        const icon = prevEl.querySelector(".phase-icon");
        if (icon) { icon.classList.remove("spinning"); icon.textContent = "✓"; }
      }
    }

    const entry = { id: Date.now() + Math.random(), name: phaseName, detail: detail || "", time: _now() };
    _phaseEntries.push(entry);
    _activeEntry = entry;

    const el = _renderEntry(entry);
    list.appendChild(el);
    _autoScroll(list);

    // バッジを点滅させてプロセスログタブに通知
    const badge = _badgeEl();
    if (badge) badge.style.display = "inline";

    // パネルが開いていてプロセスログタブが非表示なら自動切替はしない（ユーザー操作を妨げない）
  }

  function markAllDone() {
    const list = _listEl();
    if (!list) return;
    list.querySelectorAll(".process-phase-entry.phase-active").forEach(el => {
      el.classList.remove("phase-active");
      el.classList.add("phase-done");
      const icon = el.querySelector(".phase-icon");
      if (icon) { icon.classList.remove("spinning"); icon.textContent = "✓"; }
    });
    _activeEntry = null;
    const badge = _badgeEl();
    if (badge) badge.style.display = "none";
  }

  function setPromptPreview(preview, tokens) {
    const details = _previewDetails();
    const content = _previewContent();
    const badge = _tokenBadge();
    if (details) details.style.display = "";
    if (content) content.textContent = preview || "";
    if (badge && tokens) badge.textContent = `~${tokens.toLocaleString()} tokens`;
  }

  function clear() {
    _phaseEntries = [];
    _activeEntry = null;
    const list = _listEl();
    if (list) list.innerHTML = "";
    const details = _previewDetails();
    if (details) details.style.display = "none";
    const content = _previewContent();
    if (content) content.textContent = "";
    const badge = _badgeEl();
    if (badge) badge.style.display = "none";
  }

  function init() {
    const clearBtn = document.getElementById("processlog-clear-btn");
    if (clearBtn) clearBtn.addEventListener("click", clear);
  }

  return { addPhase, markAllDone, setPromptPreview, clear, init };
})();

// ---------------------------------------------------------------------------
// トークンスループット / ETA 表示（ステータスバーの #status-tokens）
// ---------------------------------------------------------------------------
const TokenStats = (() => {
  let _count = 0;
  let _startTime = 0;
  let _lastRender = 0;
  let _progressDone = 0;
  let _progressTotal = 0;

  function _el() { return document.getElementById("status-tokens"); }

  function _fmt(seconds) {
    const s = Math.max(0, Math.round(seconds));
    const m = Math.floor(s / 60);
    return m > 0 ? `${m}分${s % 60}秒` : `${s}秒`;
  }

  function _render(now) {
    const el = _el();
    if (!el || !_startTime) return;
    const elapsed = (now - _startTime) / 1000;
    if (elapsed <= 0.5) return;
    const tps = (_count / elapsed).toFixed(1);
    let eta = "";
    // 複数ユニット（ファイル/セクション）の進捗があれば平均所要時間からETAを出す
    if (_progressTotal > 1 && _progressDone > 0 && _progressDone < _progressTotal) {
      const remain = (elapsed / _progressDone) * (_progressTotal - _progressDone);
      eta = ` ・残り目安 ${_fmt(remain)}`;
    }
    el.textContent = `⚡ ${tps} tok/s ・経過 ${_fmt(elapsed)}${eta}`;
  }

  function reset() {
    _count = 0;
    _startTime = 0;
    _lastRender = 0;
    _progressDone = 0;
    _progressTotal = 0;
    const el = _el();
    if (el) el.textContent = "";
  }

  function addToken() {
    const now = performance.now();
    if (!_startTime) _startTime = now;
    _count++;
    if (now - _lastRender > 500) {
      _lastRender = now;
      _render(now);
    }
  }

  function setProgress(done, total) {
    _progressDone = done;
    _progressTotal = total;
  }

  function done() {
    _render(performance.now());
  }

  return { reset, addToken, setProgress, done };
})();

// outputEl へのトークン追記を rAF でバッチ化する WeakMap ベースバッファ。
// textContent += を毎トークン呼ぶと O(n²) になるため、フレームごとに一括更新する。
const _tokenStreamBufs = new WeakMap();

function _appendTokenToEl(el, token) {
  if (!_tokenStreamBufs.has(el)) {
    _tokenStreamBufs.set(el, { text: el.textContent, pending: false });
  }
  const state = _tokenStreamBufs.get(el);
  state.text += token;
  if (!state.pending) {
    state.pending = true;
    requestAnimationFrame(() => {
      state.pending = false;
      el.textContent = state.text;
      _autoScroll(el);
    });
  }
}

/**
 * 単一のSSEペイロードを解析して適切なハンドラーに振り分ける共通ルーター。
 * "done" / "error" を返した場合は呼び出し元がストリームを終了する。
 * @param {Object} data - パース済みSSEペイロード
 * @param {Object} handlers - イベントハンドラーマップ
 * @param {HTMLElement|null} outputEl - トークン追記先要素
 * @returns {"done"|"error"|null}
 */
function _dispatchSseEvent(data, handlers, outputEl) {
  if (data.heartbeat) return null;

  if (data.raw_token !== undefined) {
    OllamaPanel.appendToken(data.raw_token);
    return null;
  }

  if (data.phase !== undefined) {
    ProcessLog.addPhase(data.phase, data.detail || "");
    return null;
  }

  if (data.prompt_preview !== undefined) {
    ProcessLog.setPromptPreview(data.prompt_preview, data.prompt_tokens);
    return null;
  }

  if (data.done) {
    OllamaPanel.markDone();
    ProcessLog.markAllDone();
    TokenStats.done();
    if (handlers.onDone) handlers.onDone();
    return "done";
  }

  if (data.error) {
    OllamaPanel.markDone();
    ProcessLog.markAllDone();
    TokenStats.done();
    if (handlers.onError) handlers.onError(data.error);
    return "error";
  }

  if (data.token !== undefined) {
    if (outputEl) { _appendTokenToEl(outputEl, data.token); }
    if (handlers.onToken) handlers.onToken(data.token);
    // Ollamaライブパネルは token イベントから直接給電する。
    // （以前はサーバーが全トークンを raw_token として二重送信していた —
    //   SSEペイロードを半減するため通常トークンの複製は廃止。
    //   思考トークンのみ raw_token イベントとして届く。）
    OllamaPanel.appendToken(data.token);
    TokenStats.addToken();
  }

  if (data.section !== undefined && handlers.onSection) {
    handlers.onSection(data.section, data.section_idx, data.section_total);
  }

  if (data.file_written !== undefined && handlers.onFileWritten) {
    handlers.onFileWritten(data.file_written);
  }

  if (data.progress !== undefined) {
    const { done: d, total, current_file } = data.progress;
    TokenStats.setProgress(d, total);
    if (handlers.onProgress) handlers.onProgress(d, total, current_file || "");
  }

  if (data.status !== undefined) updateStatusBar(data.status);

  if (data.checkpoint !== undefined && handlers.onCheckpoint) {
    handlers.onCheckpoint(data.checkpoint);
  }

  if (data.diff_preview !== undefined && handlers.onDiffPreview) {
    handlers.onDiffPreview(data.diff_preview, data.file_path);
  }

  if (data.warning !== undefined && handlers.onWarning) {
    handlers.onWarning(data.warning);
  }

  return null;
}

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
 *   - noReconnect(bool): trueの場合、アイドルタイムアウト時に再接続せずonErrorを呼ぶ
 * @returns {{close: function}} ストリームコントローラー（常に現在のESをclose()する）
 */
function startStream(url, outputEl, handlers) {
  TokenStats.reset();
  let es = null;
  let _closed = false;
  let _idleTimer = null;
  const _noReconnect = !!(handlers && handlers.noReconnect);
  // 5分 — モデルロード時間を考慮した余裕のあるタイムアウト。
  // スレッドベースのハートビート（15秒間隔）が正常に機能すれば実際には発火しない。
  const IDLE_TIMEOUT = 300000;

  // コントローラープロキシ — 常に現在の es を参照するため再接続後も有効
  const ctrl = {
    close() {
      _closed = true;
      if (_idleTimer) clearTimeout(_idleTimer);
      if (es) es.close();
    }
  };

  function _resetIdle() {
    if (_idleTimer) clearTimeout(_idleTimer);
    _idleTimer = setTimeout(() => {
      if (!_closed) {
        if (_noReconnect) {
          console.warn("SSEアイドルタイムアウト — noReconnectのため接続を終了します");
          _closed = true;
          if (es) es.close();
          if (handlers.onError) handlers.onError("接続がタイムアウトしました。再試行してください。");
        } else {
          console.warn("SSEアイドルタイムアウト — 再接続します");
          if (es) es.close();
          es = _open();
        }
      }
    }, IDLE_TIMEOUT);
  }

  function _dispatch(data) {
    if (_closed) return;
    _resetIdle();
    const result = _dispatchSseEvent(data, handlers, outputEl);
    if (result === "done" || result === "error") {
      _closed = true;
      if (_idleTimer) clearTimeout(_idleTimer);
      if (es) es.close();
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
  return ctrl;
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
  TokenStats.reset();
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

        const result = _dispatchSseEvent(data, handlers, outputEl);
        if (result === "done" || result === "error") {
          return cancel;
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
 * marked.jsで変換した後、DOMPurifyでXSS対策のサニタイズを行う。
 * どちらも利用できない場合はエスケープしてpreで囲む。
 * @param {string} text
 * @returns {string} サニタイズ済みHTML文字列
 */
function _renderMd(text) {
  // DOMPurify が読み込めていない場合は生HTMLを返さない（XSSフォールバック防止）。
  // LLM出力はインデックスされたファイル内容に影響され得るため信頼できない。
  if (typeof marked !== "undefined" && typeof DOMPurify !== "undefined") {
    return DOMPurify.sanitize(marked.parse(text || ""));
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
