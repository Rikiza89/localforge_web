/**
 * app.js — メインSPAロジック・タブ切替・プロジェクト管理
 * バニラJSのみ使用。外部ライブラリ不使用。
 */

"use strict";

// 現在のプロジェクトルートパス
let _currentProjectRoot = null;
// 現在のモード
let _currentMode = null;
// 現在の生成プランテキスト（生のJSON文字列）
let _currentPlanText = "";
// 現在実行中の生成ストリームのキャンセル関数
let _cancelGenStream = null;
// インデックス構築済みフラグ
let _indexBuilt = false;
// 最後のチェックポイントハッシュ
let _lastCheckpointHash = null;

/**
 * LLM がセクション先頭に出力したマークダウン見出し行を除去する。
 * h3 要素で既に表示されているタイトルと重複するため。
 * @param {string} buf - 累積トークンバッファ
 * @param {string} sectionName - 現在のセクション名
 * @returns {string} 見出し除去後のバッファ
 */
function _stripLeadingMdHeading(buf, sectionName) {
  const firstNewline = buf.indexOf("\n");
  const firstLine = firstNewline === -1 ? buf : buf.slice(0, firstNewline);
  if (!/^#{1,3}\s/.test(firstLine)) return buf;
  const title = firstLine.replace(/^#{1,3}\s+/, "").trim().toLowerCase();
  const sec = sectionName.toLowerCase();
  if (title.includes(sec) || sec.includes(title)) {
    return firstNewline === -1 ? "" : buf.slice(firstNewline + 1);
  }
  return buf;
}

/**
 * RAGインデックスの有無に応じてExplainタブのボタン状態を切り替える。
 * ragReady=true: ビルドボタンを「RAG再インデックス」に変更し、移行ボタンを隠す
 * ragReady=false: 通常ラベルに戻し、移行ボタンを表示する
 * @param {boolean} ragReady
 */
function _applyRagButtonState(ragReady) {
  const buildBtn = document.getElementById("build-index-btn");
  const migrateBtn = document.getElementById("migrate-vector-btn");
  if (!buildBtn) return;
  if (ragReady) {
    buildBtn.textContent = "⚙ RAG再インデックス";
    if (migrateBtn) migrateBtn.style.display = "none";
  } else {
    buildBtn.textContent = "⚙ インデックス構築";
    if (migrateBtn) { migrateBtn.style.display = ""; migrateBtn.disabled = false; }
  }
}

// =========================================================================
// UI ロック管理 — 生成中は操作ボタンを無効化し、停止ボタンのみ表示する
// =========================================================================

const _LOCKABLE_IDS = [
  "generate-plan-btn", "approve-plan-btn", "edit-plan-btn", "reprompt-btn",
  "apply-json-btn", "cancel-edit-btn",
  "continue-generation-btn", "modify-plan-btn", "view-full-report-btn",
  "continue-qa-btn", "generate-new-files-btn",
  "build-index-btn", "migrate-vector-btn", "generate-report-btn",
  "open-project-btn",
];

let _uiLocked = false;
let _savedDisabledState = {};
let _activeCancel = null;

function _lockUI(cancelFn) {
  if (_uiLocked) return;
  _uiLocked = true;
  _activeCancel = cancelFn || null;
  _savedDisabledState = {};

  _LOCKABLE_IDS.forEach(id => {
    const el = document.getElementById(id);
    if (el) { _savedDisabledState[id] = el.disabled; el.disabled = true; }
  });
  document.querySelectorAll(".tab-btn").forEach(btn => {
    _savedDisabledState["_tab_" + btn.id] = btn.disabled;
    btn.disabled = true;
  });

  const stopBtn = document.getElementById("global-stop-btn");
  if (stopBtn) stopBtn.style.display = "";
}

function _unlockUI() {
  if (!_uiLocked) return;
  _uiLocked = false;
  _activeCancel = null;

  _LOCKABLE_IDS.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.disabled = _savedDisabledState[id] ?? false;
  });
  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.disabled = _savedDisabledState["_tab_" + btn.id] ?? false;
  });
  _savedDisabledState = {};

  const stopBtn = document.getElementById("global-stop-btn");
  if (stopBtn) stopBtn.style.display = "none";
}

// =========================================================================
// チェックポイント管理
// =========================================================================

function _setCheckpoint(hash) {
  _lastCheckpointHash = hash;
  const badge = document.getElementById("checkpoint-badge");
  const rollbackBtn = document.getElementById("rollback-btn");
  if (badge) {
    badge.textContent = hash;
    badge.style.display = "";
    badge.title = `チェックポイント: ${hash} — ロールバック可能`;
  }
  if (rollbackBtn) rollbackBtn.style.display = "";
}

function _clearCheckpoint() {
  _lastCheckpointHash = null;
  const badge = document.getElementById("checkpoint-badge");
  const rollbackBtn = document.getElementById("rollback-btn");
  if (badge) badge.style.display = "none";
  if (rollbackBtn) rollbackBtn.style.display = "none";
}

async function _loadCheckpointFromServer() {
  if (!_currentProjectRoot) return;
  try {
    const data = await apiRequest("/api/git/checkpoint", "GET");
    if (data.checkpoint && data.checkpoint.hash) {
      _setCheckpoint(data.checkpoint.hash);
    } else {
      _clearCheckpoint();
    }
  } catch (e) {
    _clearCheckpoint();
  }
}

async function _doRollback() {
  if (!_lastCheckpointHash) return;
  if (!confirm(`チェックポイント ${_lastCheckpointHash} にロールバックしますか？\n⚠ この操作は元に戻せません。`)) return;
  try {
    const res = await apiRequest("/api/git/rollback", "POST", { hash: _lastCheckpointHash });
    if (res.ok) {
      showAlert(`ロールバック完了: ${_lastCheckpointHash}`, "success");
      _clearCheckpoint();
      refreshFileTree();
      refreshContextPanel();
      refreshGitLog();
    } else {
      showAlert(`ロールバック失敗: ${res.message}`, "error");
    }
  } catch (err) {
    showAlert(`ロールバックエラー: ${err.message}`, "error");
  }
}

// =========================================================================
// タブ切替
// =========================================================================

/**
 * 指定のタブに切り替える。
 * @param {"generate"|"resume"|"explain"} tabName - タブ名
 */
function switchTab(tabName) {
  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.tab === tabName);
  });
  document.querySelectorAll(".tab-content").forEach(content => {
    content.classList.toggle("active", content.id === `tab-content-${tabName}`);
  });
  _currentMode = tabName;

  // Explainタブに切り替わったとき、保存済みレポートを自動ロードする
  if (tabName === "explain" && _currentProjectRoot && !_uiLocked) {
    loadSavedReport().then(hasReport => {
      if (hasReport) enableChat();
    });
  }
}

// =========================================================================
// プロジェクトを開く
// =========================================================================

/**
 * フォルダ選択ダイアログを開いてプロジェクトをロードする。
 */
function _resetProjectUI() {
  // Clear stale content from the previous project before loading a new one
  _indexBuilt = false;
  window._reportResumFrom = 0;

  const reportOutput = document.getElementById("report-output");
  if (reportOutput) reportOutput.innerHTML = "";

  const chatHistory = document.getElementById("chat-history");
  if (chatHistory) chatHistory.innerHTML = "";

  const planSection = document.getElementById("plan-section");
  if (planSection) planSection.style.display = "none";

  const savedPlanBanner = document.getElementById("saved-plan-banner");
  if (savedPlanBanner) savedPlanBanner.style.display = "none";

  const partialBanner = document.getElementById("report-partial-banner");
  if (partialBanner) partialBanner.style.display = "none";

  const sectionPanel = document.getElementById("section-selector-panel");
  if (sectionPanel) sectionPanel.style.display = "none";

  const indexSummary = document.getElementById("index-summary");
  if (indexSummary) indexSummary.innerHTML = "";

  const reportBtn = document.getElementById("generate-report-btn");
  if (reportBtn) reportBtn.disabled = true;

  const indexProgress = document.getElementById("index-progress-container");
  if (indexProgress) indexProgress.style.display = "none";
}

async function openProject(pathOverride = null) {
  updateStatusBar("フォルダを選択中...");
  try {
    const body = pathOverride ? { path: pathOverride } : null;
    const data = await apiRequest("/api/project/open", "POST", body);
    _resetProjectUI();
    _currentProjectRoot = data.project_root;
    _currentMode = data.mode;

    // プロジェクトパス表示
    const pathEl = document.getElementById("project-path");
    if (pathEl) pathEl.textContent = data.project_root;

    // ファイルツリーのレンダリング
    const treeContainer = document.getElementById("file-tree");
    if (treeContainer && data.file_tree) {
      renderFileTree(data.file_tree, treeContainer);
    }

    // バナーアラートを表示
    if (data.banner) {
      showAlert(data.banner, "success", 6000);
    }

    // モードに応じてタブを切り替え
    switchTab(data.mode);

    // 保存済みプランがあればGenerateタブに表示
    await loadSavedPlan();

    // モデルセレクタをプロジェクトの保存済みモデルに合わせる
    const modelSelectEl = document.getElementById("model-selector");
    if (modelSelectEl && data.model) {
      const opt = modelSelectEl.querySelector(`option[value="${data.model}"]`);
      if (opt) modelSelectEl.value = data.model;
    }

    // コンテキストパネルを更新
    await refreshContextPanel();
    await refreshGitLog();

    // Explainモードなら自動でインデックス構築を開始
    if (data.mode === "explain") {
      setTimeout(() => buildIndex(), 800);
    }

    // Resumeモードなら再開状態を表示
    if (data.mode === "resume") {
      await loadResumeState();
    }

    // ワークスペースとピン留め状態を読み込む
    if (typeof loadWorkspace === "function") await loadWorkspace();
    if (typeof loadPinnedFromServer === "function") await loadPinnedFromServer();

    // チェックポイント状態を復元
    await _loadCheckpointFromServer();

    // ステータスバーを更新
    await refreshProjectStatus();
    updateStatusBar(`プロジェクトを開きました: ${data.project_root}`);

  } catch (err) {
    if (err.message && err.message.includes("NoFolderSelected")) {
      // ネイティブダイアログが使えない環境（Docker等）ではパス入力にフォールバック
      const typed = prompt(
        "フォルダのパスを入力してください:\n例: /projects/my-app"
      );
      if (typed && typed.trim()) {
        await openProject(typed.trim());
      } else {
        updateStatusBar("フォルダ選択がキャンセルされました");
      }
    } else {
      showAlert(`プロジェクトを開けませんでした: ${err.message}`, "error");
      updateStatusBar("エラーが発生しました");
    }
  }
}

// =========================================================================
// プロジェクト状態の更新
// =========================================================================

/**
 * プロジェクトのステータス（ブランチなど）を更新する。
 */
async function refreshProjectStatus() {
  try {
    const data = await apiRequest("/api/project/status");
    const gitEl = document.getElementById("status-git");
    if (gitEl && data.git_branch) {
      gitEl.textContent = `⎇ ${data.git_branch} ✓`;
    }
  } catch (e) {
    console.warn("ステータス更新エラー:", e.message);
  }
}

/**
 * コンテキストパネルを更新する。
 */
async function refreshContextPanel() {
  try {
    const data = await apiRequest("/api/project/context");
    const contextEl = document.getElementById("context-md-content");
    if (contextEl) {
      contextEl.textContent = data.content || "（空）";
    }
  } catch (e) {
    console.warn("コンテキスト更新エラー:", e.message);
  }

  // ProjectIndexサマリー
  try {
    const summary = await apiRequest("/api/explain/summary");
    const summaryEl = document.getElementById("index-summary");
    if (summaryEl) {
      summaryEl.innerHTML = `
        <div class="index-stat"><span>ファイル数</span><span>${summary.indexed_files} / ${summary.total_files}</span></div>
        <div class="index-stat" style="margin-top:6px; color:var(--text-muted); font-size:11px;">${(summary.summary || "").slice(0, 120)}...</div>
      `;
    }
    const reportBtn = document.getElementById("generate-report-btn");
    if (reportBtn) reportBtn.disabled = false;
    _indexBuilt = true;
    _applyRagButtonState(summary.rag_ready === true);
  } catch (e) {
    // インデックスが存在しない場合はスキップ
  }
}

/**
 * Gitログを更新する。
 */
async function refreshGitLog() {
  try {
    const data = await apiRequest("/api/git/log");
    const logEl = document.getElementById("git-log-list");
    if (!logEl) return;

    const commits = data.commits || [];
    if (commits.length === 0) {
      logEl.innerHTML = '<div class="empty-state">コミットなし</div>';
      return;
    }

    logEl.innerHTML = commits.slice(0, 8).map(c => `
      <div class="git-log-entry">
        <span class="git-log-hash">${c.hash}</span>
        <span>${escapeHtml(c.message.slice(0, 50))}</span>
      </div>
    `).join("");
  } catch (e) {
    // gitリポジトリが存在しない場合はスキップ
  }
}

// =========================================================================
// モデル管理
// =========================================================================

/**
 * Ollamaモデル一覧を取得してセレクタに反映する。
 */
async function loadModels() {
  const select = document.getElementById("model-selector");
  if (!select) return;

  try {
    const data = await apiRequest("/api/project/models");
    const models = data.models || [];

    select.innerHTML = models.length === 0
      ? '<option value="">モデルなし（Ollamaを起動してください）</option>'
      : models.map(m => `<option value="${escapeHtml(m)}">${escapeHtml(m)}</option>`).join("");

    // 現在の選択を反映
    if (_currentProjectRoot) {
      const status = await apiRequest("/api/project/status");
      if (status.model) select.value = status.model;
    }
  } catch (err) {
    select.innerHTML = '<option value="">Ollama接続エラー</option>';
  }
}

/**
 * 起動時にOllamaの接続状態を確認し、問題があれば永続バナーを表示する。
 */
async function checkOllamaHealth() {
  try {
    const data = await apiRequest("/api/project/ollama-status");
    if (!data.available) {
      _showOllamaWarning(data.error || "Ollamaサーバーに接続できません。");
      return;
    }
    if (data.models.length === 0) {
      _showOllamaWarning(
        "Ollamaは起動していますがモデルがありません。" +
        " ターミナルで ollama pull <モデル名> を実行してください。"
      );
      return;
    }
    // 正常: ステータスバーにモデル数を表示
    updateStatusBar(`Ollama OK — ${data.models.length} モデル利用可能`);
  } catch (err) {
    _showOllamaWarning("Ollamaヘルスチェックに失敗しました: " + err.message);
  }
}

/**
 * Ollamaの問題を通知する永続バナーを表示する（手動で閉じるまで消えない）。
 * @param {string} message
 */
function _showOllamaWarning(message) {
  updateStatusBar("⚠ Ollama 未接続");
  showAlert("⚠ Ollama: " + message, "error", 0); // timeout=0 → 自動消去しない
}

// =========================================================================
// Generate タブのロジック
// =========================================================================

/**
 * プラン生成を開始する。
 */
async function generatePlan() {
  if (!_currentProjectRoot) {
    showAlert("先にフォルダを開いてください。", "warning");
    return;
  }

  const promptEl = document.getElementById("generate-prompt");
  const prompt = promptEl ? promptEl.value.trim() : "";
  if (!prompt) {
    showAlert("プロンプトを入力してください。", "warning");
    return;
  }

  const planSection = document.getElementById("plan-section");
  const planStream = document.getElementById("plan-stream-output");
  const planTree = document.getElementById("plan-tree");

  if (planSection) planSection.style.display = "flex";
  if (planTree) planTree.innerHTML = "";
  if (planStream) {
    planStream.style.display = "block";
    planStream.textContent = "";
  }

  _currentPlanText = "";
  updateStatusBar("プランを生成中...");

  // UIで選択中のモデルをプロジェクト設定に同期する
  const modelSelectEl = document.getElementById("model-selector");
  const selectedModel = modelSelectEl ? modelSelectEl.value : null;
  if (selectedModel) {
    try { await apiRequest("/api/project/model", "POST", { model: selectedModel }); }
    catch (e) { console.warn("モデル同期エラー:", e.message); }
  }

  // Read optional file-count hints
  const _maxFilesEl = document.getElementById("plan-max-files");
  const _minFilesEl = document.getElementById("plan-min-files");
  const _maxFiles = _maxFilesEl && _maxFilesEl.value.trim() ? parseInt(_maxFilesEl.value, 10) : null;
  const _minFiles = _minFilesEl && _minFilesEl.value.trim() ? parseInt(_minFilesEl.value, 10) : null;
  const _planBody = { prompt };
  if (_maxFiles && _maxFiles > 0) _planBody.max_files = _maxFiles;
  if (_minFiles && _minFiles > 0) _planBody.min_files = _minFiles;

  _lockUI(null);
  await startPostStream(
    "/api/generate/plan",
    _planBody,
    planStream,
    {
      onToken: (token) => { _currentPlanText += token; },
      onDone: () => {
        _unlockUI();
        if (planStream) planStream.style.display = "none";
        renderPlanTree(_currentPlanText);
        updateStatusBar("プラン生成完了");

        // JSONエディタに内容をセット
        const textarea = document.getElementById("plan-json-textarea");
        if (textarea) textarea.value = _currentPlanText;
      },
      onError: (err) => {
        _unlockUI();
        showAlert(`プラン生成エラー: ${err}`, "error");
        updateStatusBar("エラーが発生しました");
      },
    }
  );
  _unlockUI();
}

/**
 * 生成プランをビジュアルツリーとしてレンダリングする。
 * LLMが出力するMarkdownサマリー（F）を上部に表示し、
 * ファイルリストにはNEW/EDITバッジ（D）を付与する。
 * @param {string} planText - プランテキスト（Markdownサマリー + ```json...``` ブロック）
 */
function renderPlanTree(planText) {
  const treeEl = document.getElementById("plan-tree");
  const summaryEl = document.getElementById("plan-summary");
  if (!treeEl) return;

  // F: ```json ブロック前のMarkdownサマリーを抽出して表示
  const jsonBlockIdx = planText.indexOf("```json");
  const plainBlockIdx = planText.indexOf("```");
  const firstBlockIdx = jsonBlockIdx >= 0 ? jsonBlockIdx : plainBlockIdx;
  if (summaryEl) {
    const md = firstBlockIdx > 0 ? planText.slice(0, firstBlockIdx).trim() : "";
    if (md) {
      summaryEl.innerHTML = _renderMd(md);
      summaryEl.style.display = "block";
    } else {
      summaryEl.style.display = "none";
    }
  }

  // JSONを抽出してパース
  let data = null;
  try {
    let text = planText.trim();
    if (text.includes("```json")) {
      const start = text.indexOf("```json") + 7;
      const end = text.indexOf("```", start);
      text = text.slice(start, end).trim();
    } else if (text.includes("```")) {
      const start = text.indexOf("```") + 3;
      const end = text.indexOf("```", start);
      text = text.slice(start, end).trim();
    }
    const jsonStart = text.indexOf("{");
    const jsonEnd = text.lastIndexOf("}") + 1;
    if (jsonStart >= 0) text = text.slice(jsonStart, jsonEnd);
    data = JSON.parse(text);
  } catch (e) {
    treeEl.innerHTML = `<div class="text-muted" style="font-size:12px; font-family:var(--font-mono); white-space:pre-wrap;">${escapeHtml(planText.slice(0, 2000))}</div>`;
    return;
  }

  const files = data.files || [];
  treeEl.innerHTML = "";

  if (data.project_name || data.description) {
    const header = document.createElement("div");
    header.style.marginBottom = "10px";
    header.innerHTML = `
      <strong style="color:var(--accent)">${escapeHtml(data.project_name || "")}</strong>
      <div style="color:var(--text-muted); font-size:12px; margin-top:4px;">${escapeHtml(data.description || "")}</div>
    `;
    treeEl.appendChild(header);
  }

  // D: NEW / EDIT バッジ付きでファイルリストをレンダリング
  files.forEach(f => {
    const isModify = f.action === "modify";
    const item = document.createElement("div");
    item.className = "plan-file-item";
    item.innerHTML = `
      <span class="plan-badge ${isModify ? "plan-badge-edit" : "plan-badge-new"}">${isModify ? "EDIT" : "NEW"}</span>
      <span class="plan-file-path">${escapeHtml(f.path || "")}</span>
      <span class="plan-file-desc">${escapeHtml(f.description || "")}</span>
      ${f.modification_notes ? `<span class="plan-mod-notes">${escapeHtml(f.modification_notes)}</span>` : ""}
    `;
    treeEl.appendChild(item);
  });
}

/**
 * プランを承認してファイル生成を開始する。
 */
async function approvePlanAndGenerate() {
  const textarea = document.getElementById("plan-json-textarea");
  const planJson = textarea ? textarea.value.trim() : _currentPlanText;

  if (!planJson) {
    showAlert("プランが空です。", "warning");
    return;
  }

  try {
    const data = await apiRequest("/api/generate/approve", "POST", { plan_json: planJson });
    showAlert(`プランを承認しました: ${data.plan.file_count}ファイル`, "success");
  } catch (err) {
    showAlert(`プラン承認エラー: ${err.message}`, "error");
    return;
  }

  // 生成セクションを表示
  const genSection = document.getElementById("generation-section");
  const planSection = document.getElementById("plan-section");
  const genStream = document.getElementById("generation-stream-output");
  const genProgress = document.getElementById("generation-progress");
  const progressLabel = document.getElementById("progress-label");

  if (planSection) planSection.style.display = "none";
  if (genSection) genSection.style.display = "flex";
  if (genStream) genStream.textContent = "";

  updateStatusBar("ファイルを生成中...");

  // UIで選択中のモデルをプロジェクト設定に同期する
  const approveModelEl = document.getElementById("model-selector");
  const approveModel = approveModelEl ? approveModelEl.value : null;
  if (approveModel) {
    try { await apiRequest("/api/project/model", "POST", { model: approveModel }); }
    catch (e) { console.warn("モデル同期エラー:", e.message); }
  }

  // 新ブランチで生成するか確認
  const branchToggle = document.getElementById("gen-branch-toggle");
  if (branchToggle && branchToggle.checked) {
    try {
      const branchRes = await apiRequest("/api/git/branch", "POST", {});
      if (branchRes.branch) {
        showAlert(`新ブランチ「${branchRes.branch}」で生成します`, "success", 3000);
      }
    } catch (e) {
      console.warn("ブランチ作成エラー:", e.message);
    }
  }

  const genFileHeader = document.getElementById("gen-current-file-header");

  const _es = startStream("/api/generate/start", genStream, {
    onProgress: (done, total, currentFile) => {
      if (genProgress) {
        genProgress.max = total;
        genProgress.value = done;
      }
      if (progressLabel) {
        progressLabel.textContent = `${done} / ${total}: ${currentFile}`;
      }
      updateStatusBar(`生成中: ${currentFile} (${done}/${total})`);
      // Clear the output area for each new file so tokens don't accumulate across files
      if (genStream) genStream.textContent = "";
      if (genFileHeader && currentFile) {
        genFileHeader.style.display = "flex";
        genFileHeader.innerHTML =
          `<span class="gen-file-icon">▶</span>` +
          `<span class="gen-file-name">${escapeHtml(currentFile)}</span>` +
          `<span class="gen-file-count">${done + 1} / ${total}</span>`;
      }
    },
    onFileWritten: (path) => {
      refreshFileTree();
      // Briefly show a completion tick before the next file clears the header
      if (genFileHeader) {
        genFileHeader.innerHTML =
          `<span class="gen-file-icon gen-file-done">✓</span>` +
          `<span class="gen-file-name">${escapeHtml(path)}</span>`;
      }
    },
    onCheckpoint: (hash) => {
      _setCheckpoint(hash);
    },
    onWarning: (msg) => {
      if (genStream) {
        const div = document.createElement("div");
        div.className = "stream-warning";
        div.textContent = msg;
        genStream.appendChild(div);
        div.scrollIntoView({ block: "end" });
      }
      showAlert(msg, "warning", 8000);
    },
    onDone: () => {
      _unlockUI();
      updateStatusBar("ファイル生成完了");
      showAlert("すべてのファイルを生成しました！", "success");
      if (genSection) genSection.style.display = "none";
      refreshFileTree();
      refreshContextPanel();
      refreshGitLog();
    },
    onError: (err) => {
      _unlockUI();
      showAlert(`生成エラー: ${err}`, "error");
      updateStatusBar("エラーが発生しました");
    },
  });
  _lockUI(() => _es.close());
  _cancelGenStream = _es;
}

// =========================================================================
// Explain タブのロジック
// =========================================================================

/**
 * インデックスを構築する。
 */
async function buildIndex() {
  if (!_currentProjectRoot) {
    showAlert("先にフォルダを開いてください。", "warning");
    return;
  }

  // UIで選択中のモデルをプロジェクト設定に同期する
  const modelSelectEl = document.getElementById("model-selector");
  const selectedModel = modelSelectEl ? modelSelectEl.value : null;
  if (selectedModel) {
    try {
      await apiRequest("/api/project/model", "POST", { model: selectedModel });
    } catch (e) {
      console.warn("モデル同期エラー:", e.message);
    }
  }

  const progressContainer = document.getElementById("index-progress-container");
  const indexProgress = document.getElementById("index-progress");
  const progressLabel = document.getElementById("index-progress-label");
  const reportBtn = document.getElementById("generate-report-btn");

  if (progressContainer) progressContainer.style.display = "block";
  updateStatusBar("インデックスを構築中...");

  const _es = startStream("/api/explain/index", null, {
    onProgress: (done, total, currentFile) => {
      if (indexProgress) {
        indexProgress.max = Math.max(total, 1);
        indexProgress.value = done;
      }
      if (progressLabel) {
        progressLabel.textContent = total > 0
          ? `${done} / ${total}: ${currentFile}`
          : "インデックス構築中...";
      }
      if (total > 0 && currentFile) {
        updateStatusBar(`インデックス構築中: ${currentFile} (${done}/${total})`);
      }
    },
    onDone: async () => {
      _unlockUI();
      if (progressContainer) progressContainer.style.display = "none";
      if (reportBtn) reportBtn.disabled = false;
      const histBtn = document.getElementById("report-history-btn");
      if (histBtn) histBtn.disabled = false;
      _indexBuilt = true;
      updateStatusBar("インデックス構築完了");
      _applyRagButtonState(true);
      refreshContextPanel();
      _initSectionSelector();

      // セクション選択パネルを表示
      const sectionPanel = document.getElementById("section-selector-panel");
      if (sectionPanel) sectionPanel.style.display = "";

      // 保存済みレポートがあればロードしてQ&Aをそのまま有効化する
      const hasReport = await loadSavedReport();
      if (hasReport) {
        enableChat();
        showAlert("インデックス完了 — 保存済みレポートを読み込みました。Q&Aで質問できます。", "success");
      } else {
        showAlert("インデックスが完成しました。「レポート生成」を押してください。", "success");
      }
    },
    onError: (err) => {
      _unlockUI();
      if (progressContainer) progressContainer.style.display = "none";
      showAlert(`インデックス構築エラー: ${err}`, "error");
      updateStatusBar("エラーが発生しました");
    },
  });
  _lockUI(() => _es.close());
}

/**
 * 既存のJSONLインデックスをChromaDBベクトルインデックスへ移行する。
 */
async function migrateVectorIndex() {
  if (!_currentProjectRoot) {
    showAlert("先にフォルダを開いてください。", "warning");
    return;
  }

  const progressContainer = document.getElementById("index-progress-container");
  const indexProgress = document.getElementById("index-progress");
  const progressLabel = document.getElementById("index-progress-label");
  const migrateBtn = document.getElementById("migrate-vector-btn");

  if (progressContainer) progressContainer.style.display = "block";
  if (migrateBtn) migrateBtn.disabled = true;
  updateStatusBar("RAGベクトルインデックス移行中...");

  const _es = startStream("/api/explain/migrate-vector", null, {
    onProgress: (done, total, currentFile) => {
      if (indexProgress) {
        indexProgress.max = Math.max(total, 1);
        indexProgress.value = done;
      }
      if (progressLabel) {
        progressLabel.textContent = total > 0
          ? `RAG移行: ${done} / ${total}: ${currentFile}`
          : "RAGベクトルインデックス移行中...";
      }
      if (total > 0 && currentFile) {
        updateStatusBar(`RAG移行中: ${currentFile} (${done}/${total})`);
      }
    },
    onDone: () => {
      _unlockUI();
      if (progressContainer) progressContainer.style.display = "none";
      updateStatusBar("RAGベクトルインデックス移行完了");
      showAlert("RAG移行完了 — セマンティック検索が有効になりました。", "success");
    },
    onError: (err) => {
      _unlockUI();
      if (progressContainer) progressContainer.style.display = "none";
      showAlert(`RAG移行エラー: ${err}`, "error");
      updateStatusBar("エラーが発生しました");
    },
  });
  _lockUI(() => _es.close());
}

/**
 * 保存済みレポート（.localforge/report.md）を読み込んでUIに表示する。
 * レポートが存在すればtrue、存在しなければfalseを返す。
 * 部分的なレポートの場合は専用バナーを表示する。
 * @returns {Promise<boolean>}
 */
async function loadSavedReport() {
  try {
    const data = await apiRequest("/api/explain/saved-report");
    if (!data.content) {
      _hideSavedReportBanner();
      return false;
    }

    const reportOutput = document.getElementById("report-output");
    if (!reportOutput) return false;

    reportOutput.innerHTML = _renderMd(data.content);

    if (data.partial) {
      _showPartialBanner(data.sections_done, data.sections_total);
    } else {
      _hideSavedReportBanner();
    }

    await _loadQAHistory();

    return true;
  } catch (e) {
    return false;
  }
}

async function _loadQAHistory() {
  try {
    const data = await apiRequest("/api/explain/qa-history");
    if (data.entries && data.entries.length > 0) {
      const historyEl = document.getElementById("chat-history");
      if (historyEl) historyEl.innerHTML = "";
      if (typeof loadChatHistory === "function") loadChatHistory(data.entries);
    }
  } catch (e) {
    console.warn("Q&A履歴ロードエラー:", e.message);
  }
}

/**
 * 保存済みプラン（.localforge/plan.json）を読み込んでGenerateタブに表示する。
 * プランが存在しない場合は何もしない。
 */
async function loadSavedPlan() {
  if (!_currentProjectRoot) return;
  try {
    const data = await apiRequest("/api/generate/plan/saved");
    if (!data.plan) return;
    const plan = data.plan;

    const planSection = document.getElementById("plan-section");
    const planTree = document.getElementById("plan-tree");
    const planSummary = document.getElementById("plan-summary");

    if (planSection) planSection.style.display = "flex";

    if (planSummary && plan.description) {
      planSummary.innerHTML = _renderMd(plan.description);
      planSummary.style.display = "block";
    }

    if (planTree) {
      planTree.innerHTML = "";

      if (plan.project_name || plan.description) {
        const header = document.createElement("div");
        header.style.marginBottom = "10px";
        header.innerHTML = `
          <strong style="color:var(--accent)">${escapeHtml(plan.project_name || "")}</strong>
          <div style="color:var(--text-muted); font-size:12px; margin-top:4px;">${escapeHtml((plan.description || "").slice(0, 120))}</div>
        `;
        planTree.appendChild(header);
      }

      plan.files.forEach(f => {
        const isModify = f.action === "modify";
        const existsBadge = f.exists
          ? `<span class="plan-badge-exists" title="ファイルが存在します">✓</span>`
          : "";
        const item = document.createElement("div");
        item.className = "plan-file-item";
        item.innerHTML = `
          <span class="plan-badge ${isModify ? "plan-badge-edit" : "plan-badge-new"}">${isModify ? "EDIT" : "NEW"}</span>
          ${existsBadge}
          <span class="plan-file-path">${escapeHtml(f.path || "")}</span>
          <span class="plan-file-desc">${escapeHtml(f.description || "")}</span>
          ${f.modification_notes ? `<span class="plan-mod-notes">${escapeHtml(f.modification_notes)}</span>` : ""}
        `;
        planTree.appendChild(item);
      });
    }

    // Check generation progress to decide banner content
    let progress = null;
    try {
      progress = await apiRequest("/api/project/generation-progress");
    } catch (e) {
      console.warn("生成進捗取得エラー:", e.message);
    }

    _updateSavedPlanBanner(plan, progress);

  } catch (e) {
    if (e.message && (e.message.includes("404") || e.message.includes("NoPlan"))) return;
    console.warn("保存済みプランロードエラー:", e.message);
  }
}

/**
 * 保存済みプランバナーの内容を進捗に応じて更新する。
 * @param {Object} plan - プランオブジェクト（approved フラグを含む）
 * @param {Object|null} progress - 生成進捗オブジェクト
 */
function _updateSavedPlanBanner(plan, progress) {
  const banner = document.getElementById("saved-plan-banner");
  const bannerText = document.getElementById("saved-plan-banner-text");
  const bannerActions = document.getElementById("saved-plan-banner-actions");
  const resumeBtn = document.getElementById("resume-generation-btn");
  const restartBtn = document.getElementById("restart-generation-btn");

  if (!banner) return;

  const isApproved = plan && plan.approved;
  const total = progress ? progress.total : 0;
  const completed = progress ? progress.completed : 0;
  const isComplete = total > 0 && completed >= total;
  const hasPartialProgress = isApproved && completed > 0 && !isComplete;

  if (isComplete) {
    if (bannerText) bannerText.textContent = `プランは完了しています（全${total}ファイル生成済み）。新しいプロンプトを入力してください。`;
    if (bannerActions) bannerActions.style.display = "none";
  } else if (hasPartialProgress) {
    if (bannerText) bannerText.textContent = `生成が中断されています（${completed}/${total} 完了）。続きから再開するか、最初から生成できます。`;
    if (resumeBtn) resumeBtn.style.display = "";
    if (restartBtn) { restartBtn.style.display = ""; restartBtn.textContent = "▶ 最初から生成"; }
    if (bannerActions) bannerActions.style.display = "flex";
  } else if (isApproved) {
    if (bannerText) bannerText.textContent = "プランが承認済みです。生成を開始するか、新しいプロンプトで再生成できます。";
    if (resumeBtn) resumeBtn.style.display = "none";
    if (restartBtn) { restartBtn.style.display = ""; restartBtn.textContent = "▶ 生成開始"; }
    if (bannerActions) bannerActions.style.display = "flex";
  } else {
    if (bannerText) bannerText.textContent = "保存済みプランを読み込みました。承認して生成を続行するか、新しいプロンプトを入力してください。";
    if (bannerActions) bannerActions.style.display = "none";
  }

  banner.style.display = "flex";
}

/**
 * 保存済みプランの生成を続きから再開する（前回中断した箇所から）。
 */
async function _resumeFromSavedPlan() {
  const genSection = document.getElementById("generation-section");
  const planSection = document.getElementById("plan-section");
  const genStream = document.getElementById("generation-stream-output");
  const genProgress = document.getElementById("generation-progress");
  const progressLabel = document.getElementById("progress-label");
  const genFileHeader = document.getElementById("gen-current-file-header");
  const banner = document.getElementById("saved-plan-banner");

  if (banner) banner.style.display = "none";
  if (planSection) planSection.style.display = "none";
  if (genSection) genSection.style.display = "flex";
  if (genStream) genStream.textContent = "";

  updateStatusBar("生成を再開中...");

  const modelEl = document.getElementById("model-selector");
  const model = modelEl ? modelEl.value : null;
  if (model) {
    try { await apiRequest("/api/project/model", "POST", { model }); }
    catch (e) { console.warn("モデル同期エラー:", e.message); }
  }

  const _es = startStream("/api/generate/resume", genStream, {
    onProgress: (done, total, currentFile) => {
      if (genProgress) { genProgress.max = total; genProgress.value = done; }
      if (progressLabel) progressLabel.textContent = `${done} / ${total}: ${currentFile}`;
      updateStatusBar(`再開中: ${currentFile} (${done}/${total})`);
      if (genStream) genStream.textContent = "";
      if (genFileHeader && currentFile) {
        genFileHeader.style.display = "flex";
        genFileHeader.innerHTML =
          `<span class="gen-file-icon">▶</span>` +
          `<span class="gen-file-name">${escapeHtml(currentFile)}</span>` +
          `<span class="gen-file-count">${done + 1} / ${total}</span>`;
      }
    },
    onFileWritten: (path) => {
      refreshFileTree();
      if (genFileHeader) {
        genFileHeader.innerHTML =
          `<span class="gen-file-icon gen-file-done">✓</span>` +
          `<span class="gen-file-name">${escapeHtml(path)}</span>`;
      }
    },
    onDone: () => {
      _unlockUI();
      updateStatusBar("生成再開完了");
      showAlert("すべてのファイルを生成しました！", "success");
      if (genSection) genSection.style.display = "none";
      refreshFileTree();
      refreshContextPanel();
      refreshGitLog();
    },
    onError: (err) => {
      _unlockUI();
      showAlert(`再開エラー: ${err}`, "error");
      updateStatusBar("エラーが発生しました");
    },
  });
  _lockUI(() => _es.close());
  _cancelGenStream = _es;
}

/**
 * 保存済みプランの生成を最初から（全ファイル）やり直す。
 */
async function _restartFromSavedPlan() {
  const genSection = document.getElementById("generation-section");
  const planSection = document.getElementById("plan-section");
  const genStream = document.getElementById("generation-stream-output");
  const genProgress = document.getElementById("generation-progress");
  const progressLabel = document.getElementById("progress-label");
  const genFileHeader = document.getElementById("gen-current-file-header");
  const banner = document.getElementById("saved-plan-banner");

  if (banner) banner.style.display = "none";
  if (planSection) planSection.style.display = "none";
  if (genSection) genSection.style.display = "flex";
  if (genStream) genStream.textContent = "";

  updateStatusBar("ファイルを生成中...");

  const modelEl = document.getElementById("model-selector");
  const model = modelEl ? modelEl.value : null;
  if (model) {
    try { await apiRequest("/api/project/model", "POST", { model }); }
    catch (e) { console.warn("モデル同期エラー:", e.message); }
  }

  const _es = startStream("/api/generate/start", genStream, {
    onProgress: (done, total, currentFile) => {
      if (genProgress) { genProgress.max = total; genProgress.value = done; }
      if (progressLabel) progressLabel.textContent = `${done} / ${total}: ${currentFile}`;
      updateStatusBar(`生成中: ${currentFile} (${done}/${total})`);
      if (genStream) genStream.textContent = "";
      if (genFileHeader && currentFile) {
        genFileHeader.style.display = "flex";
        genFileHeader.innerHTML =
          `<span class="gen-file-icon">▶</span>` +
          `<span class="gen-file-name">${escapeHtml(currentFile)}</span>` +
          `<span class="gen-file-count">${done + 1} / ${total}</span>`;
      }
    },
    onFileWritten: (path) => {
      refreshFileTree();
      if (genFileHeader) {
        genFileHeader.innerHTML =
          `<span class="gen-file-icon gen-file-done">✓</span>` +
          `<span class="gen-file-name">${escapeHtml(path)}</span>`;
      }
    },
    onDone: () => {
      _unlockUI();
      updateStatusBar("ファイル生成完了");
      showAlert("すべてのファイルを生成しました！", "success");
      if (genSection) genSection.style.display = "none";
      refreshFileTree();
      refreshContextPanel();
      refreshGitLog();
    },
    onError: (err) => {
      _unlockUI();
      showAlert(`生成エラー: ${err}`, "error");
      updateStatusBar("エラーが発生しました");
    },
  });
  _lockUI(() => _es.close());
  _cancelGenStream = _es;
}

function _showPartialBanner(done, total) {
  const banner = document.getElementById("report-partial-banner");
  const text = document.getElementById("report-partial-text");
  if (banner) banner.style.display = "flex";
  if (text) text.textContent = `部分レポート: ${done}/${total} セクション完了 — 残りを生成するか最初から再生成できます。`;
  // resume_from をグローバルに記憶
  window._reportResumFrom = done;
}

function _hideSavedReportBanner() {
  const banner = document.getElementById("report-partial-banner");
  if (banner) banner.style.display = "none";
  window._reportResumFrom = 0;
}

/**
 * 説明レポートを生成する。
 * @param {Object} opts - オプション
 * @param {number[]} [opts.sectionIndices] - 生成するセクションのインデックスリスト（省略で全セクション）
 * @param {number} [opts.resumeFrom] - このインデックス以降を生成（0で最初から）
 * @param {string} [opts.model] - 使用モデルの上書き（省略でプロジェクト設定）
 * @param {string} [opts.lang] - 出力言語 ("ja" = 日本語デフォルト, "en" = 英語)
 */
function generateReport(opts = {}) {
  if (!_indexBuilt && !_currentProjectRoot) {
    showAlert("先にインデックスを構築してください。", "warning");
    return;
  }

  const reportOutput = document.getElementById("report-output");
  if (reportOutput && !opts.resumeFrom) reportOutput.innerHTML = "";

  _hideSavedReportBanner();

  let currentSectionEl = null;
  const _renderedSections = new Set();
  updateStatusBar("レポートを生成中...");

  // SSEエンドポイントにパラメータを付加
  const params = new URLSearchParams();
  if (opts.sectionIndices && opts.sectionIndices.length > 0) {
    params.set("sections", opts.sectionIndices.join(","));
  }
  if (opts.resumeFrom && opts.resumeFrom > 0) {
    params.set("resume_from", String(opts.resumeFrom));
  }
  if (opts.model) {
    params.set("model", opts.model);
  }
  if (opts.lang && opts.lang !== "ja") {
    params.set("lang", opts.lang);
  }
  const streamUrl = "/api/explain/report" + (params.toString() ? "?" + params.toString() : "");

  // noReconnect: true — レポート生成中のサイレント再接続を防ぐ
  const _ctrl = startStream(streamUrl, null, {
    noReconnect: true,
    onSection: (name, idx, total) => {
      if (!reportOutput) return;

      const countStr = (idx && total) ? ` (${idx}/${total})` : "";
      updateStatusBar(`レポート生成中: ${name}${countStr}`);

      if (_renderedSections.has(name)) {
        // 重複セクション = 予期しない再接続。安全に終了。
        _ctrl.close();
        showAlert("予期しないストリーム再起動を検出しました。「続きから生成」で再試行できます。", "warning");
        _unlockUI();
        return;
      }
      _renderedSections.add(name);

      const h3 = document.createElement("h3");
      h3.textContent = name;
      h3.dataset.section = name;
      const hr = document.createElement("hr");
      reportOutput.appendChild(h3);
      reportOutput.appendChild(hr);

      currentSectionEl = document.createElement("div");
      currentSectionEl.className = "md-body";
      currentSectionEl._sectionName = name;
      reportOutput.appendChild(currentSectionEl);
    },
    onToken: (token) => {
      if (currentSectionEl) {
        currentSectionEl._mdBuf = (currentSectionEl._mdBuf || "") + token;
        const buf = _stripLeadingMdHeading(currentSectionEl._mdBuf, currentSectionEl._sectionName || "");
        currentSectionEl.innerHTML = _renderMd(buf);
        if (reportOutput) _autoScroll(reportOutput);
      }
    },
    onProgress: (done, total, currentFile) => {
      updateStatusBar(`レポート生成中: ${currentFile} ✓ (${done}/${total})`);
    },
    onDone: () => {
      _unlockUI();
      updateStatusBar("レポート生成完了");
      showAlert("レポートが完成しました！Q&Aで質問できます。", "success");
      _hideSavedReportBanner();
      enableChat();
    },
    onError: (err) => {
      _unlockUI();
      showAlert(`レポート生成エラー: ${err}`, "error");
      updateStatusBar("エラーが発生しました");
      // 部分的に保存されたレポートを再ロードしてバナーを更新
      loadSavedReport();
    },
  });
  _lockUI(() => _ctrl.close());
}

// =========================================================================
// Resume タブのロジック
// =========================================================================

/**
 * 再開状態を読み込んでUIに反映する。
 */
async function loadResumeState() {
  const emptyEl = document.getElementById("resume-empty");
  const contentEl = document.getElementById("resume-content");

  try {
    const status = await apiRequest("/api/project/status");
    if (!status.root) {
      if (emptyEl) emptyEl.style.display = "block";
      if (contentEl) contentEl.style.display = "none";
      return;
    }

    if (emptyEl) emptyEl.style.display = "none";
    if (contentEl) contentEl.style.display = "flex";

    // ファイルツリーから再開情報を構築
    const treeData = await apiRequest("/api/project/tree");

    // 簡易的な再開状態を表示（実際はresumeサービスからデータを取得）
    const lastCommitEl = document.getElementById("last-commit-msg");
    const progressTextEl = document.getElementById("resume-progress-text");
    const progressBarEl = document.getElementById("resume-progress-bar");

    // Gitログから最終コミットを取得
    try {
      const gitData = await apiRequest("/api/git/log");
      const commits = gitData.commits || [];
      if (lastCommitEl && commits.length > 0) {
        lastCommitEl.textContent = commits[0].message;
      }
    } catch (e) { /* git未初期化の場合はスキップ */ }

    // プロジェクトタイプに応じてボタンを表示
    const lfActions = document.getElementById("resume-lf-actions");
    const foreignActions = document.getElementById("resume-foreign-actions");

    // .localforge/plan.jsonの存在をファイルツリーで判断（簡易）
    const hasLocalforge = (treeData.file_tree || []).some(n => n.name === ".localforge");

    if (hasLocalforge) {
      if (lfActions) lfActions.style.display = "flex";
      if (foreignActions) foreignActions.style.display = "none";
    } else {
      if (lfActions) lfActions.style.display = "none";
      if (foreignActions) foreignActions.style.display = "flex";
    }

  } catch (err) {
    showAlert(`再開状態の読み込みエラー: ${err.message}`, "error");
  }
}

/**
 * 生成を再開する。
 */
async function continueGeneration() {
  const genSection = document.getElementById("resume-generation-section");
  const genStream = document.getElementById("resume-stream-output");
  const genProgress = document.getElementById("resume-gen-progress");
  const progressLabel = document.getElementById("resume-gen-progress-label");

  if (genSection) genSection.style.display = "flex";
  if (genStream) genStream.textContent = "";

  updateStatusBar("生成を再開中...");

  // UIで選択中のモデルをプロジェクト設定に同期する
  const resumeModelEl = document.getElementById("model-selector");
  const resumeModel = resumeModelEl ? resumeModelEl.value : null;
  if (resumeModel) {
    try { await apiRequest("/api/project/model", "POST", { model: resumeModel }); }
    catch (e) { console.warn("モデル同期エラー:", e.message); }
  }

  const _es = startStream("/api/generate/start", genStream, {
    onProgress: (done, total, currentFile) => {
      if (genProgress) {
        genProgress.max = total;
        genProgress.value = done;
      }
      if (progressLabel) {
        progressLabel.textContent = `${done} / ${total}: ${currentFile}`;
      }
      updateStatusBar(`再開中: ${currentFile} (${done}/${total})`);
      if (genStream) genStream.textContent = "";
    },
    onFileWritten: () => refreshFileTree(),
    onDone: () => {
      _unlockUI();
      updateStatusBar("生成再開完了");
      showAlert("すべてのファイルを生成しました！", "success");
      if (genSection) genSection.style.display = "none";
      refreshFileTree();
    },
    onError: (err) => {
      _unlockUI();
      showAlert(`再開エラー: ${err}`, "error");
      updateStatusBar("エラーが発生しました");
    },
  });
  _lockUI(() => _es.close());
}

// =========================================================================
// CPU スレッド管理
// =========================================================================

let _cpuCount = 1;

/**
 * CPU スレッド情報をAPIから取得してUIに反映する。
 */
async function loadCpuThreadInfo() {
  try {
    const data = await apiRequest("/api/project/num-thread");
    _cpuCount = data.cpu_count || 1;

    const slider  = document.getElementById("cpu-thread-slider");
    const maxLbl  = document.getElementById("cpu-max-label");
    const coreLbl = document.getElementById("cpu-core-label");

    if (maxLbl)  maxLbl.textContent  = _cpuCount;
    if (coreLbl) coreLbl.textContent = `${_cpuCount} コア`;
    if (slider) {
      slider.max   = _cpuCount;
      slider.value = data.num_thread ?? _cpuCount;
    }

    _updateCpuUI(data.num_thread);
  } catch (e) {
    console.warn("CPUスレッド情報取得エラー:", e.message);
  }
}

/**
 * スライダーバッジとラベルを現在の設定値に合わせて更新する。
 * @param {number|null} numThread
 */
function _updateCpuUI(numThread) {
  const badge   = document.getElementById("cpu-thread-badge");
  const valueEl = document.getElementById("cpu-slider-value");

  if (numThread === null || numThread === undefined) {
    if (badge)   { badge.textContent = "自動"; badge.className = "cpu-thread-badge"; }
    if (valueEl) valueEl.textContent = "自動設定";
  } else {
    const pct = Math.round((numThread / _cpuCount) * 100);
    if (badge)   { badge.textContent = `${numThread} スレッド`; badge.className = "cpu-thread-badge active"; }
    if (valueEl) valueEl.textContent = `${numThread} スレッド (${pct}%)`;
  }
}

/**
 * スレッド数をAPIに送信して即時適用する。
 * @param {number|null} numThread - nullで自動設定に戻す
 */
async function applyCpuThread(numThread) {
  try {
    const data = await apiRequest("/api/project/num-thread", "POST", { num_thread: numThread });
    _updateCpuUI(data.num_thread);
    const msg = data.num_thread !== null
      ? `CPUスレッドを ${data.num_thread} に設定しました`
      : "CPUスレッドを自動設定に戻しました";
    updateStatusBar(msg);
    showAlert(msg, "success", 3000);
  } catch (err) {
    showAlert(`CPUスレッド設定エラー: ${err.message}`, "error");
  }
}

// =========================================================================
// セクション選択・レポート履歴・比較ビュー
// =========================================================================

const REPORT_SECTIONS = [
  "Project Overview",
  "Module Map",
  "Entry Points & Startup Flow",
  "Data Flow",
  "Key Interfaces & Contracts",
  "External Dependencies",
  "Configuration",
  "Test Coverage",
  "Notable Patterns & Design Decisions",
  "Potential Issues & Technical Debt",
  "Project Health & Code Quality Analysis",
  "How to Extend This Project",
];

/** セクション選択パネルを初期化してチェックボックスを描画する。 */
function _initSectionSelector() {
  const panel = document.getElementById("section-selector-panel");
  const container = document.getElementById("section-checkboxes");
  if (!panel || !container) return;

  container.innerHTML = "";
  REPORT_SECTIONS.forEach((name, idx) => {
    const label = document.createElement("label");
    label.className = "section-checkbox-item";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.value = String(idx);
    cb.checked = true;
    cb.dataset.sectionIdx = idx;
    cb.addEventListener("change", _updateSectionCount);
    label.appendChild(cb);
    label.appendChild(document.createTextNode(` ${name}`));
    container.appendChild(label);
  });
  _updateSectionCount();

  // レポートモデルオーバーライドセレクタにメインのオプションをコピーする
  const mainSelect = document.getElementById("model-selector");
  const overrideSelect = document.getElementById("report-model-override");
  if (mainSelect && overrideSelect) {
    // 既存オプション（「プロジェクト設定を使用」以外）をクリア
    while (overrideSelect.options.length > 1) overrideSelect.remove(1);
    Array.from(mainSelect.options).forEach(opt => {
      if (opt.value) {
        const copy = new Option(opt.text, opt.value);
        overrideSelect.appendChild(copy);
      }
    });
  }
}

function _updateSectionCount() {
  const checkboxes = document.querySelectorAll("#section-checkboxes input[type=checkbox]");
  const checked = Array.from(checkboxes).filter(cb => cb.checked).length;
  const badge = document.getElementById("section-selector-count");
  if (badge) badge.textContent = `${checked}/${REPORT_SECTIONS.length} 選択中`;
}

/** チェック済みセクションのインデックスを返す。全選択の場合はnullを返す。 */
function _getSelectedSectionIndices() {
  const checkboxes = document.querySelectorAll("#section-checkboxes input[type=checkbox]");
  const all = Array.from(checkboxes);
  const checked = all.filter(cb => cb.checked).map(cb => parseInt(cb.value, 10));
  if (checked.length === all.length) return null; // 全セクション
  return checked;
}

/** 履歴パネルを開いてリストを読み込む。 */
async function openReportHistory() {
  const panel = document.getElementById("report-history-panel");
  if (!panel) return;
  panel.style.display = "flex";

  const list = document.getElementById("history-panel-list");
  if (!list) return;
  list.innerHTML = "<p>読み込み中...</p>";

  try {
    const data = await apiRequest("/api/explain/report-history");
    const history = data.history || [];
    if (history.length === 0) {
      list.innerHTML = "<p class='history-empty'>履歴がありません。</p>";
      return;
    }
    list.innerHTML = "";
    history.forEach(entry => {
      const item = document.createElement("div");
      item.className = "history-item";
      item.dataset.id = entry.id;

      const date = new Date(entry.created_at).toLocaleString();
      const badge = entry.partial ? " <span class='badge-partial'>部分</span>" : "";
      const sections = `${entry.sections_done}/${entry.sections_total}`;

      item.innerHTML = `
        <label class="history-select-label">
          <input type="checkbox" class="history-compare-cb" value="${entry.id}">
        </label>
        <div class="history-item-body">
          <div class="history-item-date">${date}${badge}</div>
          <div class="history-item-meta">モデル: ${escapeHtml(entry.model || "不明")} | ${sections}セクション</div>
        </div>
        <div class="history-item-actions">
          <button class="btn btn-sm btn-secondary history-load-btn" data-id="${entry.id}">読込</button>
          <button class="btn btn-sm btn-ghost history-delete-btn" data-id="${entry.id}" title="削除">✕</button>
        </div>
      `;
      list.appendChild(item);
    });

    // イベント委任
    list.querySelectorAll(".history-load-btn").forEach(btn => {
      btn.addEventListener("click", () => _loadHistoricalReport(btn.dataset.id));
    });
    list.querySelectorAll(".history-delete-btn").forEach(btn => {
      btn.addEventListener("click", () => _deleteHistoricalReport(btn.dataset.id));
    });
    list.querySelectorAll(".history-compare-cb").forEach(cb => {
      cb.addEventListener("change", _updateCompareButton);
    });

    _updateCompareButton();
  } catch (e) {
    list.innerHTML = `<p class='history-empty'>エラー: ${escapeHtml(e.message)}</p>`;
  }
}

function _updateCompareButton() {
  const checked = document.querySelectorAll(".history-compare-cb:checked");
  const btn = document.getElementById("history-compare-btn");
  if (btn) btn.disabled = checked.length !== 2;
}

async function _loadHistoricalReport(reportId) {
  try {
    const data = await apiRequest(`/api/explain/report-history/${reportId}`);
    const reportOutput = document.getElementById("report-output");
    if (reportOutput) reportOutput.innerHTML = _renderMd(data.content);
    const panel = document.getElementById("report-history-panel");
    if (panel) panel.style.display = "none";
    showAlert("履歴レポートを読み込みました。", "success", 3000);
  } catch (e) {
    showAlert(`レポート読込エラー: ${e.message}`, "error");
  }
}

async function _deleteHistoricalReport(reportId) {
  if (!confirm("このレポートを削除しますか？")) return;
  try {
    await apiRequest(`/api/explain/report-history/${reportId}`, "DELETE");
    showAlert("削除しました。", "success", 2000);
    openReportHistory(); // 一覧を再読み込み
  } catch (e) {
    showAlert(`削除エラー: ${e.message}`, "error");
  }
}

/** 2つのレポートを並べて比較するビューを開く。 */
async function openCompareView() {
  const checked = document.querySelectorAll(".history-compare-cb:checked");
  if (checked.length !== 2) return;

  const [idA, idB] = Array.from(checked).map(cb => cb.value);

  try {
    const [dataA, dataB] = await Promise.all([
      apiRequest(`/api/explain/report-history/${idA}`),
      apiRequest(`/api/explain/report-history/${idB}`),
    ]);

    const sectionsA = _parseReportSections(dataA.content);
    const sectionsB = _parseReportSections(dataB.content);

    // ヒストリーパネルを閉じて比較ビューを開く
    const histPanel = document.getElementById("report-history-panel");
    if (histPanel) histPanel.style.display = "none";

    const compareView = document.getElementById("report-compare-view");
    if (!compareView) return;
    compareView.style.display = "flex";

    // セクション選択セレクトを構築
    const sectionSelect = document.getElementById("compare-section-select");
    if (sectionSelect) {
      sectionSelect.innerHTML = "";
      const allSections = Array.from(new Set([...Object.keys(sectionsA), ...Object.keys(sectionsB)]));
      allSections.forEach(name => {
        const opt = new Option(name, name);
        sectionSelect.appendChild(opt);
      });
      sectionSelect.onchange = () => _renderCompareSection(sectionsA, sectionsB, sectionSelect.value, idA, idB);
      if (allSections.length > 0) _renderCompareSection(sectionsA, sectionsB, allSections[0], idA, idB);
    }
  } catch (e) {
    showAlert(`比較エラー: ${e.message}`, "error");
  }
}

function _renderCompareSection(sectionsA, sectionsB, sectionName, idA, idB) {
  const colA = document.getElementById("compare-col-a");
  const colB = document.getElementById("compare-col-b");
  const titleA = document.getElementById("compare-col-a-title");
  const titleB = document.getElementById("compare-col-b-title");

  if (titleA) titleA.textContent = idA;
  if (titleB) titleB.textContent = idB;

  if (colA) colA.innerHTML = _renderMd(sectionsA[sectionName] || "_このセクションは含まれていません_");
  if (colB) colB.innerHTML = _renderMd(sectionsB[sectionName] || "_このセクションは含まれていません_");
}

/** report.md 文字列をセクション名→内容の辞書にパースする。 */
function _parseReportSections(content) {
  const result = {};
  if (!content) return result;
  const lines = content.split("\n");
  let currentName = null;
  let buf = [];
  for (const line of lines) {
    if (line.startsWith("## ")) {
      if (currentName !== null) result[currentName] = buf.join("\n").trim();
      currentName = line.slice(3).trim();
      buf = [];
    } else {
      if (currentName !== null) buf.push(line);
    }
  }
  if (currentName !== null) result[currentName] = buf.join("\n").trim();
  return result;
}

// =========================================================================
// 初期化
// =========================================================================

document.addEventListener("DOMContentLoaded", async () => {
  // タブ切替ボタン
  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => switchTab(btn.dataset.tab));
  });

  // フォルダを開くボタン
  const openBtn = document.getElementById("open-project-btn");
  if (openBtn) openBtn.addEventListener("click", openProject);

  // モデル変更
  const modelSelect = document.getElementById("model-selector");
  if (modelSelect) {
    modelSelect.addEventListener("change", async () => {
      const model = modelSelect.value;
      if (!model || !_currentProjectRoot) return;
      try {
        await apiRequest("/api/project/model", "POST", { model });
        updateStatusBar(`モデルを変更しました: ${model}`);
      } catch (err) {
        showAlert(`モデル変更エラー: ${err.message}`, "error");
      }
    });
  }

  // モデルアンロードボタン
  const unloadBtn = document.getElementById("unload-model-btn");
  if (unloadBtn) {
    unloadBtn.addEventListener("click", async () => {
      const model = modelSelect ? modelSelect.value : null;
      if (!model) {
        showAlert("アンロードするモデルが選択されていません。", "warning");
        return;
      }
      try {
        updateStatusBar(`${model} をアンロード中...`);
        await apiRequest("/api/project/unload", "POST", { model });
        updateStatusBar(`${model} をアンロードしました`);
        showAlert(`${model} をVRAMから解放しました`, "success");
      } catch (err) {
        showAlert(`アンロード失敗: ${err.message}`, "error");
      }
    });
  }

  // Generateタブのボタン
  const generatePlanBtn = document.getElementById("generate-plan-btn");
  if (generatePlanBtn) generatePlanBtn.addEventListener("click", generatePlan);

  const approvePlanBtn = document.getElementById("approve-plan-btn");
  if (approvePlanBtn) approvePlanBtn.addEventListener("click", approvePlanAndGenerate);

  const editPlanBtn = document.getElementById("edit-plan-btn");
  if (editPlanBtn) {
    editPlanBtn.addEventListener("click", () => {
      const editor = document.getElementById("plan-json-editor");
      if (editor) editor.style.display = editor.style.display === "none" ? "flex" : "none";
    });
  }

  const repromptBtn = document.getElementById("reprompt-btn");
  if (repromptBtn) {
    repromptBtn.addEventListener("click", () => {
      const planSection = document.getElementById("plan-section");
      if (planSection) planSection.style.display = "none";
      _currentPlanText = "";
      const summaryEl = document.getElementById("plan-summary");
      if (summaryEl) { summaryEl.innerHTML = ""; summaryEl.style.display = "none"; }
    });
  }

  const applyJsonBtn = document.getElementById("apply-json-btn");
  if (applyJsonBtn) {
    applyJsonBtn.addEventListener("click", () => {
      const textarea = document.getElementById("plan-json-textarea");
      if (textarea) {
        _currentPlanText = textarea.value;
        renderPlanTree(_currentPlanText);
        const editor = document.getElementById("plan-json-editor");
        if (editor) editor.style.display = "none";
      }
    });
  }

  const cancelEditBtn = document.getElementById("cancel-edit-btn");
  if (cancelEditBtn) {
    cancelEditBtn.addEventListener("click", () => {
      const editor = document.getElementById("plan-json-editor");
      if (editor) editor.style.display = "none";
    });
  }

  const cancelGenBtn = document.getElementById("cancel-generation-btn");
  if (cancelGenBtn) {
    cancelGenBtn.addEventListener("click", async () => {
      await apiRequest("/api/generate/cancel", "POST");
      updateStatusBar("生成をキャンセルしました");
    });
  }

  // グローバル停止ボタン
  const globalStopBtn = document.getElementById("global-stop-btn");
  if (globalStopBtn) {
    globalStopBtn.addEventListener("click", () => {
      if (_activeCancel) _activeCancel();
      apiRequest("/api/generate/cancel", "POST").catch(() => {});
      _unlockUI();
      updateStatusBar("生成を停止しました");
    });
  }

  // ロールバックボタン
  const rollbackBtn = document.getElementById("rollback-btn");
  if (rollbackBtn) rollbackBtn.addEventListener("click", _doRollback);

  // Explainタブのボタン
  const buildIndexBtn = document.getElementById("build-index-btn");
  if (buildIndexBtn) buildIndexBtn.addEventListener("click", buildIndex);

  const generateReportBtn = document.getElementById("generate-report-btn");
  if (generateReportBtn) {
    generateReportBtn.addEventListener("click", () => {
      const indices = _getSelectedSectionIndices();
      const modelOverride = (document.getElementById("report-model-override") || {}).value || "";
      const lang = (document.getElementById("report-language-select") || {}).value || "ja";
      generateReport({
        sectionIndices: indices,
        model: modelOverride || undefined,
        lang,
      });
    });
  }

  const migrateVectorBtn = document.getElementById("migrate-vector-btn");
  if (migrateVectorBtn) migrateVectorBtn.addEventListener("click", migrateVectorIndex);

  // 履歴ボタン
  const reportHistoryBtn = document.getElementById("report-history-btn");
  if (reportHistoryBtn) reportHistoryBtn.addEventListener("click", openReportHistory);

  // 部分レポートバナーのボタン
  const resumeBtn = document.getElementById("report-resume-btn");
  if (resumeBtn) {
    resumeBtn.addEventListener("click", () => {
      generateReport({ resumeFrom: window._reportResumFrom || 0 });
    });
  }
  const regenBtn = document.getElementById("report-regen-btn");
  if (regenBtn) regenBtn.addEventListener("click", () => generateReport({}));
  const dismissBanner = document.getElementById("report-partial-dismiss");
  if (dismissBanner) dismissBanner.addEventListener("click", _hideSavedReportBanner);

  const dismissPlanBanner = document.getElementById("dismiss-plan-banner-btn");
  if (dismissPlanBanner) {
    dismissPlanBanner.addEventListener("click", () => {
      const banner = document.getElementById("saved-plan-banner");
      if (banner) banner.style.display = "none";
    });
  }

  const resumeGenBtn = document.getElementById("resume-generation-btn");
  if (resumeGenBtn) resumeGenBtn.addEventListener("click", _resumeFromSavedPlan);

  const restartGenBtn = document.getElementById("restart-generation-btn");
  if (restartGenBtn) restartGenBtn.addEventListener("click", _restartFromSavedPlan);

  // セクション選択パネルの全選択/解除
  const selectAll = document.getElementById("sections-select-all");
  if (selectAll) {
    selectAll.addEventListener("click", () => {
      document.querySelectorAll("#section-checkboxes input[type=checkbox]").forEach(cb => { cb.checked = true; });
      _updateSectionCount();
    });
  }
  const deselectAll = document.getElementById("sections-deselect-all");
  if (deselectAll) {
    deselectAll.addEventListener("click", () => {
      document.querySelectorAll("#section-checkboxes input[type=checkbox]").forEach(cb => { cb.checked = false; });
      _updateSectionCount();
    });
  }

  // 履歴パネルの閉じるボタン
  const histClose = document.getElementById("history-panel-close");
  if (histClose) histClose.addEventListener("click", () => {
    const panel = document.getElementById("report-history-panel");
    if (panel) panel.style.display = "none";
  });

  // 比較ボタン
  const compareBtn = document.getElementById("history-compare-btn");
  if (compareBtn) compareBtn.addEventListener("click", openCompareView);

  // 比較ビューを閉じる
  const compareClose = document.getElementById("compare-close-btn");
  if (compareClose) compareClose.addEventListener("click", () => {
    const view = document.getElementById("report-compare-view");
    if (view) view.style.display = "none";
  });

  // Resumeタブのボタン
  const continueGenBtn = document.getElementById("continue-generation-btn");
  if (continueGenBtn) continueGenBtn.addEventListener("click", continueGeneration);

  const viewReportBtn = document.getElementById("view-full-report-btn");
  if (viewReportBtn) {
    viewReportBtn.addEventListener("click", () => {
      switchTab("explain");
      buildIndex();
    });
  }

  const continueQaBtn = document.getElementById("continue-qa-btn");
  if (continueQaBtn) {
    continueQaBtn.addEventListener("click", () => {
      switchTab("explain");
    });
  }

  const generateNewBtn = document.getElementById("generate-new-files-btn");
  if (generateNewBtn) {
    generateNewBtn.addEventListener("click", () => switchTab("generate"));
  }

  const resumeCancelBtn = document.getElementById("resume-cancel-btn");
  if (resumeCancelBtn) {
    resumeCancelBtn.addEventListener("click", async () => {
      await apiRequest("/api/generate/cancel", "POST");
      updateStatusBar("生成をキャンセルしました");
    });
  }

  const modifyPlanBtn = document.getElementById("modify-plan-btn");
  if (modifyPlanBtn) {
    modifyPlanBtn.addEventListener("click", () => switchTab("generate"));
  }

  // Gitコミットボタン（ステータスバーダブルクリックで簡易コミット）
  const statusBar = document.querySelector(".status-bar");
  if (statusBar) {
    statusBar.addEventListener("dblclick", async () => {
      if (!_currentProjectRoot) return;
      try {
        const result = await apiRequest("/api/git/commit", "POST", {
          message: "LocalForge: 変更をコミット"
        });
        showAlert(`コミット完了: ${result.hash}`, "success");
        refreshGitLog();
        refreshProjectStatus();
      } catch (err) {
        showAlert(`コミットエラー: ${err.message}`, "error");
      }
    });
  }

  // CPU スレッドマネージャーのイベント設定
  const cpuSlider = document.getElementById("cpu-thread-slider");
  if (cpuSlider) {
    cpuSlider.addEventListener("input", () => {
      const val = parseInt(cpuSlider.value);
      const pct = Math.round((val / _cpuCount) * 100);
      const valueEl = document.getElementById("cpu-slider-value");
      if (valueEl) valueEl.textContent = `${val} スレッド (${pct}%)`;
    });
  }

  const cpuApplyBtn = document.getElementById("cpu-apply-btn");
  if (cpuApplyBtn) {
    cpuApplyBtn.addEventListener("click", () => {
      const s = document.getElementById("cpu-thread-slider");
      if (s) applyCpuThread(parseInt(s.value));
    });
  }

  const cpuAutoBtn = document.getElementById("cpu-auto-btn");
  if (cpuAutoBtn) {
    cpuAutoBtn.addEventListener("click", () => applyCpuThread(null));
  }

  // 生成ログモーダル
  const showLogsBtn = document.getElementById("show-logs-btn");
  if (showLogsBtn) showLogsBtn.addEventListener("click", showGenerationLogs);

  const logsModalClose = document.getElementById("logs-modal-close");
  const logsModal = document.getElementById("logs-modal");
  if (logsModalClose && logsModal) {
    logsModalClose.addEventListener("click", () => {
      logsModal.style.display = "none";
    });
  }

  // Ollamaライブ出力パネル初期化
  OllamaPanel.init();
  ProcessLog.init();

  // 初期化: モデル一覧とCPU情報を読み込む
  await loadModels();
  await loadCpuThreadInfo();

  // 起動時Ollamaヘルスチェック
  await checkOllamaHealth();

  // システム情報（GPU/RAM）の定期更新 — 初回は5秒後に遅延起動してサーバー負荷を下げる
  setTimeout(() => {
    refreshSysInfo();
    // GPU未検出時はポーリングを停止するため、_sysInfoIntervalIdで管理する
    window._sysInfoIntervalId = setInterval(refreshSysInfo, 60000);
  }, 5000);

  // リサイズ機能の初期化
  initResizers();

  // 生成ログを表示する
  async function showGenerationLogs() {
    try {
      const data = await apiRequest("/api/generate/logs");
      const logs = data.logs || [];
      const tbody = document.getElementById("logs-tbody");
      const summaryEl = document.getElementById("token-usage-summary");
      const modal = document.getElementById("logs-modal");

      if (modal) modal.style.display = "flex";
      if (!tbody || !summaryEl) return;

      tbody.innerHTML = logs.map(l => `
        <tr>
          <td>${new Date(l.timestamp).toLocaleString()}</td>
          <td>${escapeHtml(l.model)}</td>
          <td>${escapeHtml(l.operation)}</td>
          <td>${l.prompt_tokens_estimated}</td>
          <td>${l.response_time_ms ? Math.round(l.response_time_ms) : "-"}</td>
          <td>${escapeHtml(l.status)}</td>
        </tr>
      `).join("");

      // トークン使用量の集計
      const usageByModel = {};
      logs.forEach(l => {
        usageByModel[l.model] = (usageByModel[l.model] || 0) + (l.prompt_tokens_estimated || 0);
      });

      summaryEl.innerHTML = "<strong>モデル別推定トークン使用量:</strong><br>" +
        Object.entries(usageByModel).map(([m, t]) => `${escapeHtml(m)}: ${t} tokens`).join("<br>");

    } catch (err) {
      showAlert(`ログの取得に失敗しました: ${err.message}`, "error");
    }
  }

  // 保存されたレイアウト状態を復元
  restoreLayout();

  // テーマの初期化
  initTheme();

  updateStatusBar("LocalForge 準備完了 — フォルダを開いてください");
});

/**
 * GPU/RAM情報を取得してステータスバーを更新する。
 * GPUがない場合はシステムRAMを表示してポーリングを継続する（RAMは常に有用）。
 */
async function refreshSysInfo() {
  const vramEl = document.getElementById("status-vram");
  if (!vramEl) return;

  try {
    const data = await apiRequest("/api/project/sysinfo");

    if (data && data.gpu) {
      // GPU搭載デバイス: VRAMを表示
      const usedGb = (data.gpu.used / 1024).toFixed(1);
      const totalGb = (data.gpu.total / 1024).toFixed(1);
      const pct = Math.round((data.gpu.used / data.gpu.total) * 100);
      vramEl.textContent = `VRAM: ${usedGb} / ${totalGb} GB (${pct}%)`;
      vramEl.style.display = "inline";
    } else if (data && data.ram && data.ram.total > 0) {
      // CPU専用デバイス: システムRAMを表示
      const usedGb = (data.ram.used / 1024).toFixed(1);
      const totalGb = (data.ram.total / 1024).toFixed(1);
      const pct = Math.round((data.ram.used / data.ram.total) * 100);
      vramEl.textContent = `RAM: ${usedGb} / ${totalGb} GB (${pct}%)`;
      vramEl.style.display = "inline";

      // GPU未検出が確定したらVRAMポーリングを60秒から120秒に緩める
      if (data.cuda_available === false && window._sysInfoIntervalId) {
        clearInterval(window._sysInfoIntervalId);
        window._sysInfoIntervalId = setInterval(refreshSysInfo, 120000);
      }
    } else {
      vramEl.style.display = "none";
    }
  } catch (e) {
    vramEl.style.display = "none";
  }
}

/**
 * テーマの切り替え機能を初期化する。
 */
function initTheme() {
  const themeToggle = document.getElementById("theme-toggle");
  if (!themeToggle) return;

  const currentTheme = localStorage.getItem("localforge-theme") || "dark";
  if (currentTheme === "light") {
    document.body.classList.add("light-theme");
  }

  themeToggle.addEventListener("click", () => {
    document.body.classList.toggle("light-theme");
    const newTheme = document.body.classList.contains("light-theme") ? "light" : "dark";
    localStorage.setItem("localforge-theme", newTheme);
  });
}

/**
 * サイドバーのリサイズ機能を初期化する。
 */
function initResizers() {
  const layout = document.querySelector(".app-layout");
  const leftResizer = document.getElementById("resizer-left");
  const rightResizer = document.getElementById("resizer-right");
  const leftSidebar = document.getElementById("left-sidebar");
  const rightSidebar = document.getElementById("right-sidebar");

  if (!leftResizer || !rightResizer) return;

  // 左サイドバーのリサイズ
  leftResizer.addEventListener("mousedown", (e) => {
    e.preventDefault();
    leftResizer.classList.add("resizing");
    document.addEventListener("mousemove", resizeLeft);
    document.addEventListener("mouseup", stopResizeLeft);
  });

  function resizeLeft(e) {
    const newWidth = e.clientX;
    if (newWidth > 100 && newWidth < 600) {
      layout.style.setProperty("--sidebar-l", `${newWidth}px`);
    }
  }

  function stopResizeLeft() {
    leftResizer.classList.remove("resizing");
    document.removeEventListener("mousemove", resizeLeft);
    document.removeEventListener("mouseup", stopResizeLeft);
    saveLayout();
  }

  // 右サイドバーのリサイズ
  rightResizer.addEventListener("mousedown", (e) => {
    e.preventDefault();
    rightResizer.classList.add("resizing");
    document.addEventListener("mousemove", resizeRight);
    document.addEventListener("mouseup", stopResizeRight);
  });

  function resizeRight(e) {
    const newWidth = window.innerWidth - e.clientX;
    if (newWidth > 150 && newWidth < 600) {
      layout.style.setProperty("--sidebar-r", `${newWidth}px`);
    }
  }

  function stopResizeRight() {
    rightResizer.classList.remove("resizing");
    document.removeEventListener("mousemove", resizeRight);
    document.removeEventListener("mouseup", stopResizeRight);
    saveLayout();
  }
}

/**
 * レイアウト状態をlocalStorageに保存する。
 */
function saveLayout() {
  const layout = document.querySelector(".app-layout");
  const leftWidth = layout.style.getPropertyValue("--sidebar-l");
  const rightWidth = layout.style.getPropertyValue("--sidebar-r");

  if (leftWidth) localStorage.setItem("localforge-sidebar-l", leftWidth);
  if (rightWidth) localStorage.setItem("localforge-sidebar-r", rightWidth);
}

/**
 * localStorageからレイアウト状態を復元する。
 */
function restoreLayout() {
  const layout = document.querySelector(".app-layout");
  const leftWidth = localStorage.getItem("localforge-sidebar-l");
  const rightWidth = localStorage.getItem("localforge-sidebar-r");

  if (leftWidth) layout.style.setProperty("--sidebar-l", leftWidth);
  if (rightWidth) layout.style.setProperty("--sidebar-r", rightWidth);
}
