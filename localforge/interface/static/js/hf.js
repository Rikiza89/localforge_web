/**
 * hf.js — HuggingFace モデル管理 UI
 * プロバイダー切替、モデルカタログ、ダウンロード、手動ダウンロード案内を制御する。
 */

"use strict";

// ---------------------------------------------------------------------------
// 状態
// ---------------------------------------------------------------------------

let _activeProvider = "ollama";     // "ollama" | "huggingface"
let _hfDownloadES = null;           // 進行中のダウンロード EventSource
let _hfLoadedModelPath = "";        // 現在 HF クライアントにロード済みのモデルパス

// ---------------------------------------------------------------------------
// プロバイダー切替 UI
// ---------------------------------------------------------------------------

/**
 * プロバイダーボタンの表示状態を更新する。
 * @param {"ollama"|"huggingface"} provider
 */
function _updateProviderUI(provider) {
  _activeProvider = provider;

  document.querySelectorAll(".provider-btn").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.provider === provider);
  });

  const modelSelect = document.getElementById("model-selector");
  const hfBtn = document.getElementById("hf-model-select-btn");
  const hfLabel = document.getElementById("hf-loaded-model-label");

  if (provider === "ollama") {
    if (modelSelect) modelSelect.style.display = "";
    if (hfBtn)       hfBtn.style.display = "none";
    if (hfLabel)     hfLabel.style.display = "none";
  } else {
    if (modelSelect) modelSelect.style.display = "none";
    if (hfBtn)       hfBtn.style.display = "";
    // ロード済みモデルラベル
    if (hfLabel) {
      if (_hfLoadedModelPath) {
        const fname = _hfLoadedModelPath.split("/").pop().split("\\").pop();
        hfLabel.textContent = `🤗 ${fname}`;
        hfLabel.title = _hfLoadedModelPath;
        hfLabel.style.display = "";
      } else {
        hfLabel.textContent = "モデル未ロード";
        hfLabel.style.display = "";
      }
    }
  }
}

/**
 * プロバイダーをサーバーに送信して切り替える。
 * @param {"ollama"|"huggingface"} provider
 */
async function switchProvider(provider) {
  if (provider === _activeProvider) return;
  try {
    updateStatusBar(`プロバイダーを切り替え中: ${provider}...`);
    await apiRequest("/api/hf/provider", "POST", { provider });
    _updateProviderUI(provider);
    updateStatusBar(`プロバイダー: ${provider}`);
    if (provider === "ollama") {
      await loadModels();
    }
  } catch (err) {
    showAlert(`プロバイダー切替エラー: ${err.message}`, "error");
  }
}

// ---------------------------------------------------------------------------
// HF モーダル制御
// ---------------------------------------------------------------------------

function openHFModal() {
  const modal = document.getElementById("hf-modal");
  if (modal) {
    modal.style.display = "flex";
    loadHFModels();
    populateManualSelector();
  }
}

function closeHFModal() {
  const modal = document.getElementById("hf-modal");
  if (modal) modal.style.display = "none";
}

/**
 * HF タブを切り替える。
 * @param {string} tabName  "catalog" | "local" | "manual"
 */
function switchHFTab(tabName) {
  document.querySelectorAll(".hf-tab-btn").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.hfTab === tabName);
  });
  document.querySelectorAll(".hf-tab-content").forEach(el => {
    el.classList.toggle("active", el.id === `hf-tab-${tabName}`);
  });
}

// ---------------------------------------------------------------------------
// モデルカタログの読み込みと表示
// ---------------------------------------------------------------------------

async function loadHFModels() {
  try {
    const data = await apiRequest("/api/hf/models");
    _hfLoadedModelPath = data.loaded_model || "";
    renderCatalog(data.catalog || []);
    renderLocalModels(data.local || []);
    // モデルディレクトリラベルを実パスで更新
    const dirEl = document.getElementById("hf-models-dir");
    if (dirEl && data.models_dir) dirEl.textContent = data.models_dir;
    // プロバイダー状態を同期
    if (data.active_provider !== _activeProvider) {
      _updateProviderUI(data.active_provider);
    }
  } catch (err) {
    const el = document.getElementById("hf-catalog-list");
    if (el) el.innerHTML = `<div class="empty-state">読み込みエラー: ${escapeHtml(err.message)}</div>`;
  }
}

function renderCatalog(catalog) {
  const el = document.getElementById("hf-catalog-list");
  if (!el) return;

  if (catalog.length === 0) {
    el.innerHTML = '<div class="empty-state">カタログが空です</div>';
    return;
  }

  el.innerHTML = catalog.map(m => {
    const isLoaded    = m.local_path && m.local_path === _hfLoadedModelPath;
    const isDownloaded = m.downloaded;
    const cardClass   = isLoaded ? "loaded" : isDownloaded ? "downloaded" : "";
    const badgeClass  = isLoaded ? "loaded" : isDownloaded ? "downloaded" : "not-downloaded";
    const badgeText   = isLoaded ? "✓ ロード済み" : isDownloaded ? "✓ ダウンロード済み" : "未ダウンロード";

    const tags = (m.tags || []).map(t =>
      `<span class="hf-tag tag-${escapeHtml(t)}">${escapeHtml(t)}</span>`
    ).join("");

    const actionBtns = isLoaded
      ? `<button class="btn btn-secondary btn-sm" onclick="unloadHFModel()">アンロード</button>`
      : isDownloaded
        ? `<button class="btn btn-primary btn-sm" onclick="loadHFModel('${escapeHtml(m.local_path)}')">ロード</button>
           <button class="btn btn-secondary btn-sm" onclick="startHFDownload('${escapeHtml(m.id)}')">再DL</button>`
        : `<button class="btn btn-primary btn-sm" onclick="startHFDownload('${escapeHtml(m.id)}')">ダウンロード</button>
           <button class="btn btn-secondary btn-sm" onclick="showManualDownload('${escapeHtml(m.id)}')">手動DL</button>`;

    return `
      <div class="hf-model-card ${cardClass}" id="hf-card-${escapeHtml(m.id)}">
        <div class="hf-model-info">
          <div class="hf-model-name">${escapeHtml(m.name)} ${tags}</div>
          <div class="hf-model-desc">${escapeHtml(m.description)}</div>
          <div class="hf-model-meta">
            <span>💾 ${m.size_gb} GB</span>
            <span>📁 ${escapeHtml(m.filename)}</span>
            <span>✦ ${escapeHtml(m.recommended_for || "")}</span>
          </div>
        </div>
        <div class="hf-model-actions">
          <span class="hf-status-badge ${badgeClass}">${badgeText}</span>
          ${actionBtns}
        </div>
      </div>`;
  }).join("");
}

function renderLocalModels(local) {
  const el = document.getElementById("hf-local-list");
  if (!el) return;

  if (local.length === 0) {
    el.innerHTML = '<div class="empty-state">ローカルモデルが見つかりません。カタログからダウンロードしてください。</div>';
    return;
  }

  el.innerHTML = local.map(m => {
    const isLoaded = m.path === _hfLoadedModelPath;
    const cardClass = isLoaded ? "loaded" : "";
    const fname = m.filename || m.path.split("/").pop();
    return `
      <div class="hf-model-card ${cardClass}">
        <div class="hf-model-info">
          <div class="hf-model-name">${escapeHtml(m.name || fname)}</div>
          <div class="hf-model-desc">${escapeHtml(m.description || "ローカルモデル")}</div>
          <div class="hf-model-meta">
            <span>💾 ${m.size_gb} GB</span>
            <span class="hf-url-link" title="${escapeHtml(m.path)}">${escapeHtml(m.path)}</span>
          </div>
        </div>
        <div class="hf-model-actions">
          ${isLoaded
            ? `<span class="hf-status-badge loaded">✓ ロード済み</span>
               <button class="btn btn-secondary btn-sm" onclick="unloadHFModel()">アンロード</button>`
            : `<button class="btn btn-primary btn-sm" onclick="loadHFModel('${escapeHtml(m.path)}')">ロード</button>`
          }
        </div>
      </div>`;
  }).join("");
}

// ---------------------------------------------------------------------------
// ダウンロード
// ---------------------------------------------------------------------------

function startHFDownload(modelId) {
  // 既に進行中のダウンロードを閉じる
  if (_hfDownloadES) {
    _hfDownloadES.close();
    _hfDownloadES = null;
  }

  const panel = document.getElementById("hf-download-panel");
  const nameEl = document.getElementById("hf-download-model-name");
  const statusEl = document.getElementById("hf-download-status");
  const progressBar = document.getElementById("hf-progress-bar");

  if (panel) panel.style.display = "";
  if (nameEl) nameEl.textContent = `${modelId} をダウンロード中...`;
  if (statusEl) statusEl.textContent = "接続中...";
  if (progressBar) progressBar.style.width = "0%";

  updateStatusBar(`${modelId} をダウンロード中...`);

  _hfDownloadES = new EventSource(`/api/hf/download?model_id=${encodeURIComponent(modelId)}`);

  _hfDownloadES.addEventListener("status", e => {
    const d = JSON.parse(e.data);
    if (statusEl) statusEl.textContent = d.status || "";
    updateStatusBar(d.status || "");
  });

  _hfDownloadES.addEventListener("progress", e => {
    const d = JSON.parse(e.data);
    if (progressBar && d.total > 0) {
      const pct = Math.round((d.done / d.total) * 100);
      progressBar.style.width = `${pct}%`;
    }
    if (statusEl) {
      const speed = d.speed_mbps ? ` — ${d.speed_mbps.toFixed(1)} MB/s` : "";
      const pct = d.total > 0 ? ` (${Math.round((d.done / d.total) * 100)}%)` : "";
      statusEl.textContent = `${_fmtBytes(d.done)} / ${_fmtBytes(d.total)}${pct}${speed}`;
    }
  });

  _hfDownloadES.addEventListener("done", e => {
    _hfDownloadES.close();
    _hfDownloadES = null;
    const d = JSON.parse(e.data);
    if (progressBar) progressBar.style.width = "100%";
    if (statusEl) statusEl.textContent = "ダウンロード完了！";
    updateStatusBar("ダウンロード完了");
    showAlert(`ダウンロード完了: ${modelId}`, "success");

    // 自動でモデルをロード
    if (d.path) {
      setTimeout(() => loadHFModel(d.path), 500);
    }
    // カードを再レンダリング
    setTimeout(() => loadHFModels(), 800);
  });

  _hfDownloadES.addEventListener("error", e => {
    _hfDownloadES.close();
    _hfDownloadES = null;
    try {
      const d = JSON.parse(e.data);
      if (statusEl) statusEl.textContent = `エラー: ${d.error}`;
      updateStatusBar("ダウンロードエラー");

      if (d.proxy_error) {
        // プロキシエラー → 手動ダウンロードタブを表示
        showAlert("プロキシによりブロックされました。手動ダウンロード手順を表示します。", "warning");
        if (d.instructions) {
          _showInstructions(d.instructions);
          switchHFTab("manual");
        } else {
          showManualDownload(modelId);
        }
      } else {
        showAlert(`ダウンロードエラー: ${d.error}`, "error");
      }
    } catch (_) {
      if (statusEl) statusEl.textContent = "ダウンロードエラー";
    }
  });
}

function cancelHFDownload() {
  if (_hfDownloadES) {
    _hfDownloadES.close();
    _hfDownloadES = null;
  }
  const panel = document.getElementById("hf-download-panel");
  if (panel) panel.style.display = "none";
  updateStatusBar("ダウンロードをキャンセルしました");
}

// ---------------------------------------------------------------------------
// モデルロード / アンロード
// ---------------------------------------------------------------------------

async function loadHFModel(path) {
  try {
    updateStatusBar(`モデルをロード中: ${path.split("/").pop()}...`);
    const data = await apiRequest("/api/hf/load", "POST", { path });
    _hfLoadedModelPath = data.path || path;
    _updateProviderUI("huggingface");
    updateStatusBar(`モデルをロードしました: ${path.split("/").pop()}`);
    showAlert(`🤗 モデルをロードしました`, "success");
    closeHFModal();
  } catch (err) {
    showAlert(`モデルロードエラー: ${err.message}`, "error");
    updateStatusBar("モデルロードエラー");
  }
}

async function unloadHFModel() {
  try {
    await apiRequest("/api/hf/unload", "POST", {});
    _hfLoadedModelPath = "";
    _updateProviderUI("huggingface");
    updateStatusBar("HF モデルをアンロードしました");
    showAlert("HuggingFace モデルをアンロードしました", "success");
    await loadHFModels();
  } catch (err) {
    showAlert(`アンロードエラー: ${err.message}`, "error");
  }
}

// ---------------------------------------------------------------------------
// 手動ダウンロード案内
// ---------------------------------------------------------------------------

function populateManualSelector() {
  const sel = document.getElementById("hf-manual-model-selector");
  if (!sel) return;
  // カタログからオプションを生成（既にレンダリング済みの場合はスキップ）
  if (sel.options.length > 1) return;

  apiRequest("/api/hf/models").then(data => {
    const catalog = data.catalog || [];
    catalog.forEach(m => {
      const opt = document.createElement("option");
      opt.value = m.id;
      opt.textContent = `${m.name} (${m.size_gb} GB)`;
      sel.appendChild(opt);
    });
  }).catch(() => {});
}

async function getManualInstructions() {
  const sel = document.getElementById("hf-manual-model-selector");
  const modelId = sel ? sel.value : "";
  if (!modelId) {
    showAlert("モデルを選択してください", "warning");
    return;
  }
  await showManualDownload(modelId);
}

async function showManualDownload(modelId) {
  switchHFTab("manual");
  const sel = document.getElementById("hf-manual-model-selector");
  if (sel) sel.value = modelId;

  try {
    const data = await apiRequest("/api/hf/instructions", "POST", { model_id: modelId });
    _showInstructions(data.instructions);
  } catch (err) {
    showAlert(`手順取得エラー: ${err.message}`, "error");
  }
}

function _showInstructions(inst) {
  const panel = document.getElementById("hf-instructions-panel");
  const content = document.getElementById("hf-instructions-content");
  if (!panel || !content) return;

  panel.style.display = "";

  // Store dest_path for later check
  panel.dataset.destPath = inst.dest_path || "";

  content.innerHTML = `
<b>${escapeHtml(inst.model_name)}</b> — ${inst.size_gb} GB

<b>1. ブラウザでダウンロード:</b>
   <a href="${escapeHtml(inst.download_url)}" target="_blank" class="hf-url-link">${escapeHtml(inst.download_url)}</a>
   <button class="hf-copy-btn" onclick="_copyText('${escapeHtml(inst.download_url)}')">コピー</button>

<b>2. wget でダウンロード:</b>
   <code>${escapeHtml(inst.wget_cmd)}</code>
   <button class="hf-copy-btn" onclick="_copyText('${escapeHtml(inst.wget_cmd)}')">コピー</button>

<b>3. curl でダウンロード:</b>
   <code>${escapeHtml(inst.curl_cmd)}</code>
   <button class="hf-copy-btn" onclick="_copyText('${escapeHtml(inst.curl_cmd)}')">コピー</button>

<b>4. ダウンロード先フォルダ（自動作成済み）:</b>
   <code>${escapeHtml(inst.dest_dir)}</code>
   <button class="hf-copy-btn" onclick="_copyText('${escapeHtml(inst.dest_dir)}')">コピー</button>

<b>5. 配置後のファイルパス:</b>
   <code>${escapeHtml(inst.dest_path)}</code>

<b>6.</b> ファイルを上記フォルダに配置したら下の「確認してロード」ボタンをクリックしてください。`;

  content.scrollTop = 0;
}

async function checkAndLoadManualModel() {
  const panel = document.getElementById("hf-instructions-panel");
  const destPath = panel ? panel.dataset.destPath : "";
  if (!destPath) {
    showAlert("先にモデルの手順を表示してください", "warning");
    return;
  }

  // スキャンして確認
  try {
    const data = await apiRequest("/api/hf/scan");
    const found = (data.local || []).find(m => m.path === destPath);
    if (found) {
      await loadHFModel(destPath);
    } else {
      showAlert(
        `ファイルが見つかりません:\n${destPath}\n\nファイルを正しいフォルダに配置してください。`,
        "warning",
      );
    }
  } catch (err) {
    showAlert(`スキャンエラー: ${err.message}`, "error");
  }
}

// ---------------------------------------------------------------------------
// ユーティリティ
// ---------------------------------------------------------------------------

function _fmtBytes(bytes) {
  if (!bytes) return "0 B";
  const gb = bytes / (1024 ** 3);
  if (gb >= 1) return `${gb.toFixed(2)} GB`;
  const mb = bytes / (1024 ** 2);
  if (mb >= 1) return `${mb.toFixed(1)} MB`;
  return `${Math.round(bytes / 1024)} KB`;
}

function _copyText(text) {
  navigator.clipboard.writeText(text).then(() => {
    updateStatusBar("クリップボードにコピーしました");
  }).catch(() => {
    // フォールバック
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    document.execCommand("copy");
    document.body.removeChild(ta);
    updateStatusBar("コピーしました");
  });
}

// ---------------------------------------------------------------------------
// ライブ HuggingFace 検索
// ---------------------------------------------------------------------------

let _browseExpandedRepo = null;   // 現在ファイルリストを展開中の repo_id

async function hfSearch(query) {
  const statusEl  = document.getElementById("hf-browse-status");
  const resultsEl = document.getElementById("hf-browse-results");
  if (!resultsEl) return;

  resultsEl.innerHTML = '<div class="empty-state">検索中...</div>';
  if (statusEl) { statusEl.style.display = ""; statusEl.textContent = "HuggingFace API に接続中..."; }

  try {
    const url = query
      ? `/api/hf/search?q=${encodeURIComponent(query)}&limit=20`
      : `/api/hf/search?limit=20`;
    const data = await apiRequest(url);

    if (!data.online) {
      if (statusEl) statusEl.textContent = `オフライン: ${data.error || "HuggingFace API に接続できません"}`;
      resultsEl.innerHTML = '<div class="empty-state">HuggingFace API に接続できません。ネットワークを確認してください。</div>';
      return;
    }

    const models = data.models || [];
    if (statusEl) statusEl.textContent = `${models.length} 件のモデルが見つかりました`;

    if (models.length === 0) {
      resultsEl.innerHTML = '<div class="empty-state">該当するモデルが見つかりません。別のキーワードを試してください。</div>';
      return;
    }

    resultsEl.innerHTML = models.map(m => `
      <div class="hf-model-card hf-browse-card" id="hf-browse-card-${CSS.escape(m.repo_id)}">
        <div class="hf-model-info">
          <div class="hf-model-name">${escapeHtml(m.name)}</div>
          <div class="hf-model-desc" style="font-size:11px;color:var(--text-muted);">${escapeHtml(m.repo_id)}</div>
          <div class="hf-model-meta">
            <span>⬇ ${_fmtNum(m.downloads)}</span>
            <span>♥ ${_fmtNum(m.likes)}</span>
          </div>
        </div>
        <div class="hf-model-actions">
          <button class="btn btn-secondary btn-sm" onclick="toggleBrowseFiles('${escapeHtml(m.repo_id)}')">
            ファイル一覧
          </button>
        </div>
      </div>
      <div class="hf-file-list" id="hf-files-${CSS.escape(m.repo_id)}" style="display:none;"></div>
    `).join("");

  } catch (err) {
    if (statusEl) statusEl.textContent = `エラー: ${err.message}`;
    resultsEl.innerHTML = `<div class="empty-state">エラー: ${escapeHtml(err.message)}</div>`;
  }
}

async function toggleBrowseFiles(repoId) {
  const filesEl = document.getElementById(`hf-files-${CSS.escape(repoId)}`);
  if (!filesEl) return;

  // 既に展開中なら閉じる
  if (filesEl.style.display !== "none") {
    filesEl.style.display = "none";
    _browseExpandedRepo = null;
    return;
  }

  // 他を閉じる
  if (_browseExpandedRepo) {
    const prev = document.getElementById(`hf-files-${CSS.escape(_browseExpandedRepo)}`);
    if (prev) prev.style.display = "none";
  }
  _browseExpandedRepo = repoId;

  filesEl.style.display = "";
  filesEl.innerHTML = '<div style="padding:8px;color:var(--text-muted);font-size:12px;">ファイル一覧を取得中...</div>';

  try {
    const data = await apiRequest(`/api/hf/model-files?repo_id=${encodeURIComponent(repoId)}`);
    const files = data.files || [];

    if (files.length === 0) {
      filesEl.innerHTML = '<div style="padding:8px;color:var(--text-muted);font-size:12px;">GGUF ファイルが見つかりません（ファイルサイズが 19 GB を超えているか、ファイルが存在しません）。</div>';
      return;
    }

    filesEl.innerHTML = `
      <div class="hf-file-list-inner">
        ${files.map(f => {
          const sizeStr  = f.size_gb != null ? `${f.size_gb} GB` : "サイズ不明";
          const quantBadge = f.quant
            ? `<span class="hf-tag tag-${f.quant.startsWith('Q4') || f.quant.startsWith('Q5') ? 'recommended' : 'balanced'}">${escapeHtml(f.quant)}</span>`
            : "";
          return `
            <div class="hf-file-row ${f.downloaded ? 'downloaded' : ''}">
              <div class="hf-file-info">
                <span class="hf-file-name">${escapeHtml(f.filename)}</span>
                ${quantBadge}
                <span class="hf-file-size">${sizeStr}</span>
                ${f.downloaded ? '<span class="hf-status-badge downloaded" style="font-size:10px;">✓ ダウンロード済み</span>' : ""}
              </div>
              <div class="hf-file-actions">
                ${f.downloaded
                  ? `<button class="btn btn-primary btn-sm" onclick="loadHFModel('${escapeHtml(f.local_path)}')">ロード</button>`
                  : `<button class="btn btn-primary btn-sm"
                       onclick="startHFDownloadDirect('${escapeHtml(repoId)}','${escapeHtml(f.filename)}')">
                       ダウンロード
                     </button>
                     <button class="btn btn-secondary btn-sm"
                       onclick="showFileManualDownload('${escapeHtml(repoId)}','${escapeHtml(f.filename)}','${escapeHtml(f.size_gb || '')}')">
                       手動DL
                     </button>`
                }
              </div>
            </div>`;
        }).join("")}
      </div>`;
  } catch (err) {
    filesEl.innerHTML = `<div style="padding:8px;color:var(--danger);font-size:12px;">エラー: ${escapeHtml(err.message)}</div>`;
  }
}

function startHFDownloadDirect(repoId, filename) {
  // reuse existing SSE download — pass repo_id+filename instead of model_id
  if (_hfDownloadES) {
    _hfDownloadES.close();
    _hfDownloadES = null;
  }

  const panel     = document.getElementById("hf-download-panel");
  const nameEl    = document.getElementById("hf-download-model-name");
  const statusEl  = document.getElementById("hf-download-status");
  const progressBar = document.getElementById("hf-progress-bar");

  if (panel)       panel.style.display = "";
  if (nameEl)      nameEl.textContent  = filename;
  if (statusEl)    statusEl.textContent = "接続中...";
  if (progressBar) progressBar.style.width = "0%";

  updateStatusBar(`ダウンロード中: ${filename}`);

  const url = `/api/hf/download?repo_id=${encodeURIComponent(repoId)}&filename=${encodeURIComponent(filename)}`;
  _hfDownloadES = new EventSource(url);

  _hfDownloadES.addEventListener("status", e => {
    const d = JSON.parse(e.data);
    if (statusEl) statusEl.textContent = d.status || "";
  });

  _hfDownloadES.addEventListener("done", e => {
    _hfDownloadES.close();
    _hfDownloadES = null;
    const d = JSON.parse(e.data);
    if (progressBar) progressBar.style.width = "100%";
    if (statusEl)    statusEl.textContent = "完了！";
    showAlert(`ダウンロード完了: ${filename}`, "success");
    updateStatusBar("ダウンロード完了");
    if (d.path) setTimeout(() => loadHFModel(d.path), 500);
    // ファイルリストを再読み込み
    if (_browseExpandedRepo === repoId) {
      _browseExpandedRepo = null;
      setTimeout(() => toggleBrowseFiles(repoId), 800);
    }
  });

  _hfDownloadES.addEventListener("error", e => {
    _hfDownloadES.close();
    _hfDownloadES = null;
    try {
      const d = JSON.parse(e.data);
      if (statusEl) statusEl.textContent = `エラー: ${d.error}`;
      if (d.proxy_error) {
        showAlert("プロキシによりブロックされました。手動ダウンロード手順を表示します。", "warning");
        if (d.instructions) {
          _showInstructions(d.instructions);
          switchHFTab("manual");
        } else {
          showFileManualDownload(repoId, filename, null);
        }
      } else {
        showAlert(`ダウンロードエラー: ${d.error}`, "error");
      }
    } catch (_) {
      if (statusEl) statusEl.textContent = "エラー";
    }
  });
}

async function showFileManualDownload(repoId, filename, sizeGb) {
  switchHFTab("manual");
  try {
    const data = await apiRequest("/api/hf/file-instructions", "POST", {
      repo_id: repoId, filename, size_gb: sizeGb ? parseFloat(sizeGb) : null,
    });
    _showInstructions(data.instructions);
    // Update the check button dest path
    const panel = document.getElementById("hf-instructions-panel");
    if (panel) panel.dataset.destPath = data.instructions.dest_path || "";
  } catch (err) {
    showAlert(`手順取得エラー: ${err.message}`, "error");
  }
}

function _fmtNum(n) {
  if (!n) return "0";
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000)     return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

// ---------------------------------------------------------------------------
// 初期化（DOMContentLoaded から呼ばれる）
// ---------------------------------------------------------------------------

function initHFUI() {
  // プロバイダーボタン
  document.querySelectorAll(".provider-btn").forEach(btn => {
    btn.addEventListener("click", () => switchProvider(btn.dataset.provider));
  });

  // HF モデル選択ボタン
  const hfSelectBtn = document.getElementById("hf-model-select-btn");
  if (hfSelectBtn) hfSelectBtn.addEventListener("click", openHFModal);

  // モーダル閉じる
  const closeBtn = document.getElementById("hf-modal-close");
  if (closeBtn) closeBtn.addEventListener("click", closeHFModal);

  // モーダル外クリックで閉じる
  const hfModal = document.getElementById("hf-modal");
  if (hfModal) {
    hfModal.addEventListener("click", e => {
      if (e.target === hfModal) closeHFModal();
    });
  }

  // HF タブ切替
  document.querySelectorAll(".hf-tab-btn").forEach(btn => {
    btn.addEventListener("click", () => switchHFTab(btn.dataset.hfTab));
  });

  // ダウンロードキャンセル
  const cancelBtn = document.getElementById("hf-cancel-download-btn");
  if (cancelBtn) cancelBtn.addEventListener("click", cancelHFDownload);

  // 再スキャンボタン
  const rescanBtn = document.getElementById("hf-rescan-btn");
  if (rescanBtn) {
    rescanBtn.addEventListener("click", async () => {
      const data = await apiRequest("/api/hf/scan").catch(() => ({ local: [] }));
      renderLocalModels(data.local || []);
      updateStatusBar("ローカルモデルを再スキャンしました");
    });
  }

  // ライブ検索
  const searchBtn = document.getElementById("hf-search-btn");
  if (searchBtn) {
    searchBtn.addEventListener("click", () => {
      const q = document.getElementById("hf-search-input")?.value?.trim() || "";
      hfSearch(q);
    });
  }
  const topBtn = document.getElementById("hf-search-top-btn");
  if (topBtn) topBtn.addEventListener("click", () => hfSearch(""));

  const searchInput = document.getElementById("hf-search-input");
  if (searchInput) {
    searchInput.addEventListener("keydown", e => {
      if (e.key === "Enter") {
        hfSearch(e.target.value.trim());
      }
    });
  }

  // 手動ダウンロード手順取得
  const instrBtn = document.getElementById("hf-get-instructions-btn");
  if (instrBtn) instrBtn.addEventListener("click", getManualInstructions);

  // 手動ダウンロード確認ボタン
  const checkBtn = document.getElementById("hf-check-file-btn");
  if (checkBtn) checkBtn.addEventListener("click", checkAndLoadManualModel);

  // 起動時 HF 状態を確認
  apiRequest("/api/hf/status").then(data => {
    _hfLoadedModelPath = data.loaded_model || "";
    _updateProviderUI(data.active_provider || "ollama");
  }).catch(() => {});
}
