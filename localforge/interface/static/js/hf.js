/**
 * hf.js — HuggingFace モデル管理 UI (safetensors 形式)
 * プロバイダー切替、モデルカタログ、リポジトリダウンロード、手動ダウンロード案内を制御する。
 */

"use strict";

// ---------------------------------------------------------------------------
// 状態
// ---------------------------------------------------------------------------

let _activeProvider    = "ollama";
let _hfDownloadES      = null;
let _hfLoadedModelPath = "";

// ---------------------------------------------------------------------------
// プロバイダー切替 UI
// ---------------------------------------------------------------------------

function _updateProviderUI(provider) {
  _activeProvider = provider;

  document.querySelectorAll(".provider-btn").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.provider === provider);
  });

  const modelSelect = document.getElementById("model-selector");
  const hfBtn       = document.getElementById("hf-model-select-btn");
  const hfLabel     = document.getElementById("hf-loaded-model-label");

  if (provider === "ollama") {
    if (modelSelect) modelSelect.style.display = "";
    if (hfBtn)       hfBtn.style.display = "none";
    if (hfLabel)     hfLabel.style.display = "none";
  } else {
    if (modelSelect) modelSelect.style.display = "none";
    if (hfBtn)       hfBtn.style.display = "";
    if (hfLabel) {
      if (_hfLoadedModelPath) {
        const parts = _hfLoadedModelPath.replace(/\\/g, "/").split("/");
        hfLabel.textContent = `🤗 ${parts[parts.length - 1]}`;
        hfLabel.title = _hfLoadedModelPath;
        hfLabel.style.display = "";
      } else {
        hfLabel.textContent = "モデル未ロード";
        hfLabel.style.display = "";
      }
    }
  }
}

async function switchProvider(provider) {
  if (provider === _activeProvider) return;
  try {
    updateStatusBar(`プロバイダーを切り替え中: ${provider}...`);
    await apiRequest("/api/hf/provider", "POST", { provider });
    _updateProviderUI(provider);
    updateStatusBar(`プロバイダー: ${provider}`);
    if (provider === "ollama") await loadModels();
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
  }
}

function closeHFModal() {
  const modal = document.getElementById("hf-modal");
  if (modal) modal.style.display = "none";
}

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
    const dirEl = document.getElementById("hf-models-dir");
    if (dirEl && data.models_dir) dirEl.textContent = data.models_dir;
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
    const isLoaded     = m.local_path && m.local_path === _hfLoadedModelPath;
    const isDownloaded = m.downloaded;
    const cardClass    = isLoaded ? "loaded" : isDownloaded ? "downloaded" : "";
    const badgeClass   = isLoaded ? "loaded" : isDownloaded ? "downloaded" : "not-downloaded";
    const badgeText    = isLoaded ? "✓ ロード済み" : isDownloaded ? "✓ ダウンロード済み" : "未ダウンロード";
    const tags = (m.tags || []).map(t =>
      `<span class="hf-tag tag-${escapeHtml(t)}">${escapeHtml(t)}</span>`
    ).join("");

    const actionBtns = isLoaded
      ? `<button class="btn btn-secondary btn-sm" onclick="unloadHFModel()">アンロード</button>`
      : isDownloaded
        ? `<button class="btn btn-primary btn-sm" data-path="${escapeHtml(m.local_path)}" onclick="loadHFModel(this.dataset.path)">ロード</button>
           <button class="btn btn-secondary btn-sm" data-id="${escapeHtml(m.id)}" onclick="startHFDownload(this.dataset.id)">再DL</button>`
        : `<button class="btn btn-primary btn-sm" data-id="${escapeHtml(m.id)}" onclick="startHFDownload(this.dataset.id)">ダウンロード</button>
           <button class="btn btn-secondary btn-sm" data-repo="${escapeHtml(m.repo_id)}" data-name="${escapeHtml(m.name)}" onclick="showManualDownload(this.dataset.repo, this.dataset.name)">手動DL</button>`;

    return `
      <div class="hf-model-card ${cardClass}" id="hf-card-${escapeHtml(m.id)}">
        <div class="hf-model-info">
          <div class="hf-model-name">${escapeHtml(m.name)} ${tags}</div>
          <div class="hf-model-desc">${escapeHtml(m.description)}</div>
          <div class="hf-model-meta">
            <span>💾 ${m.size_gb} GB</span>
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
    return `
      <div class="hf-model-card ${cardClass}">
        <div class="hf-model-info">
          <div class="hf-model-name">${escapeHtml(m.name)}</div>
          <div class="hf-model-desc">${escapeHtml(m.description || "ローカルモデル")}</div>
          <div class="hf-model-meta">
            <span>💾 ${m.size_gb} GB</span>
          </div>
        </div>
        <div class="hf-model-actions">
          ${isLoaded
            ? `<span class="hf-status-badge loaded">✓ ロード済み</span>
               <button class="btn btn-secondary btn-sm" onclick="unloadHFModel()">アンロード</button>`
            : `<button class="btn btn-primary btn-sm" data-path="${escapeHtml(m.path)}" onclick="loadHFModel(this.dataset.path)">ロード</button>`
          }
        </div>
      </div>`;
  }).join("");
}

// ---------------------------------------------------------------------------
// ダウンロード（カタログ: model_id / ライブ検索: repo_id）
// ---------------------------------------------------------------------------

function _startDownloadSSE(url, displayName) {
  if (_hfDownloadES) { _hfDownloadES.close(); _hfDownloadES = null; }

  const panel     = document.getElementById("hf-download-panel");
  const nameEl    = document.getElementById("hf-download-model-name");
  const statusEl  = document.getElementById("hf-download-status");
  const progressBar = document.getElementById("hf-progress-bar");

  if (panel)       panel.style.display = "";
  if (nameEl)      nameEl.textContent   = displayName;
  if (statusEl)    statusEl.textContent = "接続中...";
  if (progressBar) progressBar.style.width = "0%";
  updateStatusBar(`ダウンロード中: ${displayName}`);

  _hfDownloadES = new EventSource(url);

  _hfDownloadES.addEventListener("status", e => {
    const d = JSON.parse(e.data);
    if (statusEl) statusEl.textContent = d.status || "";
    updateStatusBar(d.status || "");
  });

  _hfDownloadES.addEventListener("progress", e => {
    const d = JSON.parse(e.data);
    if (progressBar && d.total > 0) {
      progressBar.style.width = `${Math.round((d.done / d.total) * 100)}%`;
    }
    if (statusEl && d.total > 0) {
      statusEl.textContent = `${_fmtBytes(d.done)} / ${_fmtBytes(d.total)} (${Math.round((d.done / d.total) * 100)}%)`;
    }
  });

  _hfDownloadES.addEventListener("done", e => {
    _hfDownloadES.close(); _hfDownloadES = null;
    const d = JSON.parse(e.data);
    if (progressBar) progressBar.style.width = "100%";
    if (statusEl)    statusEl.textContent = "ダウンロード完了！";
    updateStatusBar("ダウンロード完了");
    showAlert(`ダウンロード完了: ${displayName}`, "success");
    if (d.path) setTimeout(() => loadHFModel(d.path), 500);
    setTimeout(() => loadHFModels(), 800);
  });

  _hfDownloadES.addEventListener("error", e => {
    _hfDownloadES.close(); _hfDownloadES = null;
    try {
      const d = JSON.parse(e.data);
      if (statusEl) statusEl.textContent = `エラー: ${d.error}`;
      updateStatusBar("ダウンロードエラー");
      if (d.proxy_error) {
        showAlert("プロキシによりブロックされました。手動ダウンロード手順を表示します。", "warning");
        if (d.instructions) { _showInstructions(d.instructions); switchHFTab("manual"); }
      } else {
        showAlert(`ダウンロードエラー: ${d.error}`, "error");
      }
    } catch (_) {
      if (statusEl) statusEl.textContent = "エラー";
    }
  });
}

function startHFDownload(modelId) {
  const model = /* catalog lookup via DOM */ null;
  _startDownloadSSE(
    `/api/hf/download?model_id=${encodeURIComponent(modelId)}`,
    modelId,
  );
}

function startHFDownloadRepo(repoId) {
  _startDownloadSSE(
    `/api/hf/download?repo_id=${encodeURIComponent(repoId)}`,
    repoId.split("/").pop(),
  );
}

function cancelHFDownload() {
  if (_hfDownloadES) { _hfDownloadES.close(); _hfDownloadES = null; }
  const panel = document.getElementById("hf-download-panel");
  if (panel) panel.style.display = "none";
  updateStatusBar("ダウンロードをキャンセルしました");
}

// ---------------------------------------------------------------------------
// モデルロード / アンロード
// ---------------------------------------------------------------------------

async function loadHFModel(path) {
  try {
    updateStatusBar(`モデルをロード中...`);
    const data = await apiRequest("/api/hf/load", "POST", { path });
    _hfLoadedModelPath = data.path || path;
    _updateProviderUI("huggingface");
    updateStatusBar(`モデルをロードしました`);
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

async function showManualDownload(repoId, modelName) {
  switchHFTab("manual");
  try {
    const data = await apiRequest("/api/hf/instructions", "POST", {
      repo_id: repoId, model_name: modelName || "",
    });
    _showInstructions(data.instructions);
  } catch (err) {
    showAlert(`手順取得エラー: ${err.message}`, "error");
  }
}

async function getManualInstructions() {
  const sel = document.getElementById("hf-manual-model-selector");
  const modelId = sel ? sel.value : "";
  if (!modelId) { showAlert("モデルを選択してください", "warning"); return; }
  const model = /* look up name */ sel.options[sel.selectedIndex]?.text || modelId;
  try {
    const data = await apiRequest("/api/hf/instructions", "POST", { model_id: modelId });
    _showInstructions(data.instructions);
  } catch (err) {
    showAlert(`手順取得エラー: ${err.message}`, "error");
  }
}

function _showInstructions(inst) {
  const panel   = document.getElementById("hf-instructions-panel");
  const content = document.getElementById("hf-instructions-content");
  if (!panel || !content) return;
  panel.style.display = "";
  panel.dataset.repoId = inst.repo_id || "";

  const destDisplay  = inst.dest_display || inst.dest_dir || "";
  const scriptDir    = destDisplay;
  const runCmd       = `venv\\Scripts\\python "${destDisplay.replace(/\//g, "\\")}\\download.py"`;

  content.innerHTML = `
<b>${escapeHtml(inst.model_name)}</b>

アプリがダウンロードフォルダに <code>download.py</code> スクリプトを生成しました。
以下のコマンドをアプリのルートフォルダ（localforge_web）のコマンドプロンプトで実行してください:

<b>▶ 実行コマンド:</b>
   <code>${escapeHtml(runCmd)}</code>
   <button class="hf-copy-btn" data-text="${escapeHtml(runCmd)}" onclick="_copyText(this.dataset.text)">コピー</button>

<b>スクリプトの場所:</b>
   <code>${escapeHtml(destDisplay)}/download.py</code>

<p style="color:var(--text-muted);font-size:11px;margin-top:8px;">
  このスクリプトは SSL 検証を自動的に無効化するため、企業プロキシ環境でも動作します。
</p>

<b>ダウンロード完了後:</b> 「ローカルモデル」タブで「↻ 再スキャン」→「ロード」をクリックしてください。`;
}

// ---------------------------------------------------------------------------
// ライブ HuggingFace 検索
// ---------------------------------------------------------------------------

async function hfSearch(query) {
  const statusEl  = document.getElementById("hf-browse-status");
  const resultsEl = document.getElementById("hf-browse-results");
  if (!resultsEl) return;

  resultsEl.innerHTML = '<div class="empty-state">検索中...</div>';
  if (statusEl) { statusEl.style.display = ""; statusEl.textContent = "HuggingFace API に接続中..."; }

  try {
    const url = `/api/hf/search?q=${encodeURIComponent(query)}&limit=20`;
    const data = await apiRequest(url);

    if (!data.online) {
      if (statusEl) statusEl.textContent = `オフライン: ${data.error || "HuggingFace API に接続できません"}`;
      resultsEl.innerHTML = '<div class="empty-state">HuggingFace API に接続できません。ネットワークを確認してください。</div>';
      return;
    }

    const models = data.models || [];
    if (statusEl) statusEl.textContent = `${models.length} 件のモデルが見つかりました`;

    if (models.length === 0) {
      resultsEl.innerHTML = '<div class="empty-state">該当するモデルが見つかりません。</div>';
      return;
    }

    resultsEl.innerHTML = models.map(m => {
      const isDownloaded = m.downloaded;
      const isLoaded     = m.local_path && m.local_path === _hfLoadedModelPath;
      const cardClass    = isLoaded ? "loaded" : isDownloaded ? "downloaded" : "";
      return `
        <div class="hf-model-card hf-browse-card ${cardClass}">
          <div class="hf-model-info">
            <div class="hf-model-name">${escapeHtml(m.name)}</div>
            <div class="hf-model-desc" style="font-size:11px;color:var(--text-muted);">${escapeHtml(m.repo_id)}</div>
            <div class="hf-model-meta">
              <span>⬇ ${_fmtNum(m.downloads)}</span>
              <span>♥ ${_fmtNum(m.likes)}</span>
            </div>
          </div>
          <div class="hf-model-actions">
            ${isLoaded
              ? `<span class="hf-status-badge loaded">✓ ロード済み</span>
                 <button class="btn btn-secondary btn-sm" onclick="unloadHFModel()">アンロード</button>`
              : isDownloaded
                ? `<span class="hf-status-badge downloaded">✓ DL済み</span>
                   <button class="btn btn-primary btn-sm" data-path="${escapeHtml(m.local_path)}" onclick="loadHFModel(this.dataset.path)">ロード</button>`
                : `<button class="btn btn-primary btn-sm" data-repo="${escapeHtml(m.repo_id)}" onclick="startHFDownloadRepo(this.dataset.repo)">ダウンロード</button>
                   <button class="btn btn-secondary btn-sm" data-repo="${escapeHtml(m.repo_id)}" data-name="${escapeHtml(m.name)}" onclick="showManualDownload(this.dataset.repo, this.dataset.name)">手動DL</button>`
            }
          </div>
        </div>`;
    }).join("");

  } catch (err) {
    if (statusEl) statusEl.textContent = `エラー: ${err.message}`;
    resultsEl.innerHTML = `<div class="empty-state">エラー: ${escapeHtml(err.message)}</div>`;
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

function _fmtNum(n) {
  if (!n) return "0";
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000)     return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

function _copyText(text) {
  navigator.clipboard.writeText(text).catch(() => {
    const ta = document.createElement("textarea");
    ta.value = text; ta.style.position = "fixed"; ta.style.opacity = "0";
    document.body.appendChild(ta); ta.select(); document.execCommand("copy");
    document.body.removeChild(ta);
  });
  updateStatusBar("コピーしました");
}

// ---------------------------------------------------------------------------
// 初期化（DOMContentLoaded から呼ばれる）
// ---------------------------------------------------------------------------

function initHFUI() {
  document.querySelectorAll(".provider-btn").forEach(btn => {
    btn.addEventListener("click", () => switchProvider(btn.dataset.provider));
  });

  const hfSelectBtn = document.getElementById("hf-model-select-btn");
  if (hfSelectBtn) hfSelectBtn.addEventListener("click", openHFModal);

  const closeBtn = document.getElementById("hf-modal-close");
  if (closeBtn) closeBtn.addEventListener("click", closeHFModal);

  const hfModal = document.getElementById("hf-modal");
  if (hfModal) hfModal.addEventListener("click", e => { if (e.target === hfModal) closeHFModal(); });

  document.querySelectorAll(".hf-tab-btn").forEach(btn => {
    btn.addEventListener("click", () => switchHFTab(btn.dataset.hfTab));
  });

  const cancelBtn = document.getElementById("hf-cancel-download-btn");
  if (cancelBtn) cancelBtn.addEventListener("click", cancelHFDownload);

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
  if (searchBtn) searchBtn.addEventListener("click", () => {
    hfSearch(document.getElementById("hf-search-input")?.value?.trim() || "");
  });
  const topBtn = document.getElementById("hf-search-top-btn");
  if (topBtn) topBtn.addEventListener("click", () => hfSearch(""));
  const searchInput = document.getElementById("hf-search-input");
  if (searchInput) searchInput.addEventListener("keydown", e => {
    if (e.key === "Enter") hfSearch(e.target.value.trim());
  });

  // 手動ダウンロード — カタログセレクタ
  const instrBtn = document.getElementById("hf-get-instructions-btn");
  if (instrBtn) instrBtn.addEventListener("click", getManualInstructions);

  // 手動ダウンロードタブ用カタログセレクタを初期化
  _populateManualSelector();

  // 起動時 HF 状態を確認
  apiRequest("/api/hf/status").then(data => {
    _hfLoadedModelPath = data.loaded_model || "";
    _updateProviderUI(data.active_provider || "ollama");
  }).catch(() => {});
}

function _populateManualSelector() {
  const sel = document.getElementById("hf-manual-model-selector");
  if (!sel || sel.options.length > 1) return;
  apiRequest("/api/hf/models").then(data => {
    (data.catalog || []).forEach(m => {
      const opt = document.createElement("option");
      opt.value = m.id;
      opt.textContent = `${m.name} (${m.size_gb} GB)`;
      sel.appendChild(opt);
    });
  }).catch(() => {});
}
