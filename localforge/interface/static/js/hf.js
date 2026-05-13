/**
 * hf.js — HuggingFace モデル管理 UI (safetensors 形式)
 * プロバイダー切替、モデルカタログ、手順表示、ロード/アンロードを制御する。
 * ダウンロード機能は削除済み — ブラウザ手動ダウンロードを案内する。
 */

"use strict";

// ---------------------------------------------------------------------------
// 状態
// ---------------------------------------------------------------------------

let _activeProvider    = "ollama";
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

    const panelId = `hf-inst-cat-${escapeHtml(m.id)}`;

    const actionBtns = isLoaded
      ? `<button class="btn btn-secondary btn-sm" onclick="unloadHFModel()">アンロード</button>`
      : isDownloaded
        ? `<button class="btn btn-primary btn-sm" data-path="${escapeHtml(m.local_path)}" onclick="loadHFModel(this.dataset.path)">▶ ロード</button>`
        : "";

    return `
      <div class="hf-model-card ${cardClass}" id="hf-card-${escapeHtml(m.id)}">
        <div class="hf-card-main">
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
            <button class="btn btn-secondary btn-sm hf-inst-toggle"
              data-panel="${panelId}"
              data-model-id="${escapeHtml(m.id)}"
              onclick="toggleInstructions(this)">📋 手順</button>
          </div>
        </div>
        <div class="hf-inst-panel" id="${panelId}" style="display:none;"></div>
      </div>`;
  }).join("");
}

function renderLocalModels(local) {
  const el = document.getElementById("hf-local-list");
  if (!el) return;

  if (local.length === 0) {
    el.innerHTML = '<div class="empty-state">ローカルモデルが見つかりません。カタログの「📋 手順」からダウンロードしてください。</div>';
    return;
  }

  el.innerHTML = local.map(m => {
    const isLoaded = m.path === _hfLoadedModelPath;
    const cardClass = isLoaded ? "loaded" : "";
    return `
      <div class="hf-model-card ${cardClass}">
        <div class="hf-card-main">
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
              : `<button class="btn btn-primary btn-sm" data-path="${escapeHtml(m.path)}" onclick="loadHFModel(this.dataset.path)">▶ ロード</button>`
            }
          </div>
        </div>
      </div>`;
  }).join("");
}

// ---------------------------------------------------------------------------
// インライン手順パネル
// ---------------------------------------------------------------------------

async function toggleInstructions(btnEl) {
  const panelId = btnEl.dataset.panel;
  const panel   = document.getElementById(panelId);
  if (!panel) return;

  // 閉じる
  if (panel.style.display !== "none") {
    panel.style.display = "none";
    btnEl.textContent = "📋 手順";
    return;
  }

  // 開く
  panel.style.display = "";
  btnEl.textContent = "▲ 閉じる";

  // 既に読み込み済みならそのまま表示
  if (panel.dataset.loaded === "1") return;

  panel.innerHTML = '<div class="hf-inst-loading">手順を読み込み中...</div>';

  const payload = {};
  if (btnEl.dataset.modelId) payload.model_id   = btnEl.dataset.modelId;
  if (btnEl.dataset.repoId)  payload.repo_id    = btnEl.dataset.repoId;
  if (btnEl.dataset.name)    payload.model_name = btnEl.dataset.name;

  try {
    const data = await apiRequest("/api/hf/instructions", "POST", payload);
    panel.innerHTML = _buildInstructionsHTML(data.instructions);
    panel.dataset.loaded = "1";
  } catch (err) {
    panel.innerHTML = `<div class="hf-inst-error">エラー: ${escapeHtml(err.message)}</div>`;
  }
}

function _buildInstructionsHTML(inst) {
  const url         = inst.hf_url || `https://huggingface.co/${inst.repo_id}/tree/main`;
  const destDisplay = inst.dest_display || inst.dest_dir || "";
  const fileList    = inst.file_list || [];

  let filesHtml;
  if (fileList.length > 0) {
    filesHtml = fileList.map(f =>
      `<li><code class="hf-inst-file">${escapeHtml(f)}</code></li>`
    ).join("");
  } else {
    filesHtml = [
      `<li><code class="hf-inst-file">config.json</code></li>`,
      `<li><code class="hf-inst-file">tokenizer.json</code></li>`,
      `<li><code class="hf-inst-file">tokenizer_config.json</code></li>`,
      `<li><code class="hf-inst-file">*.safetensors</code> <span class="hf-inst-note-inline">（全ウェイトファイル）</span></li>`,
      `<li><code class="hf-inst-file">*.safetensors.index.json</code> <span class="hf-inst-note-inline">（存在する場合）</span></li>`,
    ].join("");
  }

  const fetchNote = fileList.length === 0
    ? `<div class="hf-inst-note">💡 ファイルリストを自動取得できませんでした。HuggingFace ページで <code>.safetensors</code> と設定ファイル（.json）をダウンロードしてください。</div>`
    : `<div class="hf-inst-note">✓ HuggingFace API からファイルリストを取得しました（${fileList.length} ファイル）</div>`;

  return `
<div class="hf-inst-content">
  <div class="hf-inst-step">
    <span class="hf-inst-num">1</span>
    <div class="hf-inst-body">
      <div class="hf-inst-label">ブラウザで HuggingFace ページを開く</div>
      <div class="hf-inst-path-box">
        <span class="hf-inst-path-text">${escapeHtml(url)}</span>
        <button class="hf-copy-btn" data-text="${escapeHtml(url)}" onclick="_copyText(this.dataset.text)">コピー</button>
      </div>
    </div>
  </div>

  <div class="hf-inst-step">
    <span class="hf-inst-num">2</span>
    <div class="hf-inst-body">
      <div class="hf-inst-label">以下のファイルをひとつずつダウンロード</div>
      ${fetchNote}
      <ul class="hf-inst-file-list">${filesHtml}</ul>
      <div class="hf-inst-note">⚠ <code>.h5</code> <code>.gguf</code> <code>flax_*</code> <code>tf_*</code> <code>onnx/</code> は不要です</div>
    </div>
  </div>

  <div class="hf-inst-step">
    <span class="hf-inst-num">3</span>
    <div class="hf-inst-body">
      <div class="hf-inst-label">ダウンロードしたファイルをこのフォルダに移動</div>
      <div class="hf-inst-path-box">
        <span class="hf-inst-path-text">${escapeHtml(destDisplay)}</span>
        <button class="hf-copy-btn" data-text="${escapeHtml(destDisplay)}" onclick="_copyText(this.dataset.text)">コピー</button>
      </div>
      <div class="hf-inst-note">フォルダは自動作成済みです。ファイルはフォルダ直下に置いてください（サブフォルダ不要）</div>
    </div>
  </div>

  <div class="hf-inst-step">
    <span class="hf-inst-num">4</span>
    <div class="hf-inst-body">
      <div class="hf-inst-label">「ローカルモデル」タブ → 「↻ 再スキャン」→「▶ ロード」</div>
    </div>
  </div>
</div>`;
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
      const safeRepoId   = m.repo_id.replace(/[^a-zA-Z0-9_-]/g, "_");
      const panelId      = `hf-inst-browse-${safeRepoId}`;

      return `
        <div class="hf-model-card hf-browse-card ${cardClass}">
          <div class="hf-card-main">
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
                     <button class="btn btn-primary btn-sm" data-path="${escapeHtml(m.local_path)}" onclick="loadHFModel(this.dataset.path)">▶ ロード</button>`
                  : ""
              }
              <button class="btn btn-secondary btn-sm hf-inst-toggle"
                data-panel="${panelId}"
                data-repo-id="${escapeHtml(m.repo_id)}"
                data-name="${escapeHtml(m.name)}"
                onclick="toggleInstructions(this)">📋 手順</button>
            </div>
          </div>
          <div class="hf-inst-panel" id="${panelId}" style="display:none;"></div>
        </div>`;
    }).join("");

  } catch (err) {
    if (statusEl) statusEl.textContent = `エラー: ${err.message}`;
    resultsEl.innerHTML = `<div class="empty-state">エラー: ${escapeHtml(err.message)}</div>`;
  }
}

// ---------------------------------------------------------------------------
// カスタムタブ（任意の repo_id で手順を表示）
// ---------------------------------------------------------------------------

async function showCustomInstructions() {
  const input = document.getElementById("hf-custom-repo-input");
  const panel = document.getElementById("hf-custom-instructions");
  if (!input || !panel) return;

  const repoId = input.value.trim();
  if (!repoId) { showAlert("repo ID を入力してください", "warning"); return; }

  panel.style.display = "";
  panel.innerHTML = '<div class="hf-inst-loading">手順を読み込み中...</div>';

  try {
    const data = await apiRequest("/api/hf/instructions", "POST", { repo_id: repoId });
    panel.innerHTML = _buildInstructionsHTML(data.instructions);
  } catch (err) {
    panel.innerHTML = `<div class="hf-inst-error">エラー: ${escapeHtml(err.message)}</div>`;
  }
}

// ---------------------------------------------------------------------------
// ユーティリティ
// ---------------------------------------------------------------------------

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

  // カスタムタブ
  const customBtn = document.getElementById("hf-custom-show-btn");
  if (customBtn) customBtn.addEventListener("click", showCustomInstructions);
  const customInput = document.getElementById("hf-custom-repo-input");
  if (customInput) customInput.addEventListener("keydown", e => {
    if (e.key === "Enter") showCustomInstructions();
  });

  // 起動時 HF 状態を確認
  apiRequest("/api/hf/status").then(data => {
    _hfLoadedModelPath = data.loaded_model || "";
    _updateProviderUI(data.active_provider || "ollama");
  }).catch(() => {});
}
