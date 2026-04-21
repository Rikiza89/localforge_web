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
}

// =========================================================================
// プロジェクトを開く
// =========================================================================

/**
 * フォルダ選択ダイアログを開いてプロジェクトをロードする。
 */
async function openProject() {
  updateStatusBar("フォルダを選択中...");
  try {
    const data = await apiRequest("/api/project/open", "POST");
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

    // ステータスバーを更新
    await refreshProjectStatus();
    updateStatusBar(`プロジェクトを開きました: ${data.project_root}`);

  } catch (err) {
    if (err.message && err.message.includes("NoFolderSelected")) {
      updateStatusBar("フォルダ選択がキャンセルされました");
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

  await startPostStream(
    "/api/generate/plan",
    { prompt },
    planStream,
    {
      onToken: (token) => { _currentPlanText += token; },
      onDone: () => {
        if (planStream) planStream.style.display = "none";
        renderPlanTree(_currentPlanText);
        updateStatusBar("プラン生成完了");

        // JSONエディタに内容をセット
        const textarea = document.getElementById("plan-json-textarea");
        if (textarea) textarea.value = _currentPlanText;
      },
      onError: (err) => {
        showAlert(`プラン生成エラー: ${err}`, "error");
        updateStatusBar("エラーが発生しました");
      },
    }
  );
}

/**
 * 生成プランをビジュアルツリーとしてレンダリングする。
 * @param {string} planText - プランのJSON文字列
 */
function renderPlanTree(planText) {
  const treeEl = document.getElementById("plan-tree");
  if (!treeEl) return;

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

  files.forEach(f => {
    const item = document.createElement("div");
    item.className = "plan-file-item";
    item.innerHTML = `
      <span class="plan-file-path">${escapeHtml(f.path || "")}</span>
      <span class="plan-file-desc">${escapeHtml(f.description || "")}</span>
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

  const cancelFn = startStream("/api/generate/start", genStream, {
    onProgress: (done, total, currentFile) => {
      if (genProgress) {
        genProgress.max = total;
        genProgress.value = done;
      }
      if (progressLabel) {
        progressLabel.textContent = `${done} / ${total}: ${currentFile}`;
      }
      updateStatusBar(`生成中: ${currentFile} (${done}/${total})`);
    },
    onFileWritten: (path) => {
      refreshFileTree();
    },
    onDone: () => {
      updateStatusBar("ファイル生成完了");
      showAlert("すべてのファイルを生成しました！", "success");
      if (genSection) genSection.style.display = "none";
      refreshFileTree();
      refreshContextPanel();
      refreshGitLog();
    },
    onError: (err) => {
      showAlert(`生成エラー: ${err}`, "error");
      updateStatusBar("エラーが発生しました");
    },
  });
  _cancelGenStream = cancelFn;
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

  startStream("/api/explain/index", null, {
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
    },
    onDone: async () => {
      if (progressContainer) progressContainer.style.display = "none";
      if (reportBtn) reportBtn.disabled = false;
      _indexBuilt = true;
      updateStatusBar("インデックス構築完了");
      _applyRagButtonState(true);
      refreshContextPanel();

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
      if (progressContainer) progressContainer.style.display = "none";
      showAlert(`インデックス構築エラー: ${err}`, "error");
      updateStatusBar("エラーが発生しました");
    },
  });
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

  startStream("/api/explain/migrate-vector", null, {
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
    },
    onDone: () => {
      if (progressContainer) progressContainer.style.display = "none";
      if (migrateBtn) migrateBtn.disabled = false;
      updateStatusBar("RAGベクトルインデックス移行完了");
      showAlert("RAG移行完了 — セマンティック検索が有効になりました。", "success");
    },
    onError: (err) => {
      if (progressContainer) progressContainer.style.display = "none";
      if (migrateBtn) migrateBtn.disabled = false;
      showAlert(`RAG移行エラー: ${err}`, "error");
      updateStatusBar("エラーが発生しました");
    },
  });
}

/**
 * 保存済みレポート（.localforge/report.md）を読み込んでUIに表示する。
 * レポートが存在すればtrue、存在しなければfalseを返す。
 * @returns {Promise<boolean>}
 */
async function loadSavedReport() {
  try {
    const data = await apiRequest("/api/explain/saved-report");
    if (!data.content) return false;

    const reportOutput = document.getElementById("report-output");
    if (!reportOutput) return false;

    reportOutput.innerHTML = "";

    // Markdown を解析して generateReport と同じDOM構造を再現する
    // ## セクション名 → h3 + hr + p
    let currentP = null;
    for (const line of data.content.split("\n")) {
      if (line.startsWith("## ")) {
        const h3 = document.createElement("h3");
        h3.textContent = line.slice(3).trim();
        const hr = document.createElement("hr");
        reportOutput.appendChild(h3);
        reportOutput.appendChild(hr);
        currentP = document.createElement("p");
        reportOutput.appendChild(currentP);
      } else if (line === "---" || line.startsWith("# ")) {
        // セクション区切り・ドキュメントタイトルはスキップ
      } else if (currentP !== null) {
        currentP.textContent += (currentP.textContent ? "\n" : "") + line;
      }
    }

    return true;
  } catch (e) {
    return false;
  }
}

/**
 * 説明レポートを生成する。
 */
function generateReport() {
  if (!_indexBuilt && !_currentProjectRoot) {
    showAlert("先にインデックスを構築してください。", "warning");
    return;
  }

  const reportOutput = document.getElementById("report-output");
  if (reportOutput) reportOutput.innerHTML = "";

  let currentSectionEl = null;
  // 重複セクション防止: 既にレンダリング済みのセクション名を追跡する
  const _renderedSections = new Set();
  updateStatusBar("レポートを生成中...");

  startStream("/api/explain/report", null, {
    onSection: (name) => {
      if (!reportOutput) return;
      // 同じセクションが再度来た場合（SSE再接続によるリスタート）はスキップ
      if (_renderedSections.has(name)) {
        // 既存のセクション要素を currentSectionEl として再利用してトークンを追記する
        const existing = reportOutput.querySelector(`h3[data-section="${CSS.escape(name)}"]`);
        if (existing) {
          currentSectionEl = existing.nextElementSibling
            ? existing.nextElementSibling.nextElementSibling
            : null;
        }
        return;
      }
      _renderedSections.add(name);

      const h3 = document.createElement("h3");
      h3.textContent = name;
      h3.dataset.section = name;
      const hr = document.createElement("hr");
      reportOutput.appendChild(h3);
      reportOutput.appendChild(hr);

      currentSectionEl = document.createElement("p");
      reportOutput.appendChild(currentSectionEl);
    },
    onToken: (token) => {
      if (currentSectionEl) {
        currentSectionEl.textContent += token;
        if (reportOutput) reportOutput.scrollTop = reportOutput.scrollHeight;
      }
    },
    onProgress: (done, total, currentFile) => {
      updateStatusBar(`レポート生成中: ${currentFile} (${done}/${total}セクション)`);
    },
    onDone: () => {
      updateStatusBar("レポート生成完了");
      showAlert("レポートが完成しました！Q&Aで質問できます。", "success");
      enableChat();
    },
    onError: (err) => {
      showAlert(`レポート生成エラー: ${err}`, "error");
      updateStatusBar("エラーが発生しました");
    },
  });
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

  startStream("/api/generate/start", genStream, {
    onProgress: (done, total, currentFile) => {
      if (genProgress) {
        genProgress.max = total;
        genProgress.value = done;
      }
      if (progressLabel) {
        progressLabel.textContent = `${done} / ${total}: ${currentFile}`;
      }
    },
    onFileWritten: () => refreshFileTree(),
    onDone: () => {
      updateStatusBar("生成再開完了");
      showAlert("すべてのファイルを生成しました！", "success");
      if (genSection) genSection.style.display = "none";
      refreshFileTree();
    },
    onError: (err) => {
      showAlert(`再開エラー: ${err}`, "error");
      updateStatusBar("エラーが発生しました");
    },
  });
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

  // Explainタブのボタン
  const buildIndexBtn = document.getElementById("build-index-btn");
  if (buildIndexBtn) buildIndexBtn.addEventListener("click", buildIndex);

  const generateReportBtn = document.getElementById("generate-report-btn");
  if (generateReportBtn) generateReportBtn.addEventListener("click", generateReport);

  const migrateVectorBtn = document.getElementById("migrate-vector-btn");
  if (migrateVectorBtn) migrateVectorBtn.addEventListener("click", migrateVectorIndex);

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

  // Ollamaライブ出力パネル初期化
  OllamaPanel.init();

  // 初期化: モデル一覧とCPU情報を読み込む
  await loadModels();
  await loadCpuThreadInfo();

  // 起動時Ollamaヘルスチェック
  await checkOllamaHealth();

  updateStatusBar("LocalForge 準備完了 — フォルダを開いてください");
});
