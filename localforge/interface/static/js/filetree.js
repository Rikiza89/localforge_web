/**
 * filetree.js — ファイルツリー展開・右クリックメニュー・ピン留めモードの管理
 * バニラJSのみ使用。外部ライブラリ不使用。
 */

"use strict";

// 現在選択中のファイルパス
let _selectedFilePath = null;

// 右クリックメニューの対象ファイルパス
let _contextMenuTarget = null;

// ── ピン留め状態 ──
let _pinModeActive = false;
const _pinnedPaths = new Set();    // プロジェクト相対パス（ファイルまたはフォルダ）

/**
 * ピン留めモードの状態を返す。
 */
function isPinModeActive() { return _pinModeActive; }

/**
 * ピン留めされたパスのリストを返す。
 */
function getPinnedPaths() { return [..._pinnedPaths]; }

/**
 * ピン留めモードを切り替える。
 */
function togglePinMode() {
  _pinModeActive = !_pinModeActive;
  const btn = document.getElementById("pin-mode-btn");
  if (btn) btn.classList.toggle("active", _pinModeActive);
  refreshFileTree();
  _updatePinStatusBar();
}

/**
 * ピン留めをすべてクリアする。
 */
function clearPinnedPaths() {
  _pinnedPaths.clear();
  _savePinnedToServer();
  refreshFileTree();
  _updatePinStatusBar();
}

/**
 * ピン留めステータスバーを更新する。
 */
function _updatePinStatusBar() {
  const bar = document.getElementById("pin-status-bar");
  const txt = document.getElementById("pin-status-text");
  if (!bar || !txt) return;
  if (_pinnedPaths.size > 0) {
    bar.style.display = "flex";
    txt.textContent = `📎 ${_pinnedPaths.size}件ピン留め`;
  } else {
    bar.style.display = "none";
  }
}

/**
 * ピン留め状態をサーバーに保存する。
 */
async function _savePinnedToServer() {
  try {
    await apiRequest("/api/project/pinned", "POST", { paths: [..._pinnedPaths] });
  } catch (e) {
    console.warn("ピン留め保存エラー:", e.message);
  }
}

/**
 * サーバーからピン留め状態を読み込む。
 */
async function loadPinnedFromServer() {
  try {
    const data = await apiRequest("/api/project/pinned");
    _pinnedPaths.clear();
    (data.pinned || []).forEach(p => _pinnedPaths.add(p));
    _updatePinStatusBar();
    if (_pinModeActive) refreshFileTree();
  } catch (e) {
    console.warn("ピン留め読み込みエラー:", e.message);
  }
}

/**
 * パスがピン留めされているかチェックする。
 */
function _isPinned(path) { return _pinnedPaths.has(path); }

/**
 * パスのピン留め状態をトグルし、フォルダなら子ファイルも一括操作する。
 * @param {string} path - トグルするパス
 * @param {boolean} isDir - ディレクトリかどうか
 * @param {Array} childPaths - ディレクトリの場合の子パスリスト
 */
function _togglePin(path, isDir, childPaths) {
  if (isDir) {
    const allPinned = childPaths.every(cp => _pinnedPaths.has(cp));
    if (allPinned) {
      childPaths.forEach(cp => _pinnedPaths.delete(cp));
      _pinnedPaths.delete(path);
    } else {
      _pinnedPaths.add(path);
      childPaths.forEach(cp => _pinnedPaths.add(cp));
    }
  } else {
    if (_pinnedPaths.has(path)) {
      _pinnedPaths.delete(path);
    } else {
      _pinnedPaths.add(path);
    }
  }
  _savePinnedToServer();
  _updatePinStatusBar();
  // チェックボックス状態を再描画せずに更新（軽量）
  document.querySelectorAll(`.tree-pin-check[data-path="${CSS.escape(path)}"]`).forEach(cb => {
    cb.checked = _pinnedPaths.has(path);
  });
  if (isDir) {
    childPaths.forEach(cp => {
      document.querySelectorAll(`.tree-pin-check[data-path="${CSS.escape(cp)}"]`).forEach(cb => {
        cb.checked = _pinnedPaths.has(cp);
      });
    });
  }
}

/**
 * ファイルツリーをレンダリングする。
 * @param {Array} nodes - FileNodeの配列
 * @param {HTMLElement} container - レンダリング先コンテナ要素
 */
function renderFileTree(nodes, container) {
  container.innerHTML = "";
  if (!nodes || nodes.length === 0) {
    container.innerHTML = '<div class="empty-state">ファイルがありません</div>';
    return;
  }
  const ul = buildTreeList(nodes, 0);
  container.appendChild(ul);
}

/**
 * ノード以下のすべてのファイルパスを再帰的に収集する。
 * @param {Object} node - FileNode
 * @returns {Array<string>} ファイルパスのリスト
 */
function _collectFilePaths(node) {
  if (!node.is_dir) return [node.path];
  const paths = [];
  if (node.children) {
    for (const child of node.children) {
      paths.push(..._collectFilePaths(child));
    }
  }
  return paths;
}

/**
 * 再帰的にツリーのUL要素を構築する。
 * @param {Array} nodes - FileNodeの配列
 * @param {number} depth - ネストの深さ
 * @returns {HTMLElement} ul要素
 */
function buildTreeList(nodes, depth) {
  const ul = document.createElement("ul");
  ul.style.paddingLeft = depth === 0 ? "0" : "16px";
  ul.style.listStyle = "none";

  for (const node of nodes) {
    const li = document.createElement("li");

    const isPinned = _isPinned(node.path);
    const nodeEl = document.createElement("div");
    nodeEl.className = `tree-node ${node.status || "unindexed"}${isPinned ? " pinned" : ""}`;
    nodeEl.dataset.path = node.path;
    nodeEl.dataset.isDir = node.is_dir ? "true" : "false";

    // インデントスペーサー
    for (let i = 0; i < depth; i++) {
      const spacer = document.createElement("span");
      spacer.style.width = "12px";
      spacer.style.display = "inline-block";
      nodeEl.appendChild(spacer);
    }

    // ピン留めモード: チェックボックスを追加
    if (_pinModeActive) {
      const childFilePaths = _collectFilePaths(node);
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.className = "tree-pin-check";
      cb.dataset.path = node.path;
      // フォルダは子ファイルがすべてピン留めされている場合にチェック
      cb.checked = node.is_dir
        ? (childFilePaths.length > 0 && childFilePaths.every(p => _pinnedPaths.has(p)))
        : isPinned;
      cb.addEventListener("change", (e) => {
        e.stopPropagation();
        _togglePin(node.path, node.is_dir, childFilePaths);
        // ノードのpinnedクラスを更新
        nodeEl.classList.toggle("pinned", _pinnedPaths.has(node.path));
      });
      cb.addEventListener("click", (e) => e.stopPropagation());
      nodeEl.appendChild(cb);
    }

    // アイコン
    const icon = document.createElement("span");
    icon.className = "tree-node-icon";
    if (node.is_dir) {
      icon.textContent = "▶";
      icon.style.fontSize = "9px";
    } else {
      icon.textContent = _getFileIcon(node.name);
    }
    nodeEl.appendChild(icon);

    // ラベル
    const label = document.createElement("span");
    label.className = "tree-node-label";
    label.textContent = node.name;
    nodeEl.appendChild(label);

    li.appendChild(nodeEl);

    if (node.is_dir && node.children && node.children.length > 0) {
      const childrenContainer = document.createElement("div");
      childrenContainer.className = "tree-children";
      childrenContainer.style.display = "none";
      const childUl = buildTreeList(node.children, 0);
      childrenContainer.appendChild(childUl);
      li.appendChild(childrenContainer);

      // ディレクトリクリックで展開・折り畳み
      nodeEl.addEventListener("click", () => {
        const isExpanded = childrenContainer.style.display !== "none";
        childrenContainer.style.display = isExpanded ? "none" : "block";
        icon.textContent = isExpanded ? "▶" : "▼";
      });
    } else if (!node.is_dir) {
      // ファイルクリックで選択（ピン留めモード中も選択は維持）
      nodeEl.addEventListener("click", () => {
        document.querySelectorAll(".tree-node.selected").forEach(el =>
          el.classList.remove("selected")
        );
        nodeEl.classList.add("selected");
        _selectedFilePath = node.path;
      });
    }

    // 右クリックメニュー
    nodeEl.addEventListener("contextmenu", (e) => {
      e.preventDefault();
      if (!node.is_dir) {
        _showContextMenu(e.clientX, e.clientY, node.path);
      }
    });

    ul.appendChild(li);
  }

  return ul;
}

/**
 * ファイルアイコンを返す。
 * @param {string} name - ファイル名
 * @returns {string} アイコン文字
 */
function _getFileIcon(name) {
  const ext = name.split(".").pop().toLowerCase();
  const icons = {
    "py": "🐍", "js": "📄", "ts": "📄", "html": "🌐", "css": "🎨",
    "json": "📋", "md": "📝", "yaml": "⚙", "yml": "⚙",
    "go": "📄", "rs": "🦀", "java": "☕", "sh": "⚡",
    "toml": "⚙", "txt": "📝", "sql": "🗄",
  };
  return icons[ext] || "📄";
}

/**
 * 右クリックコンテキストメニューを表示する。
 * @param {number} x - マウスのX座標
 * @param {number} y - マウスのY座標
 * @param {string} filePath - 対象ファイルパス
 */
function _showContextMenu(x, y, filePath) {
  _contextMenuTarget = filePath;
  const menu = document.getElementById("context-menu");
  if (!menu) return;

  menu.style.left = `${x}px`;
  menu.style.top = `${y}px`;
  menu.style.display = "block";

  // ビューポート外にはみ出す場合の補正
  const rect = menu.getBoundingClientRect();
  if (rect.right > window.innerWidth) {
    menu.style.left = `${x - rect.width}px`;
  }
  if (rect.bottom > window.innerHeight) {
    menu.style.top = `${y - rect.height}px`;
  }
}

/**
 * 右クリックコンテキストメニューを非表示にする。
 */
function hideContextMenu() {
  const menu = document.getElementById("context-menu");
  if (menu) menu.style.display = "none";
  _contextMenuTarget = null;
}

/**
 * コンテキストメニューのアクションを処理する。
 * @param {string} action - アクション名 ("view", "edit-file", "regenerate", "explain-file")
 */
async function handleContextMenuAction(action) {
  const filePath = _contextMenuTarget;
  hideContextMenu();
  if (!filePath) return;

  if (action === "view") {
    await showFileContent(filePath);
  } else if (action === "edit-file") {
    await openFileEditor(filePath);
  } else if (action === "regenerate") {
    await triggerRegenerateFile(filePath);
  } else if (action === "explain-file") {
    await explainSingleFile(filePath);
  }
}

/**
 * ファイル内容をモーダルで表示する。
 * @param {string} filePath - 表示するファイルの相対パス
 */
async function showFileContent(filePath) {
  try {
    const data = await apiRequest(`/api/project/file-content?path=${encodeURIComponent(filePath)}`);
    const modal = document.getElementById("file-modal");
    const title = document.getElementById("modal-title");
    const content = document.getElementById("modal-content");

    if (!modal || !title || !content) return;

    title.textContent = filePath;
    content.className = "modal-content";
    content.innerHTML = '<pre class="code-block">' + escapeHtml(data.content) + "</pre>";
    modal.style.display = "flex";
  } catch (err) {
    showAlert(`ファイルの読み込みに失敗しました: ${err.message}`, "error");
  }
}

/**
 * ファイルを編集モードのモーダルで開く。
 * @param {string} filePath - 編集するファイルの相対パス
 */
async function openFileEditor(filePath) {
  try {
    const data = await apiRequest(`/api/project/file-content?path=${encodeURIComponent(filePath)}`);
    const modal = document.getElementById("file-modal");
    const title = document.getElementById("modal-title");
    const content = document.getElementById("modal-content");

    if (!modal || !title || !content) return;

    title.textContent = `✏ 編集: ${filePath}`;

    const toolbar = document.createElement("div");
    toolbar.className = "modal-edit-toolbar";

    const saveBtn = document.createElement("button");
    saveBtn.className = "btn btn-primary btn-sm";
    saveBtn.textContent = "保存";

    const cancelBtn = document.createElement("button");
    cancelBtn.className = "btn btn-secondary btn-sm";
    cancelBtn.textContent = "キャンセル";

    const statusSpan = document.createElement("span");
    statusSpan.style.marginLeft = "auto";
    statusSpan.style.color = "var(--text-muted)";
    statusSpan.style.fontSize = "11px";

    toolbar.appendChild(saveBtn);
    toolbar.appendChild(cancelBtn);
    toolbar.appendChild(statusSpan);

    const textarea = document.createElement("textarea");
    textarea.className = "modal-edit-textarea";
    textarea.value = data.content;
    textarea.spellcheck = false;

    content.className = "modal-content";
    content.innerHTML = "";
    content.appendChild(toolbar);
    content.appendChild(textarea);

    modal.style.display = "flex";
    textarea.focus();

    saveBtn.addEventListener("click", async () => {
      saveBtn.disabled = true;
      statusSpan.textContent = "保存中...";
      try {
        await apiRequest("/api/project/save-file", "POST", { path: filePath, content: textarea.value });
        statusSpan.textContent = "保存完了 ✓";
        statusSpan.style.color = "var(--success)";
        showAlert(`保存しました: ${filePath}`, "success");
        refreshFileTree();
      } catch (err) {
        statusSpan.textContent = "保存失敗";
        statusSpan.style.color = "var(--danger)";
        showAlert(`保存エラー: ${err.message}`, "error");
      } finally {
        saveBtn.disabled = false;
      }
    });

    cancelBtn.addEventListener("click", () => {
      modal.style.display = "none";
    });
  } catch (err) {
    showAlert(`ファイルの読み込みに失敗しました: ${err.message}`, "error");
  }
}

/**
 * 単一ファイルをAIで説明するストリームを開始する。
 * @param {string} filePath - 説明するファイルのパス
 */
async function explainSingleFile(filePath) {
  try {
    const data = await apiRequest(`/api/project/file-content?path=${encodeURIComponent(filePath)}`);
    const question = `このファイル (${filePath}) の役割・構造・重要なポイントを詳しく説明してください:\n\n${data.content.slice(0, 1000)}`;

    const modal = document.getElementById("file-modal");
    const title = document.getElementById("modal-title");
    const content = document.getElementById("modal-content");

    if (!modal || !title || !content) return;

    title.textContent = `説明: ${filePath}`;
    content.className = "modal-content md-body";
    content.innerHTML = "説明を生成中...";
    modal.style.display = "flex";

    let explainBuf = "";
    await startPostStream(
      "/api/explain/ask",
      { question, history: [] },
      null,
      {
        onToken: (token) => {
          if (explainBuf === "") content.innerHTML = "";
          explainBuf += token;
          content.innerHTML = _renderMd(explainBuf);
        },
        onDone: () => { updateStatusBar("説明完了"); },
        onError: (err) => { content.innerHTML += `<p style="color:var(--danger)">[エラー: ${escapeHtml(String(err))}]</p>`; },
      }
    );
  } catch (err) {
    showAlert(`ファイル説明に失敗しました: ${err.message}`, "error");
  }
}

/**
 * ファイルの再生成をトリガーする。
 * @param {string} filePath - 再生成するファイルパス
 */
async function triggerRegenerateFile(filePath) {
  if (!confirm(`${filePath} を再生成しますか？`)) return;

  updateStatusBar(`再生成中: ${filePath}`);

  await startPostStream(
    "/api/generate/regenerate",
    { file_path: filePath },
    null,
    {
      onFileWritten: (path) => {
        showAlert(`ファイルを再生成しました: ${path}`, "success");
        refreshFileTree();
      },
      onDone: () => { updateStatusBar("再生成完了"); },
      onError: (err) => {
        showAlert(`再生成エラー: ${err}`, "error");
        updateStatusBar("エラーが発生しました");
      },
    }
  );
}

/**
 * ファイルツリーを更新する。
 */
async function refreshFileTree() {
  try {
    const data = await apiRequest("/api/project/tree");
    const container = document.getElementById("file-tree");
    if (container && data.file_tree) {
      renderFileTree(data.file_tree, container);
    }
  } catch (err) {
    console.warn("ファイルツリー更新エラー:", err.message);
  }
}

// ---------------------------------------------------------------------------
// イベントリスナーの初期化
// ---------------------------------------------------------------------------

document.addEventListener("DOMContentLoaded", () => {
  // モーダルを閉じる
  const modalOverlay = document.getElementById("file-modal");
  const modalClose = document.getElementById("modal-close");

  if (modalClose) {
    modalClose.addEventListener("click", () => {
      if (modalOverlay) modalOverlay.style.display = "none";
    });
  }

  if (modalOverlay) {
    modalOverlay.addEventListener("click", (e) => {
      if (e.target === modalOverlay) modalOverlay.style.display = "none";
    });
  }

  // コンテキストメニューのアクション
  const contextMenu = document.getElementById("context-menu");
  if (contextMenu) {
    contextMenu.querySelectorAll("li[data-action]").forEach(li => {
      li.addEventListener("click", () => {
        handleContextMenuAction(li.dataset.action);
      });
    });
  }

  // クリックでコンテキストメニューを非表示
  document.addEventListener("click", () => hideContextMenu());

  // ツリー更新ボタン
  const refreshBtn = document.getElementById("refresh-tree-btn");
  if (refreshBtn) {
    refreshBtn.addEventListener("click", refreshFileTree);
  }
});
