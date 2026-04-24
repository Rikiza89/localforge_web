/**
 * filetree.js — ファイルツリー展開・右クリックメニューの管理
 * バニラJSのみ使用。外部ライブラリ不使用。
 */

"use strict";

// 現在選択中のファイルパス
let _selectedFilePath = null;

// 右クリックメニューの対象ファイルパス
let _contextMenuTarget = null;

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

    const nodeEl = document.createElement("div");
    nodeEl.className = `tree-node ${node.status || "unindexed"}`;
    nodeEl.dataset.path = node.path;
    nodeEl.dataset.isDir = node.is_dir ? "true" : "false";

    // インデントスペーサー
    for (let i = 0; i < depth; i++) {
      const spacer = document.createElement("span");
      spacer.style.width = "12px";
      spacer.style.display = "inline-block";
      nodeEl.appendChild(spacer);
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
      // ファイルクリックで選択
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
 * @param {string} action - アクション名 ("view", "regenerate", "explain-file")
 */
async function handleContextMenuAction(action) {
  const filePath = _contextMenuTarget;
  hideContextMenu();
  if (!filePath) return;

  if (action === "view") {
    await showFileContent(filePath);
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
