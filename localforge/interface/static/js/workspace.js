/**
 * workspace.js — マルチプロジェクトワークスペースの管理
 * バニラJSのみ使用。外部ライブラリ不使用。
 */

"use strict";

/**
 * ワークスペースのプロジェクト一覧を読み込み、サイドバーに表示する。
 */
async function loadWorkspace() {
  const bar = document.getElementById("workspace-bar");
  const list = document.getElementById("workspace-projects");
  if (!bar || !list) return;

  try {
    const data = await apiRequest("/api/workspace/list");
    const projects = data.projects || [];

    if (projects.length === 0) {
      bar.style.display = "none";
      return;
    }

    bar.style.display = "block";
    list.innerHTML = "";

    for (const proj of projects) {
      const item = document.createElement("div");
      item.className = "workspace-project-item";
      item.title = proj.root;

      const nameEl = document.createElement("span");
      nameEl.className = "workspace-project-name";
      nameEl.textContent = proj.name;

      const badge = document.createElement("span");
      badge.className = `workspace-badge ${proj.auto ? "badge-auto" : "badge-manual"}`;
      badge.textContent = proj.auto ? "自動" : "手動";

      const meta = document.createElement("span");
      meta.className = "workspace-project-meta";
      meta.textContent = proj.indexed ? `${proj.file_count}ファイル` : "未インデックス";

      item.appendChild(nameEl);
      item.appendChild(badge);
      item.appendChild(meta);

      // 手動追加プロジェクトは削除ボタンを表示
      if (!proj.auto) {
        const removeBtn = document.createElement("button");
        removeBtn.className = "workspace-remove-btn icon-btn";
        removeBtn.textContent = "✕";
        removeBtn.title = "ワークスペースから削除";
        removeBtn.addEventListener("click", async (e) => {
          e.stopPropagation();
          await removeFromWorkspace(proj.root);
        });
        item.appendChild(removeBtn);
      }

      list.appendChild(item);
    }
  } catch (e) {
    console.warn("ワークスペース読み込みエラー:", e.message);
    const bar = document.getElementById("workspace-bar");
    if (bar) bar.style.display = "none";
  }
}

/**
 * プロジェクトをワークスペースに追加する。
 */
async function addToWorkspace() {
  const path = prompt("追加するプロジェクトのフォルダパスを入力してください:\n例: /projects/my-other-app");
  if (!path || !path.trim()) return;

  try {
    await apiRequest("/api/workspace/add", "POST", { path: path.trim() });
    await loadWorkspace();
    showAlert(`ワークスペースに追加しました: ${path.trim()}`, "success");
  } catch (e) {
    showAlert(`ワークスペース追加エラー: ${e.message}`, "error");
  }
}

/**
 * プロジェクトをワークスペースから削除する。
 * @param {string} root - 削除するプロジェクトのルートパス
 */
async function removeFromWorkspace(root) {
  try {
    await apiRequest("/api/workspace/remove", "POST", { path: root });
    await loadWorkspace();
  } catch (e) {
    showAlert(`ワークスペース削除エラー: ${e.message}`, "error");
  }
}

document.addEventListener("DOMContentLoaded", () => {
  // ワークスペース追加ボタン
  const addBtn = document.getElementById("workspace-add-btn");
  if (addBtn) {
    addBtn.addEventListener("click", addToWorkspace);
  }

  // ピン留めモード切替ボタン
  const pinModeBtn = document.getElementById("pin-mode-btn");
  if (pinModeBtn) {
    pinModeBtn.addEventListener("click", () => {
      if (typeof togglePinMode === "function") togglePinMode();
    });
  }

  // ピン留めクリアボタン
  const pinClearBtn = document.getElementById("pin-clear-btn");
  if (pinClearBtn) {
    pinClearBtn.addEventListener("click", () => {
      if (typeof clearPinnedPaths === "function") clearPinnedPaths();
    });
  }
});
