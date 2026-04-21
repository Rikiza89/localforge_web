/**
 * chat.js — Q&Aチャットインターフェースの管理
 * 会話履歴をメモリ内に保持し、最大10件をAPIに送信する。
 */

"use strict";

// 会話履歴（全件保持、API送信時は最新10件）
let _chatHistory = [];

// 送信中フラグ
let _chatSending = false;

/**
 * チャット履歴をリセットする。
 */
function resetChatHistory() {
  _chatHistory = [];
  const historyEl = document.getElementById("chat-history");
  if (historyEl) historyEl.innerHTML = "";
}

/**
 * チャット機能を有効化する。
 */
function enableChat() {
  const input = document.getElementById("chat-input");
  const sendBtn = document.getElementById("chat-send");
  if (input) input.disabled = false;
  if (sendBtn) sendBtn.disabled = false;
}

/**
 * チャット機能を無効化する。
 */
function disableChat() {
  const input = document.getElementById("chat-input");
  const sendBtn = document.getElementById("chat-send");
  if (input) input.disabled = true;
  if (sendBtn) sendBtn.disabled = true;
}

/**
 * ユーザーの質問を送信してストリーミング回答を表示する。
 * @param {string} question - ユーザーの質問テキスト
 */
async function sendChatMessage(question) {
  if (!question.trim() || _chatSending) return;

  _chatSending = true;
  disableChat();
  _lockUI(null);

  // 会話ターンのDOM要素を作成
  const historyEl = document.getElementById("chat-history");
  if (!historyEl) return;

  const turn = document.createElement("div");
  turn.className = "chat-turn";

  const qEl = document.createElement("div");
  qEl.className = "chat-q";
  qEl.textContent = question;
  turn.appendChild(qEl);

  const aEl = document.createElement("div");
  aEl.className = "chat-a";
  aEl.textContent = "回答生成中...";
  turn.appendChild(aEl);

  historyEl.appendChild(turn);
  historyEl.scrollTop = historyEl.scrollHeight;

  // 会話履歴に質問を追加
  _chatHistory.push({ role: "user", content: question });

  // 応答テキストを蓄積するバッファ
  let answerBuffer = "";
  let isFirstToken = true;

  await startPostStream(
    "/api/explain/ask",
    {
      question,
      history: _chatHistory.slice(-10).map(m => ({ role: m.role, content: m.content })),
    },
    null,
    {
      onToken: (token) => {
        if (isFirstToken) {
          aEl.textContent = "";
          isFirstToken = false;
        }
        answerBuffer += token;
        aEl.textContent = answerBuffer;
        historyEl.scrollTop = historyEl.scrollHeight;
      },
      onDone: () => {
        // 会話履歴にアシスタントの回答を追加
        _chatHistory.push({ role: "assistant", content: answerBuffer });
        _chatSending = false;
        enableChat();
        _unlockUI();

        // Q&Aをディスクに保存（エラーは無視してUIをブロックしない）
        apiRequest("/api/explain/qa-save", "POST", { question, answer: answerBuffer })
          .catch(e => console.warn("Q&A保存エラー:", e.message));

        // 入力フィールドにフォーカスを戻す
        const input = document.getElementById("chat-input");
        if (input) input.focus();
      },
      onError: (err) => {
        aEl.textContent = `[エラー: ${err}]`;
        aEl.style.color = "var(--danger)";
        _chatSending = false;
        enableChat();
        _unlockUI();
      },
    }
  );
}

// ---------------------------------------------------------------------------
// イベントリスナーの初期化
// ---------------------------------------------------------------------------

document.addEventListener("DOMContentLoaded", () => {
  const sendBtn = document.getElementById("chat-send");
  const chatInput = document.getElementById("chat-input");

  if (sendBtn) {
    sendBtn.addEventListener("click", () => {
      const input = document.getElementById("chat-input");
      if (!input) return;
      const question = input.value.trim();
      if (!question) return;
      input.value = "";
      sendChatMessage(question);
    });
  }

  if (chatInput) {
    chatInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        const question = chatInput.value.trim();
        if (!question) return;
        chatInput.value = "";
        sendChatMessage(question);
      }
    });
  }
});
