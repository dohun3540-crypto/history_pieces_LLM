"use strict";

const state = {
  app: "initializing",
  request: "idle",
  session: null,
  freeOpen: false,
  lastRequest: "",
  lastFocus: null,
};

// The supplied images are complete vertical design studies, not verified game scenes.
// They are therefore used only as obscured atmosphere backgrounds. Mapping is visual:
// three pieces -> 02/07/01, skipped -> 03, paused -> 04, complete -> 05, loading -> 06.
const BACKGROUND_BY_PIECE = {
  "demo-piece-1": "/assets/backgrounds/background_02.png",
  "demo-piece-2": "/assets/backgrounds/background_07.png",
  "demo-piece-3": "/assets/backgrounds/background_01.png",
};
const BACKGROUND_BY_STATE = {
  skipped: "/assets/backgrounds/background_03.png",
  paused: "/assets/backgrounds/background_04.png",
  journey_complete: "/assets/backgrounds/background_05.png",
  default: "/assets/backgrounds/background_06.png",
};
const ACTION_PRESENTATION = {
  PAUSE_JOURNEY: { label: "잠시 쉬기", description: "지금 여정을 잠시 멈출 수 있어요." },
  CONTINUE_WITH_SHORT_MODE: { label: "짧게 계속하기", description: "이번 세션에서는 답변을 짧게 이어갑니다." },
  SKIP_REFLECTION: { label: "감상 건너뛰기", description: "감상을 남기지 않고 다음 단계로 갈 수 있어요." },
  GO_NEXT_PIECE: { label: "다음 조각으로", description: "명시적으로 선택할 때만 데모 진행 상태가 바뀝니다." },
  OPEN_FREE_CHAT: { label: "자유대화 열기", description: "원래 질문과 현재 여정 문맥을 보존해 이어서 물어봅니다." },
  OFFER_MORE_HISTORY_IN_FREE_CHAT: { label: "자세히 물어보기", description: "출처가 필요한 설명은 자유대화에서 이어갈 수 있어요." },
};

const byId = (id) => document.getElementById(id);
const statusNode = byId("app-status");

function announce(message) {
  statusNode.textContent = "";
  window.requestAnimationFrame(() => { statusNode.textContent = message; });
}

function setRequestState(value, message = "") {
  state.request = value;
  const busy = value === "loading";
  document.querySelectorAll("button, input").forEach((node) => { node.disabled = busy; });
  byId("free-chat-panel").setAttribute("aria-busy", String(busy));
  const freeState = byId("free-state");
  freeState.dataset.state = value;
  const labels = {
    idle: "질문을 기다리고 있어요",
    loading: "기록과 출처를 확인하고 있어요",
    success: "답변을 확인했어요",
    insufficient_evidence: "확인 가능한 근거가 부족해요",
    error: "요청을 처리하지 못했어요",
  };
  byId("free-state-text").textContent = message || labels[value] || labels.idle;
  announce(message || labels[value] || "상태가 변경됐습니다.");
}

async function api(path, options = {}) {
  const response = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.message || "요청을 처리하지 못했습니다.");
  return payload;
}

function pieceLabel(pieceId) {
  const pieces = state.session && Array.isArray(state.session.demo_pieces) ? state.session.demo_pieces : [];
  const found = pieces.find((piece) => piece.piece_id === pieceId);
  return found ? found.label : pieceId;
}

function resolveBackground(session) {
  if (session.current_journey_step === "journey_complete") return BACKGROUND_BY_STATE.journey_complete;
  if (session.piece_ui_state === "paused") return BACKGROUND_BY_STATE.paused;
  if (session.piece_ui_state === "skipped") return BACKGROUND_BY_STATE.skipped;
  return BACKGROUND_BY_PIECE[session.current_piece_id] || BACKGROUND_BY_STATE.default;
}

function renderSession(session) {
  state.session = session;
  const currentLabel = session.current_piece_label || "여정 완료";
  const completed = Array.isArray(session.completed_piece_ids) ? session.completed_piece_ids : [];
  const completedLabels = completed.map(pieceLabel).join(", ") || "아직 없음";
  byId("current-place").textContent = session.current_place_id || "장소 정보 없음";
  byId("current-piece").textContent = currentLabel;
  byId("piece-title").textContent = currentLabel;
  byId("piece-context-place").textContent = session.current_place_id || "현재 장소";
  byId("piece-context-label").textContent = currentLabel;
  byId("completed-count").textContent = String(completed.length);
  byId("journey-step").textContent = session.current_journey_step === "journey_complete" ? "데모 여정 완료" : `현재 단계 · ${session.current_journey_step}`;
  byId("chat-mode").textContent = session.chat_mode;
  byId("free-context").textContent = `현재 장소 ${session.current_place_id || "정보 없음"} · 현재 조각 ${currentLabel} · 완료한 조각 ${completedLabels}`;
  byId("journey-stage").style.setProperty("--stage-image", `url('${resolveBackground(session)}')`);
  byId("progress-fill").style.width = `${Math.min(100, (completed.length / 3) * 100)}%`;
  byId("progress-fill").parentElement.setAttribute("aria-valuenow", String(completed.length));
  document.querySelectorAll("[data-action='GO_NEXT_PIECE']").forEach((button) => { button.disabled = session.current_piece_id === null; });
}

function appendMessage(role, text, extraClass = "", outputDomain = "character_dialogue") {
  const item = document.createElement("p");
  item.className = `message ${role} ${extraClass}`.trim();
  item.textContent = text;
  item.dataset.outputDomain = outputDomain;
  byId("free-messages").appendChild(item);
  item.scrollIntoView({ block: "nearest" });
}

function renderCitations(citations) {
  const panel = byId("citation-panel");
  const toggle = byId("citation-toggle");
  panel.replaceChildren();
  const valid = Array.isArray(citations) ? citations : [];
  toggle.hidden = valid.length === 0;
  panel.hidden = true;
  toggle.setAttribute("aria-expanded", "false");
  byId("citation-count").textContent = String(valid.length);
  valid.forEach((citation) => {
    const article = document.createElement("article");
    article.className = "citation";
    const header = document.createElement("div");
    header.className = "citation-header";
    const title = document.createElement("strong");
    title.textContent = citation.title || citation.source_id || "출처";
    header.appendChild(title);
    if (typeof citation.badge_label === "string" && citation.badge_label.trim()) {
      const badge = document.createElement("span");
      badge.className = "citation-badge";
      badge.textContent = citation.badge_label.trim();
      header.appendChild(badge);
    }
    article.appendChild(header);
    const institution = citation.institution || citation.publisher;
    if (institution) {
      const meta = document.createElement("div");
      meta.textContent = institution;
      article.appendChild(meta);
    }
    if (citation.excerpt) {
      const excerpt = document.createElement("p");
      excerpt.textContent = citation.excerpt;
      article.appendChild(excerpt);
    }
    if (typeof citation.usage_notice === "string" && citation.usage_notice.trim()) {
      const notice = document.createElement("p");
      notice.className = "citation-notice";
      notice.textContent = citation.usage_notice.trim();
      article.appendChild(notice);
    }
    const url = citation.source_url || citation.url;
    if (url) {
      const link = document.createElement("a");
      link.href = url;
      link.textContent = "원문 보기";
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      article.appendChild(link);
    }
    panel.appendChild(article);
  });
}

function renderSuggestions(values) {
  const host = byId("suggestions");
  host.replaceChildren();
  (values || []).forEach((value) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = value;
    button.addEventListener("click", () => sendFree(value));
    host.appendChild(button);
  });
}

function executeSuggestedAction(action, transition = null) {
  if (action === "OPEN_FREE_CHAT" || action === "OFFER_MORE_HISTORY_IN_FREE_CHAT") return openFree(transition);
  if (ACTION_PRESENTATION[action]) return journeyAction(action);
  announce("현재 데모에서 바로 실행할 수 없는 동작입니다.");
  return Promise.resolve();
}

function renderActionHint(action, transition = null) {
  const host = byId("action-hint");
  const presentation = ACTION_PRESENTATION[action];
  if (!presentation) {
    host.hidden = true;
    return;
  }
  byId("action-hint-text").textContent = presentation.description;
  const button = byId("action-hint-button");
  button.textContent = presentation.label;
  button.onclick = () => executeSuggestedAction(action, transition);
  host.hidden = false;
}

async function sendPiece(message) {
  const normalized = message.trim();
  if (!normalized || state.request === "loading" || state.lastRequest === `piece:${normalized}`) return;
  state.lastRequest = `piece:${normalized}`;
  setRequestState("loading", "기록새가 답을 준비하고 있어요");
  let pendingTransition = null;
  let transitionNotice = "";
  try {
    const result = await api("/api/chat/piece", {
      method: "POST",
      body: JSON.stringify({ session_id: state.session.session_id, user_message: normalized, ui_state: "awaiting_reflection" }),
    });
    byId("piece-response").textContent = result.response_text;
    byId("piece-response").dataset.outputDomain = result.output_domain;
    renderActionHint(result.next_action_code, result.mode_transition);
    if (result.mode_transition) {
      pendingTransition = result.mode_transition;
      transitionNotice = result.response_text;
    }
    setRequestState(result.request_state === "insufficient_evidence" ? "insufficient_evidence" : "success");
  } catch (error) {
    byId("piece-response").textContent = error.message;
    byId("piece-response").dataset.outputDomain = "system_ui";
    setRequestState("error", error.message);
  } finally {
    state.lastRequest = "";
    if (state.request !== "error" && state.request !== "insufficient_evidence") setRequestState("idle");
    byId("piece-input").focus();
  }
  if (pendingTransition) await openFree(pendingTransition, transitionNotice);
}

async function sendFree(message) {
  const normalized = message.trim();
  if (!normalized || state.request === "loading" || state.lastRequest === `free:${normalized}`) return;
  state.lastRequest = `free:${normalized}`;
  appendMessage("user", normalized);
  setRequestState("loading");
  try {
    const result = await api("/api/chat/free", {
      method: "POST",
      body: JSON.stringify({ session_id: state.session.session_id, user_message: normalized, ui_state: "active" }),
    });
    const insufficient = result.request_state === "insufficient_evidence";
    appendMessage("assistant", result.response_text, insufficient ? "state" : "", result.output_domain);
    renderCitations(result.citations);
    renderSuggestions(result.suggested_questions);
    setRequestState(insufficient ? "insufficient_evidence" : "success");
  } catch (error) {
    appendMessage("assistant", error.message, "state", "system_ui");
    renderCitations([]);
    setRequestState("error", error.message);
  } finally {
    state.lastRequest = "";
    byId("free-input").focus();
  }
}

async function journeyAction(action, extra = {}) {
  if (state.request === "loading") return;
  setRequestState("loading", "여정 상태를 확인하고 있어요");
  try {
    const result = await api("/api/journey/action", {
      method: "POST",
      body: JSON.stringify({ session_id: state.session.session_id, action_code: action, ...extra }),
    });
    renderSession(result.session);
    byId("action-hint").hidden = true;
    setRequestState("success", action === "GO_NEXT_PIECE" ? "다음 조각으로 이동했어요" : "여정 상태에 반영했어요");
  } catch (error) {
    setRequestState("error", error.message);
  } finally {
    if (state.request === "success") window.setTimeout(() => setRequestState("idle"), 700);
  }
}

async function openFree(transition = null, notice = "") {
  if (!state.session || state.request === "loading" || state.freeOpen) return;
  state.lastFocus = document.activeElement;
  byId("free-chat-backdrop").hidden = false;
  state.freeOpen = true;
  setRequestState("loading", "자유대화를 열고 있어요");
  try {
    const result = await api("/api/chat/transition", {
      method: "POST",
      body: JSON.stringify({ session_id: state.session.session_id, from_mode: "piece_chat", to_mode: "free_chat", mode_transition: transition }),
    });
    renderSession(result.session);
    if (notice) appendMessage("assistant", notice);
    setRequestState("idle");
    byId("free-chat-panel").focus();
    if (transition && transition.pending_user_question) await sendFree(transition.pending_user_question);
  } catch (error) {
    appendMessage("assistant", error.message, "state", "system_ui");
    setRequestState("error", error.message);
  }
}

async function closeFree() {
  if (!state.freeOpen || state.request === "loading") return;
  setRequestState("loading", "게임 화면으로 돌아가고 있어요");
  try {
    const result = await api("/api/chat/transition", {
      method: "POST",
      body: JSON.stringify({ session_id: state.session.session_id, from_mode: "free_chat", to_mode: "game" }),
    });
    renderSession(result.session);
    byId("free-chat-backdrop").hidden = true;
    state.freeOpen = false;
    renderCitations([]);
    setRequestState("idle");
    if (state.lastFocus) state.lastFocus.focus();
  } catch (error) {
    appendMessage("assistant", error.message, "state", "system_ui");
    setRequestState("error", error.message);
  }
}

function trapFocus(event) {
  if (!state.freeOpen) return;
  if (event.key === "Escape") {
    event.preventDefault();
    closeFree();
    return;
  }
  if (event.key !== "Tab") return;
  const focusable = [...byId("free-chat-panel").querySelectorAll("button:not([disabled]),input:not([disabled]),a[href]")];
  if (!focusable.length) return;
  const first = focusable[0];
  const last = focusable[focusable.length - 1];
  if (event.shiftKey && document.activeElement === first) {
    event.preventDefault();
    last.focus();
  } else if (!event.shiftKey && document.activeElement === last) {
    event.preventDefault();
    first.focus();
  }
}

async function initialize() {
  try {
    await api("/health");
    byId("server-status").textContent = "연결됨";
    byId("server-pill").dataset.state = "ready";
    const saved = sessionStorage.getItem("historyPiecesSession");
    let session = null;
    if (saved) {
      try { session = await api(`/api/session/${encodeURIComponent(saved)}`); } catch (_error) { session = null; }
    }
    if (!session) session = await api("/api/session", { method: "POST", body: JSON.stringify({ locale: "ko" }) });
    sessionStorage.setItem("historyPiecesSession", session.session_id);
    renderSession(session);
    state.app = "ready";
    setRequestState("idle");
  } catch (error) {
    state.app = "disconnected";
    byId("server-status").textContent = "연결 안 됨";
    byId("server-pill").dataset.state = "error";
    setRequestState("error", error.message);
  }
}

byId("piece-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const input = byId("piece-input");
  sendPiece(input.value);
  input.value = "";
});
byId("free-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const input = byId("free-input");
  sendFree(input.value);
  input.value = "";
});
document.querySelectorAll("[data-quick]").forEach((button) => button.addEventListener("click", () => sendPiece(button.dataset.quick)));
document.querySelectorAll("[data-action]").forEach((button) => button.addEventListener("click", () => journeyAction(button.dataset.action)));
["floating-chat", "open-free-chat", "header-free-chat"].forEach((id) => byId(id).addEventListener("click", () => openFree()));
["close-free-chat", "return-to-game"].forEach((id) => byId(id).addEventListener("click", closeFree));
byId("citation-toggle").addEventListener("click", () => {
  const panel = byId("citation-panel");
  panel.hidden = !panel.hidden;
  byId("citation-toggle").setAttribute("aria-expanded", String(!panel.hidden));
  if (!panel.hidden) panel.focus?.();
});
document.addEventListener("keydown", trapFocus);
initialize();
