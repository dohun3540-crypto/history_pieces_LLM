"use strict";

const state = { app:"initializing", request:"idle", session:null, freeOpen:false, lastRequest:"", lastFocus:null };
const byId = (id) => document.getElementById(id);
const statusNode = byId("app-status");

function announce(message) { statusNode.textContent = message; }
function setBusy(busy) {
  state.request = busy ? "loading" : "idle";
  document.querySelectorAll("button, input").forEach((node) => { node.disabled = busy; });
  announce(busy ? "요청을 처리하고 있습니다." : "요청 처리가 끝났습니다.");
}
async function api(path, options={}) {
  const response = await fetch(path, {headers:{"Content-Type":"application/json"}, ...options});
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.message || "요청을 처리하지 못했습니다.");
  return payload;
}
function renderSession(session) {
  state.session = session;
  byId("current-place").textContent = session.current_place_id || "없음";
  byId("current-piece").textContent = session.current_piece_label || "여정 완료";
  byId("piece-title").textContent = session.current_piece_label || "여정 완료";
  byId("completed-count").textContent = String(session.completed_piece_ids.length);
  byId("journey-step").textContent = session.current_journey_step;
  byId("chat-mode").textContent = session.chat_mode;
  byId("free-context").textContent = `장소 ${session.current_place_id} · 현재 ${session.current_piece_label} · 완료 ${session.completed_piece_ids.length}개`;
}
function appendMessage(role, text, extraClass="", outputDomain="character_dialogue") {
  const item = document.createElement("p");
  item.className = `message ${role} ${extraClass}`.trim();
  item.textContent = text;
  item.dataset.outputDomain = outputDomain;
  byId("free-messages").appendChild(item);
  item.scrollIntoView({block:"nearest"});
}
function renderCitations(citations) {
  const panel = byId("citation-panel");
  const toggle = byId("citation-toggle");
  panel.replaceChildren();
  const valid = Array.isArray(citations) ? citations : [];
  toggle.hidden = valid.length === 0;
  panel.hidden = true;
  toggle.setAttribute("aria-expanded", "false");
  valid.forEach((citation) => {
    const article = document.createElement("article"); article.className = "citation";
    const title = document.createElement("strong"); title.textContent = citation.title || citation.source_id || "출처"; article.appendChild(title);
    const institution = citation.institution || citation.publisher;
    if (institution) { const meta=document.createElement("div"); meta.textContent=institution; article.appendChild(meta); }
    if (citation.excerpt) { const excerpt=document.createElement("p"); excerpt.textContent=citation.excerpt; article.appendChild(excerpt); }
    const url = citation.source_url || citation.url;
    if (url) { const link=document.createElement("a"); link.href=url; link.textContent="원문 보기"; link.target="_blank"; link.rel="noopener noreferrer"; article.appendChild(link); }
    panel.appendChild(article);
  });
}
function renderSuggestions(values) {
  const host=byId("suggestions"); host.replaceChildren();
  (values || []).forEach((value) => { const button=document.createElement("button"); button.type="button"; button.textContent=value; button.addEventListener("click",()=>sendFree(value)); host.appendChild(button); });
}
async function sendPiece(message) {
  const normalized=message.trim(); if (!normalized || state.request==="loading" || state.lastRequest===`piece:${normalized}`) return;
  state.lastRequest=`piece:${normalized}`; setBusy(true); let pendingTransition=null; let transitionNotice="";
  try {
    const result=await api("/api/chat/piece",{method:"POST",body:JSON.stringify({session_id:state.session.session_id,user_message:normalized,ui_state:"awaiting_reflection"})});
    byId("piece-response").textContent=result.response_text;
    byId("piece-response").dataset.outputDomain=result.output_domain;
    if (result.mode_transition) { pendingTransition=result.mode_transition; transitionNotice=result.response_text; }
  } catch(error) { byId("piece-response").textContent=error.message; state.request="error"; }
  finally { state.lastRequest=""; setBusy(false); byId("piece-input").focus(); }
  if (pendingTransition) await openFree(pendingTransition, transitionNotice);
}
async function sendFree(message) {
  const normalized=message.trim(); if (!normalized || state.request==="loading" || state.lastRequest===`free:${normalized}`) return;
  state.lastRequest=`free:${normalized}`; appendMessage("user",normalized); setBusy(true);
  try {
    const result=await api("/api/chat/free",{method:"POST",body:JSON.stringify({session_id:state.session.session_id,user_message:normalized,ui_state:"active"})});
    appendMessage("assistant",result.response_text,result.request_state==="insufficient_evidence"?"state":"",result.output_domain);
    renderCitations(result.citations); renderSuggestions(result.suggested_questions);
    if (result.request_state==="insufficient_evidence") announce("확인 가능한 근거가 부족합니다.");
  } catch(error) { appendMessage("assistant",error.message,"state"); state.request="error"; }
  finally { state.lastRequest=""; setBusy(false); byId("free-input").focus(); }
}
async function journeyAction(action, extra={}) {
  if (state.request==="loading") return; setBusy(true);
  try { const result=await api("/api/journey/action",{method:"POST",body:JSON.stringify({session_id:state.session.session_id,action_code:action,...extra})}); renderSession(result.session); }
  catch(error) { announce(error.message); state.request="error"; }
  finally { setBusy(false); }
}
async function openFree(transition=null, notice="") {
  if (!state.session || state.request==="loading") return;
  state.lastFocus=document.activeElement; byId("free-chat-backdrop").hidden=false; state.freeOpen=true;
  try {
    const result=await api("/api/chat/transition",{method:"POST",body:JSON.stringify({session_id:state.session.session_id,from_mode:"piece_chat",to_mode:"free_chat",mode_transition:transition})});
    renderSession(result.session); if (notice) appendMessage("assistant",notice);
    byId("free-chat-panel").focus();
    if (transition && transition.pending_user_question) await sendFree(transition.pending_user_question);
  } catch(error) { appendMessage("assistant",error.message,"state"); }
}
async function closeFree() {
  if (!state.freeOpen || state.request==="loading") return;
  byId("free-chat-panel").setAttribute("aria-busy","true");
  try {
    const result=await api("/api/chat/transition",{method:"POST",body:JSON.stringify({session_id:state.session.session_id,from_mode:"free_chat",to_mode:"game"})});
    renderSession(result.session); byId("free-chat-backdrop").hidden=true; state.freeOpen=false; renderCitations([]); if(state.lastFocus) state.lastFocus.focus();
  } catch(error) { appendMessage("assistant",error.message,"state"); }
  finally { byId("free-chat-panel").removeAttribute("aria-busy"); }
}
function trapFocus(event) {
  if (!state.freeOpen) return;
  if (event.key==="Escape") { closeFree(); return; }
  if (event.key!=="Tab") return;
  const focusable=[...byId("free-chat-panel").querySelectorAll("button:not([disabled]),input:not([disabled]),a[href]")];
  if (!focusable.length) return; const first=focusable[0],last=focusable[focusable.length-1];
  if (event.shiftKey && document.activeElement===first) { event.preventDefault(); last.focus(); }
  else if (!event.shiftKey && document.activeElement===last) { event.preventDefault(); first.focus(); }
}
async function initialize() {
  try { await api("/health"); byId("server-status").textContent="연결됨"; const saved=sessionStorage.getItem("historyPiecesSession");
    let session; if(saved){ try{session=await api(`/api/session/${encodeURIComponent(saved)}`);}catch(_error){session=null;} }
    if(!session) session=await api("/api/session",{method:"POST",body:JSON.stringify({locale:"ko"})});
    sessionStorage.setItem("historyPiecesSession",session.session_id); renderSession(session); state.app="ready";
  } catch(error) { state.app="disconnected"; byId("server-status").textContent="연결 안 됨"; announce(error.message); }
}
byId("piece-form").addEventListener("submit",(event)=>{event.preventDefault();const input=byId("piece-input");sendPiece(input.value);input.value="";});
byId("free-form").addEventListener("submit",(event)=>{event.preventDefault();const input=byId("free-input");sendFree(input.value);input.value="";});
document.querySelectorAll("[data-quick]").forEach((button)=>button.addEventListener("click",()=>sendPiece(button.dataset.quick)));
document.querySelectorAll("[data-action]").forEach((button)=>button.addEventListener("click",()=>journeyAction(button.dataset.action)));
byId("floating-chat").addEventListener("click",()=>openFree()); byId("open-free-chat").addEventListener("click",()=>openFree()); byId("close-free-chat").addEventListener("click",closeFree);
byId("citation-toggle").addEventListener("click",()=>{const panel=byId("citation-panel");panel.hidden=!panel.hidden;byId("citation-toggle").setAttribute("aria-expanded",String(!panel.hidden));});
document.addEventListener("keydown",trapFocus); initialize();
