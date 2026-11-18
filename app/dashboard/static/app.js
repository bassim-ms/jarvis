/*
 * Jarvis dashboard (Dashboard Slice 1). Plain JavaScript, no libraries, no external requests.
 * Every value from the server is written with textContent - never as HTML - so nothing stored in Jarvis can run as code here.
 * Reading the dashboard never calls an AI provider; only sending a chat message does (through the existing POST /chat).
 */
"use strict";

const REFRESH_MS = 15000;
const CONVERSATION_KEY = "jarvis.conversation_id";

const $ = (id) => document.getElementById(id);

function el(tag, options = {}, children = []) {
  const node = document.createElement(tag);
  if (options.className) node.className = options.className;
  if (options.text !== undefined && options.text !== null) node.textContent = String(options.text);
  if (options.title) node.title = options.title;
  if (options.attrs) {
    for (const [name, value] of Object.entries(options.attrs)) node.setAttribute(name, value);
  }
  for (const child of children) if (child) node.append(child);
  return node;
}

function readStored(key) {
  try { return window.localStorage.getItem(key); } catch (error) { return null; }
}

function writeStored(key, value) {
  try {
    if (value === null) window.localStorage.removeItem(key);
    else window.localStorage.setItem(key, value);
  } catch (error) { /* private window or blocked storage: the conversation just is not remembered */ }
}

async function api(path, options = {}) {
  const init = { method: options.method || "GET", headers: { Accept: "application/json" } };
  if (init.method !== "GET") {
    init.headers["Content-Type"] = "application/json";
    init.headers["X-Jarvis-Dashboard"] = "1";
    init.body = JSON.stringify(options.body || {});
  }
  const response = await fetch(path, init);
  let data = null;
  try { data = await response.json(); } catch (error) { data = null; }
  if (!response.ok) {
    const detail = data && data.detail ? (typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail)) : response.statusText;
    const failure = new Error(detail || `HTTP ${response.status}`);
    failure.status = response.status;
    throw failure;
  }
  return data;
}

let toastTimer = null;
function toast(message, bad = false) {
  const box = $("toast");
  box.textContent = message;
  box.className = bad ? "toast bad" : "toast";
  box.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { box.hidden = true; }, 4000);
}

function when(iso) {
  if (!iso) return "";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return String(iso);
  return date.toLocaleString(undefined, { weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
}

function emptyItem(text) {
  return el("li", { className: "muted", text });
}

function pill(value) {
  return el("span", { className: `pill ${String(value).replace(/[^a-z_]/gi, "")}`, text: String(value).replace(/_/g, " ") });
}

/* ---------------------------------------------------------------- chat */

let conversationId = readStored(CONVERSATION_KEY);
let sending = false;

function addMessage(role, text, extras = {}) {
  const box = $("messages");
  const placeholder = box.querySelector(".empty");
  if (placeholder) placeholder.remove();
  const bubble = el("div", { className: `msg ${role}`, text });
  if (extras.sources && extras.sources.length) {
    const list = el("ol", { className: "sources" });
    for (const source of extras.sources) {
      const item = el("li");
      const safe = /^https?:\/\//i.test(source.url || "");
      if (safe) {
        item.append(el("a", { text: source.title || source.url, attrs: { href: source.url, target: "_blank", rel: "noopener noreferrer" } }));
      } else {
        item.textContent = source.title || source.url || "source";
      }
      list.append(item);
    }
    bubble.append(list);
  }
  if (extras.meta) bubble.append(el("span", { className: "meta", text: extras.meta }));
  box.append(bubble);
  box.scrollTop = box.scrollHeight;
  return bubble;
}

function showEmptyChat() {
  const box = $("messages");
  box.replaceChildren(el("p", { className: "empty", text: "Say hello, sir - or ask for anything." }));
}

async function restoreConversation() {
  if (!conversationId) { showEmptyChat(); return; }
  try {
    const data = await api(`/api/conversations/${encodeURIComponent(conversationId)}/messages`);
    $("messages").replaceChildren();
    if (!data.messages.length) showEmptyChat();
    for (const message of data.messages) addMessage(message.role === "user" ? "user" : "assistant", message.content);
  } catch (error) {
    if (error.status === 404) { conversationId = null; writeStored(CONVERSATION_KEY, null); }
    showEmptyChat();
  }
}

async function sendChat(text) {
  if (sending) return;
  sending = true;
  $("chat-send").disabled = true;
  addMessage("user", text);
  const waiting = addMessage("assistant pending", "Thinking...");
  try {
    const data = await api("/chat", { method: "POST", body: { message: text, conversation_id: conversationId } });
    waiting.remove();
    conversationId = data.conversation_id;
    writeStored(CONVERSATION_KEY, conversationId);
    addMessage("assistant", data.reply, {
      sources: data.sources,
      meta: data.job_id ? `Started a Job (${data.job_id.slice(0, 8)}) - follow it under Jobs.` : "",
    });
    refresh();
  } catch (error) {
    waiting.remove();
    if (error.status === 404) { conversationId = null; writeStored(CONVERSATION_KEY, null); }
    addMessage("error", `Jarvis could not answer: ${error.message}`);
  } finally {
    sending = false;
    $("chat-send").disabled = false;
    $("chat-input").focus();
  }
}

/* ---------------------------------------------------------------- panels */

function renderOverview(data) {
  const ok = data.health && data.health.status === "ok";
  $("health-dot").className = ok ? "dot ok" : "dot bad";
  const counts = data.counts;
  $("status-line").textContent =
    `${counts.open_tasks} open task${counts.open_tasks === 1 ? "" : "s"} - ${counts.active_reminders} reminder${counts.active_reminders === 1 ? "" : "s"} - ` +
    `${counts.active_jobs} active Job${counts.active_jobs === 1 ? "" : "s"} - ${counts.waiting_for_you} waiting for you`;
  const list = $("provider-list");
  list.replaceChildren();
  if (!data.providers.length) list.append(emptyItem("No AI provider is configured."));
  for (const provider of data.providers) {
    const resting = provider.resting_seconds > 0 ? ` - resting ${Math.ceil(provider.resting_seconds / 60)} min` : "";
    list.append(el("li", { title: provider.key_status_text || "" }, [
      el("strong", { text: provider.display_name || provider.name }),
      el("span", { className: "sub", text: `${provider.model || ""}${resting}` }),
      document.createTextNode(" "),
      pill(provider.enabled === false ? "disabled" : provider.key_status || "unknown"),
    ]));
  }
}

function renderTasks(data) {
  const list = $("task-list");
  list.replaceChildren();
  if (!data.tasks.length) list.append(emptyItem("Nothing here."));
  for (const task of data.tasks) {
    const open = task.status === "todo";
    const details = [task.priority !== "medium" ? `${task.priority} priority` : "", task.due_date ? `due ${task.due_date}` : "", !open ? task.status : ""]
      .filter(Boolean).join(" - ");
    const body = el("div", { className: "grow" }, [
      el("div", { className: open ? "" : "done-text", text: task.text }),
      details ? el("div", { className: "sub", text: details }) : null,
    ]);
    const actions = el("div", { className: "actions" });
    if (open) {
      const done = el("button", { className: "small", text: "Done", attrs: { type: "button" } });
      done.addEventListener("click", () => taskAction(task.id, "complete", done));
      const cancel = el("button", { className: "small danger", text: "Cancel", attrs: { type: "button", "aria-label": "Cancel task" } });
      cancel.addEventListener("click", () => taskAction(task.id, "cancel", cancel));
      actions.append(done, cancel);
    }
    list.append(el("li", {}, [el("div", { className: "row" }, [body, actions])]));
  }
}

async function taskAction(taskId, action, button) {
  button.disabled = true;
  try {
    const data = await api(`/api/tasks/${encodeURIComponent(taskId)}/${action}`, { method: "POST" });
    toast(data.message || "Done.");
  } catch (error) {
    toast(error.message, true);
  }
  refresh();
}

function renderReminders(data) {
  const list = $("reminder-list");
  list.replaceChildren();
  $("reminder-count").textContent = String(data.reminders.length);
  if (!data.reminders.length) list.append(emptyItem("No reminders."));
  for (const reminder of data.reminders) {
    const sub = [when(reminder.due_at), reminder.repeats ? "repeats" : "", reminder.status !== "pending" ? reminder.status : ""]
      .filter(Boolean).join(" - ");
    list.append(el("li", {}, [el("div", { text: reminder.text }), el("div", { className: "sub", text: sub })]));
  }
}

function renderJobs(data) {
  const list = $("job-list");
  list.replaceChildren();
  $("job-count").textContent = String(data.jobs.filter((job) => job.active).length);
  if (!data.jobs.length) list.append(emptyItem("No Jobs yet."));
  for (const job of data.jobs.slice(0, 10)) {
    const outcome = job.error || job.result || "";
    list.append(el("li", {}, [
      el("div", { className: "row" }, [el("div", { className: "grow", text: job.goal }), pill(job.status)]),
      el("div", { className: "sub", text: `${when(job.updated_at)}${outcome ? " - " + outcome : ""}` }),
    ]));
  }
}

function renderInteractions(data) {
  const list = $("waiting-list");
  list.replaceChildren();
  const count = data.interactions.length;
  const badge = $("waiting-count");
  badge.textContent = String(count);
  badge.className = count ? "badge attention" : "badge";
  if (!count) list.append(emptyItem("Nothing is waiting for you."));
  for (const item of data.interactions) {
    const parts = [el("div", { text: item.prompt }),
      el("div", { className: "sub", text: `${item.type === "confirmation" ? "Needs your approval" : "Needs your answer"} - code ${item.code}${item.expires_at ? " - until " + when(item.expires_at) : ""}` })];
    const row = el("div", { className: "answer" });
    if (item.type === "confirmation") {
      const yes = el("button", { className: "small", text: "Approve", attrs: { type: "button" } });
      yes.addEventListener("click", () => sendChat(`answer ${item.code}: yes`));
      const no = el("button", { className: "small secondary", text: "Decline", attrs: { type: "button" } });
      no.addEventListener("click", () => sendChat(`answer ${item.code}: no`));
      row.append(yes, no);
    } else {
      const input = el("input", { attrs: { type: "text", placeholder: "Your answer...", maxlength: "2000", "aria-label": "Answer" } });
      const send = el("button", { className: "small", text: "Answer", attrs: { type: "button" } });
      const submit = () => { const value = input.value.trim(); if (value) { sendChat(`answer ${item.code}: ${value}`); input.value = ""; } };
      send.addEventListener("click", submit);
      input.addEventListener("keydown", (event) => { if (event.key === "Enter") { event.preventDefault(); submit(); } });
      row.append(input, send);
    }
    parts.push(row);
    list.append(el("li", {}, parts));
  }
}

function renderNotes(data) {
  const list = $("note-list");
  list.replaceChildren();
  if (!data.notes.length) list.append(emptyItem("No notes yet."));
  for (const note of data.notes.slice(0, 10)) {
    const body = el("div", { className: "note-body", text: note.preview || "" });
    const details = el("details", {}, [el("summary", { text: note.title || "(untitled)" }), el("div", { className: "sub", text: when(note.created_at) }), body]);
    let loaded = false;
    details.addEventListener("toggle", async () => {
      if (!details.open || loaded) return;
      try { body.textContent = (await api(`/api/notes/${encodeURIComponent(note.id)}`)).content; loaded = true; } catch (error) { /* keep the preview */ }
    });
    list.append(el("li", {}, [details]));
  }
}

let refreshing = false;
async function refresh() {
  if (refreshing) return;
  refreshing = true;
  const filter = $("task-filter").value;
  const jobs = [
    ["/api/overview", renderOverview],
    [`/api/tasks?status=${encodeURIComponent(filter)}`, renderTasks],
    ["/api/reminders", renderReminders],
    ["/api/jobs", renderJobs],
    ["/api/interactions", renderInteractions],
    ["/api/notes", renderNotes],
  ];
  const results = await Promise.allSettled(jobs.map(([path]) => api(path)));
  results.forEach((result, index) => {
    if (result.status === "fulfilled") jobs[index][1](result.value);
  });
  if (results[0].status === "rejected") {
    $("health-dot").className = "dot bad";
    $("status-line").textContent = "Jarvis is not answering - is the server running?";
  }
  refreshing = false;
}

function tickClock() {
  $("clock").textContent = new Date().toLocaleString(undefined, { weekday: "short", hour: "2-digit", minute: "2-digit" });
}

/* ---------------------------------------------------------------- wiring */

document.addEventListener("DOMContentLoaded", () => {
  $("chat-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const input = $("chat-input");
    const text = input.value.trim();
    if (!text) return;
    input.value = "";
    sendChat(text);
  });
  $("chat-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      $("chat-form").requestSubmit();
    }
  });
  $("new-chat").addEventListener("click", () => {
    conversationId = null;
    writeStored(CONVERSATION_KEY, null);
    showEmptyChat();
    $("chat-input").focus();
  });
  $("task-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const text = $("task-text").value.trim();
    if (!text) return;
    const body = { text, priority: $("task-priority").value };
    if ($("task-due").value) body.due_date = $("task-due").value;
    try {
      const data = await api("/api/tasks", { method: "POST", body });
      $("task-text").value = "";
      $("task-due").value = "";
      toast(data.message || "Task added.");
    } catch (error) {
      toast(error.message, true);
    }
    refresh();
  });
  $("task-filter").addEventListener("change", refresh);

  tickClock();
  setInterval(tickClock, 30000);
  restoreConversation();
  refresh();
  setInterval(refresh, REFRESH_MS);
});
