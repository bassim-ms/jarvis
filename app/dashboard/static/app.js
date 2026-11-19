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
      el("div", { className: "sub", text: providerStatsLine(provider.stats) }),
    ]));
  }
}

/* What the provider's REAL requests show. The answered share is always given as "answered / sample" - never a bare
   percentage, so 2 lucky requests never read as "100% reliable". */
function providerStatsLine(stats) {
  if (!stats || !stats.reliability_sample) return "No requests yet - nothing measured.";
  const parts = [`${stats.reliability_answered}/${stats.reliability_sample} answered (last ${stats.reliability_sample} request${stats.reliability_sample === 1 ? "" : "s"})`];
  if (stats.avg_latency_ms !== null) parts.push(`~${stats.avg_latency_ms} ms`);
  const today = stats.today || {};
  let todayText = `${today.requests || 0} today`;
  if (today.request_limit) todayText += ` of ${today.request_limit}`;
  if (today.tokens) todayText += `, ${today.tokens} tokens`;
  parts.push(todayText);
  if (today.rate_limited) parts.push(`rate-limited ${today.rate_limited}x today`);
  if (stats.last_success_at) parts.push(`last answered ${when(stats.last_success_at)}`);
  if (stats.last_failure_kind && stats.last_failure_at) parts.push(`last failure: ${stats.last_failure_kind.replace(/_/g, " ")} ${when(stats.last_failure_at)}`);
  return parts.join(" - ");
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

/* One action through the server (which runs the EXISTING tool for it), then a toast and a refresh. */
async function act(path, button, confirmText) {
  if (confirmText && !window.confirm(confirmText)) return;
  if (button) button.disabled = true;
  try {
    const data = await api(path, { method: "POST" });
    toast(data.message || "Done.");
  } catch (error) {
    toast(error.message, true);
  }
  refresh();
}

function actionButton(label, path, options = {}) {
  const button = el("button", { className: `small ${options.style || ""}`.trim(), text: label, attrs: { type: "button" } });
  button.addEventListener("click", () => act(path, button, options.confirm));
  return button;
}

function renderReminders(data) {
  const list = $("reminder-list");
  list.replaceChildren();
  $("reminder-count").textContent = String(data.reminders.length);
  if (!data.reminders.length) list.append(emptyItem("No reminders."));
  for (const reminder of data.reminders) {
    const sub = [when(reminder.due_at), reminder.repeats ? `repeats (${reminder.schedule})` : "", reminder.status !== "pending" ? reminder.status : ""]
      .filter(Boolean).join(" - ");
    const base = `/api/reminders/${encodeURIComponent(reminder.id)}`;
    const actions = el("div", { className: "actions" });
    if (reminder.status === "paused") actions.append(actionButton("Resume", `${base}/resume`));
    else if (reminder.status === "triggered") actions.append(actionButton("Done", `${base}/complete`));
    else actions.append(actionButton("Pause", `${base}/pause`, { style: "secondary" }));
    actions.append(actionButton("Cancel", `${base}/cancel`, { style: "danger", confirm: `Cancel the reminder "${reminder.text}"?` }));
    list.append(el("li", {}, [el("div", { className: "row" }, [
      el("div", { className: "grow" }, [el("div", { text: reminder.text }), el("div", { className: "sub", text: sub })]),
      actions,
    ])]));
  }
}

const openJobs = new Set();  // Job details stay open across the 15 s refresh

async function loadJobDetail(jobId, box) {
  try {
    const job = await api(`/api/jobs/${encodeURIComponent(jobId)}`);
    const parts = [];
    if (job.confirmation_message) parts.push(el("div", { className: "sub", text: `Waiting for approval: ${job.confirmation_message}` }));
    if (job.steps.length) {
      const steps = el("ol", { className: "steps" });
      for (const step of job.steps) {
        const extra = step.error ? ` - ${step.error}` : step.result ? ` - ${step.result}` : "";
        steps.append(el("li", {}, [el("strong", { text: step.tool }), document.createTextNode(" "), pill(step.status),
          document.createTextNode(`${step.attempts > 1 ? ` (${step.attempts} attempts)` : ""}${extra}`)]));
      }
      parts.push(steps);
    } else {
      parts.push(el("div", { className: "sub", text: "No plan yet." }));
    }
    if (job.events.length) {
      const last = job.events[job.events.length - 1];
      parts.push(el("div", { className: "sub", text: `Last event: ${last.kind}${last.detail ? " - " + last.detail : ""} (${when(last.at)})` }));
    }
    box.replaceChildren(...parts);
  } catch (error) {
    box.replaceChildren(el("div", { className: "sub", text: `Could not load the plan: ${error.message}` }));
  }
}

function renderJobs(data) {
  const list = $("job-list");
  list.replaceChildren();
  $("job-count").textContent = String(data.jobs.filter((job) => job.active).length);
  if (!data.jobs.length) list.append(emptyItem("No Jobs yet."));
  for (const job of data.jobs.slice(0, 10)) {
    const outcome = job.error || job.result || "";
    const base = `/api/jobs/${encodeURIComponent(job.id)}`;
    const actions = el("div", { className: "actions" });
    if (job.status === "paused") actions.append(actionButton("Resume", `${base}/resume`));
    else if (job.active) actions.append(actionButton("Pause", `${base}/pause`, { style: "secondary" }));
    if (job.active) actions.append(actionButton("Cancel", `${base}/cancel`, { style: "danger", confirm: "Cancel this Job? Steps already done stay done." }));
    const detail = el("div");
    const details = el("details", {}, [el("summary", { text: "Plan and progress" }), detail]);
    details.addEventListener("toggle", () => {
      if (details.open) { openJobs.add(job.id); loadJobDetail(job.id, detail); } else { openJobs.delete(job.id); }
    });
    if (openJobs.has(job.id)) details.open = true;
    list.append(el("li", {}, [
      el("div", { className: "row" }, [el("div", { className: "grow", text: job.goal }), pill(job.status), actions]),
      el("div", { className: "sub", text: `${when(job.updated_at)}${outcome ? " - " + outcome : ""}` }),
      details,
    ]));
  }
}

/* ---------------------------------------------------------------- Slice 2 views */

function renderWorld(data) {
  const projects = $("project-list");
  projects.replaceChildren();
  if (!data.available) projects.append(emptyItem("The World Model is not available."));
  else if (!data.projects.length) projects.append(emptyItem("No projects yet. Tell Jarvis about one in chat."));
  for (const project of data.projects) {
    const goals = el("ul", { className: "goals" });
    for (const goal of project.goals) {
      const sub = [goal.target_date ? `by ${goal.target_date}` : "", goal.linked_tasks ? `${goal.linked_tasks} linked task${goal.linked_tasks === 1 ? "" : "s"}` : ""]
        .filter(Boolean).join(" - ");
      goals.append(el("li", {}, [el("div", { className: "row" }, [el("div", { className: "grow", text: goal.title }), pill(goal.status)]),
        sub ? el("div", { className: "sub", text: sub }) : null]));
    }
    if (!project.goals.length) goals.append(el("li", { className: "sub", text: "No goals yet." }));
    projects.append(el("li", {}, [
      el("div", { className: "row" }, [el("strong", { className: "grow", text: project.name }), pill(project.priority), pill(project.status)]),
      project.description ? el("div", { className: "sub", text: project.description }) : null,
      goals,
    ]));
  }
  const commitments = $("commitment-list");
  commitments.replaceChildren();
  $("commitment-count").textContent = String(data.commitments.length);
  if (!data.commitments.length) commitments.append(emptyItem("No open commitments."));
  for (const item of data.commitments) {
    commitments.append(el("li", {}, [el("div", { text: item.title }), el("div", { className: "sub", text: item.due_at ? `due ${when(item.due_at)}` : "no due date" })]));
  }
  const blockers = $("blocker-list");
  blockers.replaceChildren();
  $("blocker-count").textContent = String(data.blockers.length);
  if (!data.blockers.length) blockers.append(emptyItem("Nothing is blocked."));
  for (const item of data.blockers) {
    blockers.append(el("li", {}, [el("div", { className: "row" }, [el("div", { className: "grow", text: item.title }), pill(item.severity)]),
      item.description ? el("div", { className: "sub", text: item.description }) : null]));
  }
  const events = $("world-event-list");
  events.replaceChildren();
  if (!data.events.length) events.append(emptyItem("No upcoming events."));
  for (const item of data.events.slice(0, 15)) {
    events.append(el("li", {}, [el("div", { text: item.title }), el("div", { className: "sub", text: [when(item.starts_at), item.location].filter(Boolean).join(" - ") })]));
  }
}

const ORIGIN_WORDS = {
  user_command: "you asked me to remember it",
  user_statement: "you told me",
  inference: "I concluded it",
  external: "from an outside source",
};

function renderMemory(data) {
  const kept = $("kept-list");
  kept.replaceChildren();
  $("kept-count").textContent = String(data.remembered.length);
  if (!data.remembered.length) kept.append(emptyItem("Nothing kept yet. Say \"remember that ...\" in chat."));
  for (const item of data.remembered) {
    const p = item.provenance;
    const from = p ? `${ORIGIN_WORDS[p.origin] || p.origin} - ${p.action} ${when(p.saved_at)}` : "no record of when it was saved";
    const value = item.value && !item.text.includes(item.value) ? ` (stored as: ${item.value})` : "";
    const forget = el("button", { className: "small danger", text: "Forget", attrs: { type: "button" } });
    forget.addEventListener("click", () => forgetItem(item, forget));
    kept.append(el("li", {}, [
      el("div", { className: "row" }, [el("div", { className: "grow", text: item.text + value }), pill(item.kind), forget]),
      el("div", { className: "sub", text: from }),
    ]));
  }
  const candidates = $("candidate-list");
  candidates.replaceChildren();
  $("candidate-count").textContent = String(data.candidates.length);
  if (!data.candidates.length) candidates.append(emptyItem("Nothing waiting."));
  for (const item of data.candidates) {
    candidates.append(el("li", {}, [el("div", { className: "kv" }, [el("span", { className: "k", text: item.key }), el("span", { text: item.value })]),
      el("div", { className: "sub", text: `${item.kind} - confidence ${Math.round((item.confidence || 0) * 100)}%` })]));
  }
  const history = $("memory-history");
  history.replaceChildren();
  if (!data.history.length) history.append(emptyItem("No changes yet."));
  for (const entry of data.history.slice(0, 15)) {
    const what = entry.value === null || entry.value === undefined ? entry.key : `${entry.key}: ${entry.value}`;
    history.append(el("li", {}, [el("div", { className: "row" }, [el("div", { className: "grow", text: what }), pill(entry.action)]),
      el("div", { className: "sub", text: `${when(entry.at)} - ${entry.reason}` })]));
  }
}

/* Forgetting is two steps on the SERVER: prepare names the exact item and issues a single-use token; only confirming with it
   forgets. The browser's own "are you sure?" sits between them, showing the item exactly as kept. */
async function forgetItem(item, button) {
  button.disabled = true;
  try {
    const prepared = await api("/api/memory/forget/prepare", { method: "POST", body: { kind: item.kind, id: item.id } });
    if (!window.confirm(`Forget this? It cannot be undone.

"${prepared.text}"`)) { button.disabled = false; return; }
    const done = await api("/api/memory/forget/confirm", { method: "POST", body: { token: prepared.token } });
    toast(done.message);
  } catch (error) {
    toast(error.message, true);
  }
  loadTab("memory");
}

function renderAutomations(data) {
  const list = $("automation-list");
  list.replaceChildren();
  $("automation-count").textContent = String(data.automations.filter((item) => item.status === "enabled").length);
  if (!data.available) list.append(emptyItem("Automations are not available."));
  else if (!data.automations.length) list.append(emptyItem("No automations yet. Ask Jarvis in chat to create one."));
  for (const item of data.automations) {
    const base = `/api/automations/${encodeURIComponent(item.id)}`;
    const actions = el("div", { className: "actions" });
    if (item.status === "enabled") actions.append(actionButton("Disable", `${base}/disable`, { style: "secondary" }));
    if (item.status === "disabled") actions.append(actionButton("Enable", `${base}/enable`));
    if (item.status !== "cancelled") actions.append(actionButton("Cancel", `${base}/cancel`, { style: "danger", confirm: `Cancel the automation "${item.name}"? This cannot be undone.` }));
    const trigger = item.every ? `every ${item.every}` : item.trigger.replace(/_/g, " ");
    const last = item.last_run ? `last run ${when(item.last_run.at)}: ${item.last_run.outcome}` : "never run";
    list.append(el("li", {}, [
      el("div", { className: "row" }, [el("div", { className: "grow", text: item.name }), pill(item.status), actions]),
      el("div", { className: "sub", text: `${trigger} - runs ${item.action}${item.next_run_at && item.status === "enabled" ? " - next " + when(item.next_run_at) : ""}` }),
      el("div", { className: "sub", text: `${last} (${item.runs} run${item.runs === 1 ? "" : "s"})` }),
    ]));
  }
}

function renderCalendar(data) {
  const list = $("calendar-list");
  list.replaceChildren();
  if (!data.connected) { list.append(emptyItem(data.message || "Calendar is not connected.")); return; }
  if (!data.events.length) list.append(emptyItem("Nothing in the next 7 days."));
  for (const event of data.events) {
    list.append(el("li", {}, [el("div", { text: event.title || "(no title)" }),
      el("div", { className: "sub", text: [when(event.start), event.location].filter(Boolean).join(" - ") })]));
  }
}

function megabytes(bytes) {
  if (bytes === null || bytes === undefined) return "?";
  const mb = bytes / 1024 / 1024;
  if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`;
  return `${mb.toFixed(mb < 10 ? 1 : 0)} MB`;
}

function renderSystem(data) {
  const status = $("system-status");
  const findings = $("system-findings");
  findings.replaceChildren();
  if (!data.available) { status.textContent = "not available"; return; }
  status.textContent = data.status;
  status.className = `pill ${data.status === "ok" ? "completed" : data.status === "warning" ? "paused" : "failed"}`;
  for (const problem of data.problems) findings.append(el("li", { className: "", text: `Problem: ${problem}` }));
  for (const warning of data.warnings) findings.append(el("li", { className: "muted", text: warning }));

  const services = $("service-list");
  services.replaceChildren();
  for (const service of data.services) {
    const word = service.ok === true ? "ok" : service.ok === false ? "failed" : "off";
    services.append(el("li", {}, [el("div", { className: "row" }, [el("div", { className: "grow", text: service.name }),
      el("span", { className: `pill ${service.ok === true ? "completed" : service.ok === false ? "failed" : "disabled"}`, text: word })]),
      el("div", { className: "sub", text: service.detail })]));
  }
  const r = data.resources || {};
  const disk = r.disk ? `${megabytes(r.disk.free_bytes)} free disk (${r.disk.free_percent}%)` : "disk unknown";
  const uptime = data.uptime_seconds === null ? "" : ` - up ${Math.floor(data.uptime_seconds / 3600)} h ${Math.floor((data.uptime_seconds % 3600) / 60)} min`;
  $("resource-line").textContent = `${disk} - databases ${megabytes(r.databases_bytes)} - memory ${megabytes(r.process_memory_bytes)} - ${r.threads} threads${uptime} - checked ${when(data.checked_at)}`;

  const databases = $("database-list");
  databases.replaceChildren();
  for (const db of data.databases) {
    const word = db.ok === true ? "ok" : db.ok === false ? "failed" : "not created";
    const sub = db.ok === true ? `${db.tables} table${db.tables === 1 ? "" : "s"}, ${db.rows} rows, ${megabytes(db.size_bytes)}` : db.detail;
    databases.append(el("li", {}, [el("div", { className: "row" }, [el("div", { className: "grow", text: db.file }), pill(word === "ok" ? "completed" : word === "failed" ? "failed" : "pending")]),
      el("div", { className: "sub", text: sub })]));
  }
  loadBackups();
}

async function loadBackups() {
  const list = $("backup-list");
  try {
    const data = await api("/api/system/backups");
    list.replaceChildren();
    if (!data.backups.length) list.append(emptyItem("No backup yet."));
    for (const backup of data.backups.slice(0, 10)) {
      const result = el("div", { className: "sub" });
      const verify = el("button", { className: "small secondary", text: "Verify", attrs: { type: "button" } });
      verify.addEventListener("click", async () => {
        verify.disabled = true;
        try {
          const report = await api(`/api/system/backups/${encodeURIComponent(backup.name)}/verify`, { method: "POST" });
          result.textContent = report.ok ? `Verified: all ${report.files} files present, unchanged and sound.` : `NOT OK: ${report.problems.join("; ")}`;
        } catch (error) { result.textContent = `Could not verify: ${error.message}`; }
        verify.disabled = false;
      });
      list.append(el("li", {}, [
        el("div", { className: "row" }, [el("div", { className: "grow", text: backup.created_at ? when(backup.created_at) : backup.name }), verify]),
        el("div", { className: "sub", text: `${backup.files} files, ${megabytes(backup.size_bytes)}${backup.unreadable ? " - manifest unreadable" : ""}` }),
        result,
      ]));
    }
  } catch (error) {
    list.replaceChildren(emptyItem(`Backups could not be listed: ${error.message}`));
  }
}

function renderRemote(data) {
  const state = $("remote-state");
  const detail = $("remote-detail");
  const list = $("remote-devices");
  list.replaceChildren();
  if (!data.available) { state.textContent = "not available"; return; }
  state.textContent = data.enabled ? "on" : "off";
  state.className = `pill ${data.enabled ? "running" : "disabled"}`;
  detail.textContent = data.enabled
    ? `Reachable only at ${data.hosts.join(", ") || "(no host configured - nothing can reach it remotely)"} and only on /remote/. Sessions last ${data.session_hours} h. ${data.active_sessions} active session(s).`
    : "Off (the default). Nothing outside this computer can reach Jarvis. To turn it on, see README: Remote access.";
  $("remote-session-count").textContent = String(data.active_sessions);
  if (!data.devices.length) list.append(emptyItem("No devices registered."));
  for (const device of data.devices) {
    const actions = el("div", { className: "actions" });
    if (device.trust !== "revoked") {
      actions.append(actionButton("Revoke", `/api/remote/devices/${encodeURIComponent(device.device_id)}/revoke`,
        { style: "danger", confirm: `Revoke "${device.name}"? It will be disconnected now and can never connect again (register it anew to bring it back).` }));
    }
    const sessions = el("ul", { className: "goals" });
    for (const session of device.sessions) {
      const end = actionButton("End", `/api/remote/sessions/${encodeURIComponent(session.session_id)}/end`, { style: "secondary", confirm: "End this session now?" });
      sessions.append(el("li", {}, [el("div", { className: "row" }, [
        el("div", { className: "grow sub", text: `session ${session.session} - last used ${when(session.last_used)}, ${session.requests} requests, until ${when(session.expires_at)}` }), end])]));
    }
    list.append(el("li", {}, [
      el("div", { className: "row" }, [el("div", { className: "grow", text: device.name }), pill(device.trust), actions]),
      el("div", { className: "sub", text: `${device.type} - ${device.status}${device.last_seen ? " - last seen " + when(device.last_seen) : ""}` }),
      device.sessions.length ? sessions : null,
    ]));
  }
}

const TAB_VIEWS = {
  projects: ["/api/world", renderWorld],
  memory: ["/api/memory", renderMemory],
  automations: ["/api/automations", renderAutomations],
  calendar: ["/api/calendar", renderCalendar],
  system: ["/api/system", renderSystem],
  remote: ["/api/remote", renderRemote],
};
const TAB_KEY = "jarvis.tab";
let currentTab = "today";

async function loadTab(name, force = false) {
  const view = TAB_VIEWS[name];
  if (!view) return;
  const path = force && name === "calendar" ? `${view[0]}?refresh=true` : view[0];
  try { view[1](await api(path)); } catch (error) { toast(`Could not load ${name}: ${error.message}`, true); }
}

function showTab(name) {
  if (!document.querySelector(`.tab-panel[data-panel="${name}"]`)) name = "today";
  currentTab = name;
  writeStored(TAB_KEY, name);
  for (const tab of document.querySelectorAll(".tab")) {
    const active = tab.dataset.tab === name;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", active ? "true" : "false");
  }
  for (const panel of document.querySelectorAll(".tab-panel")) panel.hidden = panel.dataset.panel !== name;
  loadTab(name);
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
  if (currentTab !== "today" && currentTab !== "calendar") await loadTab(currentTab);  // the calendar refreshes on demand
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
  for (const tab of document.querySelectorAll(".tab")) tab.addEventListener("click", () => showTab(tab.dataset.tab));
  $("calendar-refresh").addEventListener("click", () => loadTab("calendar", true));
  $("backup-now").addEventListener("click", async () => {
    const button = $("backup-now");
    button.disabled = true;
    try {
      const made = await api("/api/system/backup", { method: "POST" });
      toast(`Backed up and verified: ${made.files} files (${megabytes(made.size_bytes)}).`);
    } catch (error) {
      toast(error.message, true);
    }
    button.disabled = false;
    loadTab("system");
  });
  showTab(readStored(TAB_KEY) || "today");

  tickClock();
  setInterval(tickClock, 30000);
  restoreConversation();
  refresh();
  setInterval(refresh, REFRESH_MS);
});
