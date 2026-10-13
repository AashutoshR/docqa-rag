const $ = (id) => document.getElementById(id);

const ACTIVE = new Set(["queued", "reading", "indexing"]);
const KIND_LABELS = { book: "Book", article: "Article", paper: "Research paper", document: "Document" };
const CITATION = /\[(\d+)(?:†[^\]]*)?\]/g;

const state = {
  documents: [],
  jobs: [],
  excluded: new Set(), // names of documents unticked in the library
  history: [],
  busy: false,
  chats: [], // saved conversations, kept in this browser only
  chatId: null, // the saved chat on screen, or null for a new one
};
let pollTimer;

// Values kept in this browser. Storage can be blocked, so fall back to memory.
const memory = {};
function stored(key, value) {
  try {
    if (value !== undefined) localStorage.setItem(key, value);
    return localStorage.getItem(key);
  } catch {
    if (value !== undefined) memory[key] = value;
    return memory[key] || null;
  }
}

// Each browser gets a random ID, and the server keeps a separate library for each one.
function workspaceId() {
  let id = stored("docqa-workspace");
  if (!id) {
    const bytes = crypto.getRandomValues(new Uint8Array(16));
    id = [...bytes].map((byte) => byte.toString(16).padStart(2, "0")).join("");
    stored("docqa-workspace", id);
  }
  return id;
}

function identity() {
  return { "X-Workspace": workspaceId(), "X-Access-Code": stored("docqa-access-code") || "" };
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function icon(name) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", "icon");
  const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
  use.setAttribute("href", `#i-${name}`);
  svg.append(use);
  return svg;
}

// "Alan_Beaulieu-Learning_SQL-EN.pdf" -> "Alan Beaulieu-Learning SQL-EN"
function displayName(fileName) {
  return fileName
    .replace(/\.[a-z0-9]+$/i, "")
    .replace(/\s*\([^)]*\.(com|org|net)[^)]*\)/gi, "")
    .replace(/[_.]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

async function api(path, options = {}) {
  const response = await fetch(path, { ...options, headers: { ...identity(), ...options.headers } });
  if (response.status === 401) {
    askForAccessCode();
    throw new Error("Enter the access code to continue.");
  }
  if (!response.ok) {
    const detail = await response.json().then((body) => body.detail, () => null);
    throw new Error(typeof detail === "string" ? detail : `Request failed (${response.status})`);
  }
  return response.json();
}

const notices = {};
function notify(topic, message) {
  notices[topic] = message;
  const current = Object.values(notices).find(Boolean) || "";
  $("banner").textContent = current;
  $("banner").hidden = !current;
}

/* Library */

async function refresh() {
  clearTimeout(pollTimer);
  try {
    const data = await api("/api/library");
    state.documents = data.documents;
    state.jobs = data.jobs;
    notify("server", "");
  } catch {
    if (!$("gate").open) notify("server", "Can't reach the DocQA server. Is it still running?");
  }
  render();
  if (state.jobs.some((job) => ACTIVE.has(job.status))) pollTimer = setTimeout(refresh, 1000);
}

function includedDocuments() {
  return state.documents.filter((doc) => !state.excluded.has(doc.name)).map((doc) => doc.name);
}

function render() {
  const onboarding = state.documents.length === 0;
  $("welcome").hidden = !onboarding;
  $("workspace").hidden = onboarding;

  const home = $(onboarding ? "welcome-uploader" : "dialog-uploader");
  if ($("uploader").parentElement !== home) home.append($("uploader"));
  if (onboarding && $("add-dialog").open) $("add-dialog").close();

  $(onboarding ? "welcome-jobs" : "jobs").replaceChildren(...state.jobs.map(jobRow));
  $(onboarding ? "jobs" : "welcome-jobs").replaceChildren();
  $("documents").replaceChildren(...state.documents.map(documentRow));

  const total = state.documents.length;
  const included = includedDocuments().length;
  $("scope").textContent =
    included === 0
      ? "Tick at least one document to ask a question."
      : included === total
        ? `Searching ${total === 1 ? "1 document" : `all ${total} documents`}`
        : `Searching ${included} of ${total} documents`;
  renderSuggestions();
  updateComposer();
}

// Up to four example questions, taken in turn from each ticked document.
function renderSuggestions() {
  const lists = state.documents
    .filter((doc) => !state.excluded.has(doc.name))
    .map((doc) => doc.questions || []);
  const picked = [];
  for (let round = 0; picked.length < 4 && lists.some((list) => list.length > round); round++) {
    for (const list of lists) {
      if (list.length > round && picked.length < 4) picked.push(list[round]);
    }
  }
  const chips = picked.map((question) => {
    const chip = el("button", "suggestion", question);
    chip.type = "button";
    chip.onclick = () => {
      if (!state.busy) ask(question);
    };
    return chip;
  });
  $("suggestions").replaceChildren($("suggestions").firstElementChild, ...chips);
  $("suggestions").hidden = chips.length === 0;
}

function jobRow(job) {
  const row = el("li", job.status === "error" ? "job error" : "job");
  const top = el("div", "job-top");
  top.append(el("span", "job-name", displayName(job.name)));
  row.append(top);

  if (job.status === "error") {
    const dismiss = el("button", "icon-button");
    dismiss.type = "button";
    dismiss.setAttribute("aria-label", `Dismiss ${displayName(job.name)}`);
    dismiss.append(icon("close"));
    dismiss.onclick = () => api(`/api/jobs/${job.id}`, { method: "DELETE" }).then(refresh);
    top.append(dismiss);
    row.append(el("div", "job-status", job.message));
    return row;
  }

  const progress = el("progress");
  let status = "Waiting for the previous upload…";
  if (job.status === "reading") status = "Reading pages…";
  if (job.status === "indexing") {
    status = `Indexing ${job.done.toLocaleString()} of ${job.total.toLocaleString()} passages`;
    progress.max = job.total;
    progress.value = job.done;
  }
  row.append(el("div", "job-status", status), progress);
  return row;
}

function documentRow(doc) {
  const row = el("li", "document");
  const label = el("label");
  const checkbox = el("input");
  checkbox.type = "checkbox";
  checkbox.checked = !state.excluded.has(doc.name);
  checkbox.onchange = () => {
    if (checkbox.checked) state.excluded.delete(doc.name);
    else state.excluded.add(doc.name);
    render();
  };

  const text = el("span");
  const isPdf = /\.pdf$/i.test(doc.name);
  const isWord = /[.]docx?$/i.test(doc.name);
  const size = isPdf
    ? `${doc.pages.toLocaleString()} ${doc.pages === 1 ? "page" : "pages"}`
    : isWord ? "Word document" : "Text file";
  const title = el("div", "document-title", displayName(doc.name));
  title.title = doc.name;
  text.append(
    title,
    el("div", "document-meta", `${KIND_LABELS[doc.kind] || KIND_LABELS.document} · ${size}`),
  );
  label.append(checkbox, text);

  const remove = el("button", "icon-button");
  remove.type = "button";
  remove.setAttribute("aria-label", `Remove ${displayName(doc.name)}`);
  remove.append(icon("close"));
  remove.onclick = async () => {
    if (!confirm(`Remove "${displayName(doc.name)}" from your library?`)) return;
    remove.disabled = true;
    try {
      await api(`/api/library/${encodeURIComponent(doc.name)}`, { method: "DELETE" });
      state.excluded.delete(doc.name);
      notify("remove", "");
    } catch (error) {
      notify("remove", error.message);
    }
    refresh();
  };

  row.append(label, remove);
  return row;
}

/* Uploading */

async function uploadFiles(files) {
  const kind = document.querySelector("input[name=kind]:checked").value;
  const errors = [];
  for (const file of files) {
    const body = new FormData();
    body.append("file", file);
    body.append("kind", kind);
    try {
      await api("/api/upload", { method: "POST", body });
    } catch (error) {
      errors.push(error.message);
    }
  }
  $("files").value = "";
  $("upload-error").textContent = errors.join(" ");
  $("upload-error").hidden = errors.length === 0;
  if (errors.length === 0 && $("add-dialog").open) $("add-dialog").close();
  refresh();
}

function setUpUploader() {
  const dropzone = $("dropzone");
  $("files").onchange = () => uploadFiles([...$("files").files]);
  dropzone.ondragover = (event) => {
    event.preventDefault();
    dropzone.classList.add("over");
  };
  dropzone.ondragleave = () => dropzone.classList.remove("over");
  dropzone.ondrop = (event) => {
    event.preventDefault();
    dropzone.classList.remove("over");
    uploadFiles([...event.dataTransfer.files]);
  };
  // A file dropped outside the dropzone would otherwise replace the page.
  window.addEventListener("dragover", (event) => event.preventDefault());
  window.addEventListener("drop", (event) => event.preventDefault());

  $("add").onclick = () => $("add-dialog").showModal();
  $("add-close").onclick = () => $("add-dialog").close();
  $("add-dialog").onclick = (event) => {
    if (event.target === $("add-dialog")) $("add-dialog").close();
  };
}

/* Asking */

function updateComposer() {
  const ready = !state.busy && includedDocuments().length > 0 && $("question").value.trim() !== "";
  $("send").disabled = !ready;
  $("new-chat").disabled = state.busy || !document.querySelector(".turn");
}

function newChat() {
  document.querySelectorAll(".turn").forEach((turn) => turn.remove());
  state.history = [];
  state.chatId = null;
  $("empty").hidden = false;
  renderChats();
  updateComposer();
  $("question").focus();
}

/* Saved chats: stored in this browser, never on the server */

const CHATS_KEY = "docqa-chats";
const MAX_CHATS = 50;

function loadChats() {
  try {
    const chats = JSON.parse(stored(CHATS_KEY) || "[]");
    return Array.isArray(chats) ? chats.filter((chat) => chat?.id && Array.isArray(chat.turns)) : [];
  } catch {
    return [];
  }
}

function saveChats() {
  state.chats.sort((a, b) => b.updated - a.updated);
  state.chats.splice(MAX_CHATS);
  for (;;) {
    try {
      localStorage.setItem(CHATS_KEY, JSON.stringify(state.chats));
      break;
    } catch (error) {
      // When storage is full, let go of the oldest chat and try again. If storage
      // is unavailable altogether, the chats last until the page is closed.
      const full = /quota/i.test(error?.name || "");
      const oldest = state.chats.findLast((chat) => chat.id !== state.chatId);
      if (!full || !oldest) break;
      state.chats.splice(state.chats.indexOf(oldest), 1);
    }
  }
  renderChats();
}

// Add a finished question and answer to the chat on screen, creating it if new.
function recordTurn(turn) {
  let chat = state.chats.find((saved) => saved.id === state.chatId);
  if (!chat) {
    const id = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
    chat = { id, title: turn.question.slice(0, 120), turns: [] };
    state.chats.push(chat);
    state.chatId = id;
  }
  chat.turns.push(turn);
  chat.updated = Date.now();
  saveChats();
}

function openChat(id) {
  const chat = state.chats.find((saved) => saved.id === id);
  if (!chat || state.busy) return;
  document.querySelectorAll(".turn").forEach((turn) => turn.remove());
  state.chatId = id;
  state.history = chat.turns.flatMap((turn) => [
    { role: "user", content: turn.question },
    { role: "assistant", content: turn.text },
  ]);
  $("empty").hidden = true;
  $("thread").append(...chat.turns.map(turnElement));
  $("thread").scrollTop = $("thread").scrollHeight;
  $("sidebar").classList.remove("open");
  renderChats();
  updateComposer();
}

function deleteChat(id) {
  state.chats = state.chats.filter((saved) => saved.id !== id);
  if (id === state.chatId) newChat();
  saveChats();
}

function renderChats() {
  $("chats").replaceChildren(...state.chats.map(chatRow));
  $("chats-note").textContent = state.chats.length
    ? "Saved in this browser only, not on the server."
    : "Your chats will be saved here, in this browser only.";
}

function chatRow(chat) {
  const row = el("li", chat.id === state.chatId ? "chat-row current" : "chat-row");
  const open = el("button", "chat-open");
  open.type = "button";
  open.title = chat.title;
  const date = new Date(chat.updated).toLocaleDateString(undefined, { day: "numeric", month: "short" });
  const count = chat.turns.length;
  open.append(
    el("span", "chat-title", chat.title),
    el("span", "chat-date", `${date} · ${count} ${count === 1 ? "question" : "questions"}`),
  );
  open.onclick = () => openChat(chat.id);

  const remove = el("button", "icon-button");
  remove.type = "button";
  remove.setAttribute("aria-label", `Delete saved chat: ${chat.title}`);
  remove.append(icon("close"));
  remove.onclick = () => {
    if (state.busy || !confirm(`Delete this saved chat?\n\n"${chat.title}"`)) return;
    deleteChat(chat.id);
  };

  row.append(open, remove);
  return row;
}

// A saved question and answer, drawn the way it looked when it was first given.
function turnElement(saved) {
  const turn = el("article", "turn");
  const searched = el("div", "searched", saved.searched || "");
  searched.hidden = !saved.searched;
  const answer = el("div", "answer");
  answer.innerHTML = toHtml(saved.text);
  linkCitations(answer, saved.sources.length);
  turn.append(el("div", "asked", saved.question), searched, answer);
  if (saved.found && saved.sources.length) turn.append(sourcesPanel(saved.sources, saved.text));
  return turn;
}

function toHtml(markdown) {
  if (window.marked && window.DOMPurify) return DOMPurify.sanitize(marked.parse(markdown));
  const plain = el("div", "", markdown);
  plain.style.whiteSpace = "pre-wrap";
  return plain.outerHTML;
}

// Turn "[2]" in the answer into buttons that jump to source 2.
function linkCitations(root, sourceCount) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  const textNodes = [];
  while (walker.nextNode()) {
    if (!walker.currentNode.parentElement.closest("code, pre")) textNodes.push(walker.currentNode);
  }
  for (const node of textNodes) {
    const parts = node.textContent.split(CITATION);
    if (parts.length === 1) continue;
    const fragment = document.createDocumentFragment();
    parts.forEach((part, i) => {
      if (i % 2 === 0) return fragment.append(part);
      const number = Number(part);
      if (number < 1 || number > sourceCount) return fragment.append(`[${part}]`);
      const chip = el("button", "cite", part);
      chip.type = "button";
      chip.dataset.source = part;
      chip.setAttribute("aria-label", `Source ${part}`);
      fragment.append(chip);
    });
    node.replaceWith(fragment);
  }
}

function citedNumbers(text, sourceCount) {
  const numbers = [...text.matchAll(CITATION)].map((match) => Number(match[1]));
  return new Set(numbers.filter((n) => n >= 1 && n <= sourceCount));
}

function sourcesPanel(sources, text) {
  const cited = citedNumbers(text, sources.length);
  const showAll = cited.size === 0;
  const panel = el("details", "sources");
  panel.append(el("summary", "", showAll ? `Passages consulted (${sources.length})` : `Sources (${cited.size})`));

  const list = el("div", "source-list");
  sources.forEach((source, i) => {
    const card = el("div", "source");
    card.dataset.source = i + 1;
    card.hidden = !showAll && !cited.has(i + 1);
    const title = el("div", "source-title", displayName(source.source));
    const lastPage = source.page_end || source.page;
    const pages = lastPage > source.page ? `pages ${source.page}-${lastPage}` : `page ${source.page}`;
    // Only PDFs have real page numbers; Word and text files are one long page.
    if (/[.]pdf$/i.test(source.source)) title.append(el("span", "", ` · ${pages}`));
    const passage = el("div", "source-text", source.text);
    passage.onclick = () => card.classList.toggle("open");
    card.append(el("span", "source-number", String(i + 1)), title, passage);
    list.append(card);
  });

  const hiddenCount = sources.length - cited.size;
  if (!showAll && hiddenCount > 0) {
    const more = el("button", "more", `Show ${hiddenCount} more retrieved ${hiddenCount === 1 ? "passage" : "passages"}`);
    more.type = "button";
    more.onclick = () => {
      list.querySelectorAll(".source").forEach((card) => (card.hidden = false));
      more.remove();
    };
    list.append(more);
  }
  panel.append(list);
  return panel;
}

async function* readEvents(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop();
    for (const line of lines) if (line.trim()) yield JSON.parse(line);
  }
}

async function ask(question) {
  state.busy = true;
  updateComposer();
  $("empty").hidden = true;

  const thread = $("thread");
  const turn = el("article", "turn");
  const searched = el("div", "searched");
  searched.hidden = true;
  const answer = el("div", "answer pending", "Searching your library…");
  turn.append(el("div", "asked", question), searched, answer);
  thread.append(turn);
  thread.scrollTop = thread.scrollHeight;

  const included = includedDocuments();
  let sources = [];
  let text = "";
  let frame = 0;
  const paint = () => {
    const pinned = thread.scrollHeight - thread.scrollTop - thread.clientHeight < 80;
    answer.className = "answer";
    answer.innerHTML = toHtml(text);
    linkCitations(answer, sources.length);
    if (pinned) thread.scrollTop = thread.scrollHeight;
  };
  const fail = (message) => {
    cancelAnimationFrame(frame);
    answer.className = "answer error";
    answer.textContent = message;
  };

  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json", ...identity() },
      body: JSON.stringify({
        question,
        history: state.history.slice(-6),
        documents: included.length === state.documents.length ? null : included,
      }),
    });
    if (response.status === 401) {
      askForAccessCode();
      throw new Error("Enter the access code to continue.");
    }
    if (!response.ok) throw new Error(`Request failed (${response.status})`);

    for await (const event of readEvents(response)) {
      if (event.type === "sources") {
        sources = event.sources;
        const same = (a, b) => a.trim().toLowerCase() === b.trim().toLowerCase();
        if (event.searches.length > 1 || !same(event.searches[0] || question, question)) {
          searched.textContent = `Searched for: ${event.searches.join(" · ")}`;
          searched.hidden = false;
        }
      } else if (event.type === "token") {
        text += event.text;
        cancelAnimationFrame(frame);
        frame = requestAnimationFrame(paint);
      } else if (event.type === "done") {
        cancelAnimationFrame(frame);
        paint();
        if (event.found && sources.length) turn.append(sourcesPanel(sources, text));
        state.history.push({ role: "user", content: question }, { role: "assistant", content: text });
        recordTurn({
          question,
          searched: searched.hidden ? "" : searched.textContent,
          text,
          sources: event.found ? sources : [],
          found: event.found,
        });
      } else if (event.type === "error") {
        fail(event.message);
      }
    }
  } catch (error) {
    fail(error.message || "Something went wrong.");
  }

  state.busy = false;
  updateComposer();
  $("question").focus();
}

function setUpChat() {
  const question = $("question");
  const grow = () => {
    question.style.height = "auto";
    question.style.height = `${question.scrollHeight}px`;
    updateComposer();
  };
  question.oninput = grow;
  question.onkeydown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      $("composer").requestSubmit();
    }
  };
  $("composer").onsubmit = (event) => {
    event.preventDefault();
    const text = question.value.trim();
    if ($("send").disabled || !text) return;
    question.value = "";
    grow();
    ask(text);
  };

  // Citation chips open the sources panel and point at the passage.
  $("thread").addEventListener("click", (event) => {
    const chip = event.target.closest(".cite");
    if (!chip) return;
    const panel = chip.closest(".turn").querySelector(".sources");
    const card = panel?.querySelector(`.source[data-source="${chip.dataset.source}"]`);
    if (!card) return;
    panel.open = true;
    card.hidden = false;
    card.scrollIntoView({ behavior: "smooth", block: "nearest" });
    card.classList.add("flash");
    setTimeout(() => card.classList.remove("flash"), 1400);
  });

  $("new-chat").onclick = newChat;
  state.chats = loadChats();
  renderChats();
  // Keep the list current when another tab of this app saves or deletes a chat.
  window.addEventListener("storage", (event) => {
    if (event.key !== CHATS_KEY) return;
    state.chats = loadChats();
    renderChats();
  });
  $("menu").onclick = (event) => {
    event.stopPropagation();
    $("sidebar").classList.toggle("open");
  };
  document.querySelector(".chat").addEventListener("click", () => $("sidebar").classList.remove("open"));
}

/* Access code: only asked for when the server is configured with one */

function askForAccessCode() {
  if (!$("gate").open) $("gate").showModal();
  $("gate-code").focus();
}

function setUpGate() {
  $("gate").addEventListener("cancel", (event) => event.preventDefault());
  $("gate-form").onsubmit = async (event) => {
    event.preventDefault();
    stored("docqa-access-code", $("gate-code").value.trim());
    const check = await fetch("/api/library", { headers: identity() }).catch(() => null);
    const rejected = !check || check.status === 401;
    $("gate-error").hidden = !rejected;
    if (rejected) return;
    $("gate").close();
    $("gate-code").value = "";
    refresh();
  };
}

async function start() {
  let status = null;
  try {
    status = await fetch("/api/status").then((response) => response.json());
  } catch {
    /* refresh() reports an unreachable server */
  }
  if (status) {
    if (status.missing_keys.length) {
      notify("keys", `Add ${status.missing_keys.join(" and ")} to the server's settings, then restart it.`);
    }
    $("models").textContent = `Answers by ${status.chat_model} · search by ${status.embedding_model}`;
    $("upload-limit").textContent = `PDF, Word, TXT or Markdown, up to ${status.max_upload_mb} MB each`;
  }
  if (status?.access_code_required && !stored("docqa-access-code")) askForAccessCode();
  else refresh();
}

setUpUploader();
setUpChat();
setUpGate();
start();
