const state = {
  models: {},
  history: JSON.parse(localStorage.getItem("qqpHistory") || "[]"),
};

const el = (id) => document.getElementById(id);

const formatScore = (value) => Number(value).toFixed(3);

async function getJson(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Request failed");
  }
  return data;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll("\"", "&quot;")
    .replaceAll("'", "&#039;");
}

function setBusy(isBusy) {
  el("analyzeBtn").disabled = isBusy;
  el("analyzeBtn").textContent = isBusy ? "Checking" : "Run check";
}

function fillModels(models) {
  state.models = models;
  el("modelSelect").innerHTML = "";
  Object.entries(models).forEach(([key, label]) => {
    el("modelSelect").add(new Option(label, key));
  });
  el("modelSelect").value = "model2";
}

function setView(view) {
  const isHistory = view === "history";
  el("compareView").hidden = isHistory;
  el("historyView").hidden = !isHistory;
  el("newBtn").classList.toggle("active", !isHistory);
  el("historyBtn").classList.toggle("active", isHistory);
  if (isHistory) {
    renderHistoryResults();
  }
}

function resetForm() {
  el("question1").value = "";
  el("question2").value = "";
  resetResult();
  setView("compare");
  el("question1").focus();
}

function resetResult() {
  el("resultCard").className = "result-card idle";
  el("verdictBadge").textContent = "Ready";
  el("scoreValue").textContent = "--";
  el("scoreMeta").textContent = "No score yet";
  el("scoreMeter").style.width = "0%";
  el("message").textContent = "Enter both fields, then run the check.";
}

function renderResult(result) {
  const score = Number(result.score);
  const labelClass = result.label === 1 ? "duplicate" : "not-duplicate";
  el("resultCard").className = `result-card ${labelClass}`;
  el("verdictBadge").textContent = result.verdict;
  el("scoreValue").textContent = formatScore(score);
  el("scoreMeta").textContent = `${result.model} - ${score >= 0.5 ? "+" : ""}${formatScore(score - 0.5)} from threshold`;
  el("scoreMeter").style.width = `${Math.max(0, Math.min(100, score * 100))}%`;
  el("message").textContent = result.label === 1
    ? "These questions are likely duplicates."
    : "These questions are likely not duplicates.";
}

function saveHistory(result, question1, question2) {
  state.history.unshift({
    id: Date.now(),
    verdict: result.verdict,
    label: result.label,
    score: result.score,
    model: result.model,
    question1,
    question2,
  });
  state.history = state.history.slice(0, 20);
  localStorage.setItem("qqpHistory", JSON.stringify(state.history));
  renderSidebarHistory();
}

function renderSidebarHistory() {
  if (!state.history.length) {
    el("historyList").textContent = "No saved checks yet.";
    return;
  }

  el("historyList").innerHTML = state.history.slice(0, 4).map((item) => `
    <button class="history-item" type="button" data-history-id="${item.id}">
      <strong>${escapeHtml(item.verdict)} - ${formatScore(item.score)}</strong>
      <span>${escapeHtml(item.question1)}</span>
    </button>
  `).join("");
}

function renderHistoryResults() {
  if (!state.history.length) {
    el("historyResults").textContent = "No saved checks yet.";
    return;
  }

  el("historyResults").innerHTML = state.history.map((item) => {
    const labelClass = item.label === 1 ? "duplicate" : "not-duplicate";
    return `
      <article class="history-result ${labelClass}">
        <h3>${escapeHtml(item.verdict)} - ${formatScore(item.score)}</h3>
        <dl>
          <div>
            <dt>Input 1</dt>
            <dd>${escapeHtml(item.question1)}</dd>
          </div>
          <div>
            <dt>Input 2</dt>
            <dd>${escapeHtml(item.question2)}</dd>
          </div>
          <div>
            <dt>Output</dt>
            <dd>${escapeHtml(item.verdict)} using ${escapeHtml(item.model)}</dd>
          </div>
        </dl>
      </article>
    `;
  }).join("");
}

function loadHistoryItem(id) {
  const item = state.history.find((entry) => String(entry.id) === String(id));
  if (!item) return;
  el("question1").value = item.question1;
  el("question2").value = item.question2;
  renderResult(item);
  setView("compare");
}

async function analyze(event) {
  event.preventDefault();

  const question1 = el("question1").value.trim();
  const question2 = el("question2").value.trim();
  if (!question1 || !question2) {
    el("message").textContent = "Both fields are required.";
    return;
  }

  setBusy(true);
  el("message").textContent = "Scoring the question pair.";

  try {
    const data = await getJson("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question1,
        question2,
        model: el("modelSelect").value,
        compare: false,
        include_sbert: false,
      }),
    });
    renderResult(data.result);
    saveHistory(data.result, question1, question2);
  } catch (error) {
    el("message").textContent = error.message;
  } finally {
    setBusy(false);
  }
}

function wireEvents() {
  el("compareForm").addEventListener("submit", analyze);
  el("newBtn").addEventListener("click", resetForm);
  el("historyBtn").addEventListener("click", () => setView("history"));
  el("clearBtn").addEventListener("click", resetForm);
  el("swapBtn").addEventListener("click", () => {
    const q1 = el("question1").value;
    el("question1").value = el("question2").value;
    el("question2").value = q1;
  });
  el("clearHistoryBtn").addEventListener("click", () => {
    state.history = [];
    localStorage.removeItem("qqpHistory");
    renderSidebarHistory();
    renderHistoryResults();
  });
  el("historyList").addEventListener("click", (event) => {
    const button = event.target.closest("[data-history-id]");
    if (button) {
      loadHistoryItem(button.dataset.historyId);
    }
  });
}

async function boot() {
  wireEvents();
  renderSidebarHistory();
  try {
    const data = await getJson("/api/summary");
    fillModels(data.models);
    el("serverState").textContent = "Ready";
    el("serverState").classList.add("ready");
  } catch (error) {
    el("serverState").textContent = "Error";
    el("message").textContent = error.message;
  }
}

boot();
