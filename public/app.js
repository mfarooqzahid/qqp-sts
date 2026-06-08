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
  el("analyzeBtn").textContent = isBusy ? "Checking..." : "Run check";
}

function fillModels(models) {
  state.models = models;
  el("modelSelect").innerHTML = "";
  Object.entries(models).forEach(([key, label]) => {
    el("modelSelect").add(new Option(label, key));
  });
  el("modelSelect").value = "model5"; // Default to recommended Ensemble model
}

function setView(view) {
  el("compareView").hidden = view !== "compare";
  el("historyView").hidden = view !== "history";
  el("metricsView").hidden = view !== "metrics";
  
  el("newBtn").classList.toggle("active", view === "compare");
  el("historyBtn").classList.toggle("active", view === "history");
  el("metricsBtn").classList.toggle("active", view === "metrics");
  
  if (view === "history") {
    renderHistoryResults();
  } else if (view === "metrics") {
    loadMetrics();
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
  
  el("comparisonPanelSection").hidden = true;
  el("comparisonGrid").innerHTML = "";
}

function renderResult(result) {
  const score = Number(result.score);
  const labelClass = result.label === 1 ? "duplicate" : "not-duplicate";
  el("resultCard").className = `result-card ${labelClass}`;
  el("verdictBadge").textContent = result.verdict;
  el("scoreValue").textContent = formatScore(score);
  
  const diffStr = score >= result.threshold ? "+" : "";
  el("scoreMeta").textContent = `${result.model} - ${diffStr}${formatScore(score - result.threshold)} from threshold (${formatScore(result.threshold)})`;
  el("scoreMeter").style.width = `${Math.max(0, Math.min(100, score * 100))}%`;
  
  el("message").textContent = result.label === 1
    ? "These questions are likely duplicates."
    : "These questions are likely not duplicates.";
}

function renderComparison(comparison) {
  if (!comparison || comparison.length === 0) {
    el("comparisonPanelSection").hidden = true;
    return;
  }
  
  el("comparisonPanelSection").hidden = false;
  el("comparisonGrid").innerHTML = comparison.map(item => {
    const isRecommended = item.model_key === 'model5';
    const labelClass = item.label === 1 ? 'duplicate' : 'not-duplicate';
    const recommendedClass = isRecommended ? 'recommended' : '';
    const scorePct = Math.max(0, Math.min(100, item.score * 100));
    
    return `
      <div class="comp-card ${labelClass} ${recommendedClass}">
        <div>
          <div class="comp-model-name">${escapeHtml(item.model)} ${isRecommended ? '★' : ''}</div>
          <div class="comp-verdict">${escapeHtml(item.verdict)}</div>
        </div>
        <div>
          <div class="comp-score-row">
            <span>Confidence</span>
            <span class="comp-score">${formatScore(item.score)}</span>
          </div>
          <div class="comp-meter"><span style="width: ${scorePct}%"></span></div>
        </div>
      </div>
    `;
  }).join("");
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

  el("historyList").innerHTML = state.history.slice(0, 4).map((item) => {
    const labelClass = item.label === 1 ? "duplicate" : "not-duplicate";
    return `
      <button class="history-item ${labelClass}" type="button" data-history-id="${item.id}">
        <strong>${escapeHtml(item.verdict)} - ${formatScore(item.score)}</strong>
        <span>${escapeHtml(item.question1)}</span>
      </button>
    `;
  }).join("");
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
            <dt>First Question</dt>
            <dd>${escapeHtml(item.question1)}</dd>
          </div>
          <div>
            <dt>Second Question</dt>
            <dd>${escapeHtml(item.question2)}</dd>
          </div>
          <div>
            <dt>Prediction Details</dt>
            <dd>${escapeHtml(item.verdict)} classified using ${escapeHtml(item.model)}</dd>
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
  el("comparisonPanelSection").hidden = true; // Clear multi-model comparisons for historical items
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
  el("message").textContent = "Scoring the question pair using all models...";

  try {
    const data = await getJson("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question1,
        question2,
        model: el("modelSelect").value,
        compare: true,
        include_sbert: true,
      }),
    });
    
    renderResult(data.result);
    renderComparison(data.comparison);
    saveHistory(data.result, question1, question2);
  } catch (error) {
    el("message").textContent = error.message;
  } finally {
    setBusy(false);
  }
}

async function loadMetrics() {
  const tableBody = el("metricsTableBody");
  tableBody.innerHTML = `<tr><td colspan="6" class="table-loading">Loading metrics...</td></tr>`;
  
  // Update plot image
  updatePlot();

  try {
    const data = await getJson("/api/metrics");
    const modelDisplayNames = {
      'model1': 'Model 1: TF-IDF Cosine (Calibrated)',
      'model2': 'Model 2: TF-IDF + Logistic Regression',
      'model3': 'Model 3: SBERT Cosine (Calibrated)',
      'model4': 'Model 4: SBERT Supervised Classifier',
      'model5': 'Model 5: Lexical+Semantic Ensemble'
    };

    tableBody.innerHTML = data.metrics.map(row => {
      const modelName = modelDisplayNames[row.ModelKey] || row.ModelKey;
      const isBest = row.ModelKey === 'model5';
      const rowClass = isBest ? 'class="best-row"' : '';
      
      return `
        <tr ${rowClass}>
          <td>${escapeHtml(modelName)} ${isBest ? '★' : ''}</td>
          <td>${formatScore(row.Accuracy)}</td>
          <td>${formatScore(row.Precision)}</td>
          <td>${formatScore(row.Recall)}</td>
          <td>${formatScore(row.F1)}</td>
          <td>${formatScore(row["ROC-AUC"])}</td>
        </tr>
      `;
    }).join("");
  } catch (error) {
    tableBody.innerHTML = `<tr><td colspan="6" class="table-loading" style="color: var(--danger)">${escapeHtml(error.message)}</td></tr>`;
  }
}

function updatePlot() {
  const filename = el("plotSelect").value;
  const plotImg = el("evaluationPlot");
  const plotFallback = el("plotFallback");
  
  plotImg.hidden = false;
  plotFallback.hidden = true;
  
  plotImg.src = `/outputs/${filename}`;
  plotImg.onerror = () => {
    plotImg.hidden = true;
    plotFallback.hidden = false;
  };
}

function wireEvents() {
  el("compareForm").addEventListener("submit", analyze);
  el("newBtn").addEventListener("click", resetForm);
  el("historyBtn").addEventListener("click", () => setView("history"));
  el("metricsBtn").addEventListener("click", () => setView("metrics"));
  
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

  el("plotSelect").addEventListener("change", updatePlot);

  // Quick preset click handler
  document.querySelectorAll(".preset-btn").forEach(btn => {
    btn.addEventListener("click", (e) => {
      resetResult();
      el("question1").value = e.target.dataset.q1;
      el("question2").value = e.target.dataset.q2;
      el("question1").focus();
    });
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
