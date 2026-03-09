// AirPulse HCMC — Main Application Logic
// API_BASE is declared in map.js — reuse it
/* global API_BASE, renderAvgForecast, updateHeatIntensity */

// ── Current Status ────────────────────────────────────────────────────────────
async function loadCurrentStatus() {
    try {
        const res = await fetch(`${API_BASE}/api/current`);
        const json = await res.json();

        if (json.status === "ok" && json.data.length > 0) {
            // Average PM2.5 across all districts
            const avgPm = json.data.reduce((s, d) => s + d.pm25, 0) / json.data.length;
            updateCurrentStatus(avgPm);
        }
    } catch (_) {
        // Backend offline — use demo value
        updateCurrentStatus(22.5);
    }
}

function updateCurrentStatus(pm25) {
    const pm25El = document.getElementById("currentPm25");
    const statusEl = document.getElementById("currentStatus");

    pm25El.textContent = pm25.toFixed(1);

    let label, cls;
    if (pm25 < 12) {
        label = "Good"; cls = "status-good";
    } else if (pm25 < 35) {
        label = "Moderate"; cls = "status-moderate";
    } else {
        label = "Poor"; cls = "status-poor";
    }

    statusEl.textContent = label;
    statusEl.className = `metric-value status-badge ${cls}`;

    // Data period is now static in HTML ("Jan 2026"), no live timestamp needed
}

// ── Prediction ────────────────────────────────────────────────────────────────
async function runPrediction() {
    const btn = document.getElementById("predictBtn");
    const spinner = document.getElementById("btnSpinner");
    const icon = document.querySelector(".btn-icon");
    const text = document.querySelector(".btn-text");

    // Start loading state
    btn.disabled = true;
    spinner.classList.add("active");
    icon.classList.add("hidden");
    text.textContent = "Analyzing…";

    try {
        const res = await fetch(`${API_BASE}/api/predict`, { method: "POST" });
        const json = await res.json();

        if (json.status === "ok") {
            showResult(json.data);
        } else {
            throw new Error(json.message || "Unknown error");
        }
    } catch (err) {
        console.warn("Backend unavailable, using demo prediction:", err.message);
        // Demo fallback so UI still works without backend
        showResult(getDemoPrediction());
    } finally {
        btn.disabled = false;
        spinner.classList.remove("active");
        icon.classList.remove("hidden");
        text.textContent = "Predict Next 24 Hours";
    }
}

function showResult(data) {
    const card = document.getElementById("resultCard");
    const badge = document.getElementById("resultBadge");
    const pm25El = document.getElementById("resultPm25");

    // Result badge
    badge.textContent = data.category;
    badge.className = "result-badge";
    if (data.category === "Good") badge.classList.add("status-good");
    if (data.category === "Moderate") badge.classList.add("status-moderate");
    if (data.category === "Poor") badge.classList.add("status-poor");

    // Probability bars (animate after slight delay so slide-in finishes)
    card.classList.remove("hidden");
    setTimeout(() => {
        setBar("probGood", "pctGood", data.probabilities.Good);
        setBar("probMod", "pctMod", data.probabilities.Moderate);
        setBar("probPoor", "pctPoor", data.probabilities.Poor);
    }, 60);

    // PM2.5
    pm25El.textContent = data.pm25_current !== null ? `${data.pm25_current}` : "—";

    // Render avg forecast panel
    if (data.avg_pm25_24h !== undefined && data.avg_pm25_24h !== null) {
        renderAvgForecast(data.avg_pm25_24h, data.category);
        document.getElementById("chartSection").scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    // Refresh map heat
    if (typeof updateHeatIntensity === "function") {
        updateHeatIntensity(data.category === "Poor" ? 1.3 : 1.0);
    }
}

function setBar(fillId, pctId, value) {
    document.getElementById(fillId).style.width = `${Math.min(100, value)}%`;
    document.getElementById(pctId).textContent = `${value}%`;
}

// ── Demo Prediction (offline fallback) ───────────────────────────────────────
function getDemoPrediction() {
    return {
        category: "Moderate",
        probabilities: { Good: 22.5, Moderate: 61.3, Poor: 16.2 },
        pm25_current: 24.5,
        avg_pm25_24h: 24.5,
    };
}

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    loadCurrentStatus();
    // Refresh current status every 5 minutes
    setInterval(loadCurrentStatus, 5 * 60 * 1000);
});

// Expose to window for onclick attribute in HTML
window.runPrediction = runPrediction;

// ── Day-Period Prediction ─────────────────────────────────────────────────────
let _dpPeriod = "morning";

function setDpPeriod(btn) {
    document.querySelectorAll(".dp-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    _dpPeriod = btn.dataset.period;
}

async function runDayPeriodPrediction() {
    const dateVal = document.getElementById("dpDate").value;
    if (!dateVal) {
        alert("Please select a date first.");
        return;
    }

    const btn = document.getElementById("dpPredictBtn");
    const spinner = document.getElementById("dpSpinner");
    const textEl = btn.querySelector(".btn-text");

    btn.disabled = true;
    spinner.classList.add("active");
    textEl.textContent = "Predicting…";

    try {
        const res = await fetch(`${API_BASE}/api/day-period`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ date: dateVal, period: _dpPeriod }),
        });
        const json = await res.json();
        if (json.status === "ok") {
            renderDpResult(json.data);
        } else {
            throw new Error(json.message || "Unknown error");
        }
    } catch (err) {
        console.warn("Day-period prediction failed:", err.message);
        renderDpResult({
            date: dateVal, period: _dpPeriod,
            period_label: _dpPeriod + " (demo)",
            category: "Moderate",
            probabilities: { Good: 28.4, Moderate: 52.1, Poor: 19.5 },
        });
    } finally {
        btn.disabled = false;
        spinner.classList.remove("active");
        textEl.textContent = "Predict";
    }
}

function renderDpResult(data) {
    const container = document.getElementById("dpResult");
    container.classList.remove("hidden");

    let color, bgColor;
    if (data.category === "Good") {
        color = "#4ade80"; bgColor = "rgba(74,222,128,0.12)";
    } else if (data.category === "Moderate") {
        color = "#facc15"; bgColor = "rgba(250,204,21,0.12)";
    } else {
        color = "#f87171"; bgColor = "rgba(248,113,113,0.12)";
    }

    const fmt = new Intl.DateTimeFormat("en-GB", { day: "2-digit", month: "short", year: "numeric" });
    const dateLabel = fmt.format(new Date(data.date + "T12:00:00"));

    const p = data.probabilities;
    container.innerHTML = `
      <div class="dp-result-header">
        <div class="dp-meta">
          <span class="dp-meta-date">${dateLabel}</span>
          <span class="dp-meta-period">${data.period_label}</span>
        </div>
        <span class="dp-cat-badge" style="background:${bgColor};color:${color};border:1px solid ${color}40;">${data.category}</span>
      </div>
      <div class="dp-prob-bars">
        ${dpBar("Good", "#4ade80", p.Good)}
        ${dpBar("Moderate", "#facc15", p.Moderate)}
        ${dpBar("Poor", "#f87171", p.Poor)}
      </div>
    `;

    // Animate bars
    requestAnimationFrame(() => {
        container.querySelectorAll(".prob-fill").forEach(el => {
            el.style.transition = "width 0.9s cubic-bezier(0.34,1.1,0.64,1)";
        });
    });

    container.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function dpBar(label, color, value) {
    const cls = label === "Good" ? "good-fill" : label === "Moderate" ? "mod-fill" : "poor-fill";
    const textCls = label === "Good" ? "good-text" : label === "Moderate" ? "mod-text" : "poor-text";
    return `
      <div class="dp-prob-row">
        <span class="dp-prob-name ${textCls}">${label}</span>
        <div class="prob-track"><div class="prob-fill ${cls}" style="width:${Math.min(100, value)}%"></div></div>
        <span class="dp-prob-pct">${value}%</span>
      </div>`;
}

window.setDpPeriod = setDpPeriod;
window.runDayPeriodPrediction = runDayPeriodPrediction;
