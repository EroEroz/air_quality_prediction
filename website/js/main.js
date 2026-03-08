// AirPulse HCMC — Main Application Logic
// API_BASE is declared in map.js — reuse it
/* global API_BASE, renderForecastChart, updateHeatIntensity */

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

    // Last updated
    const now = new Date();
    document.getElementById("lastUpdated").textContent =
        now.toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit" });
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

    // Render chart
    if (data.forecast_24h && data.forecast_24h.length) {
        renderForecastChart(data.forecast_24h);
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
    const base = 24.5;
    const forecast_24h = [];
    const now = new Date();
    const nowH = now.getHours();

    for (let i = 0; i < 24; i++) {
        const h = (nowH + i) % 24;
        let mult = 1.0;
        if (h >= 7 && h <= 9) mult = 1.22;
        if (h >= 17 && h <= 19) mult = 1.18;
        if (h >= 2 && h <= 5) mult = 0.75;

        const pm25 = +(base * mult + (Math.random() * 4 - 2)).toFixed(1);
        forecast_24h.push({
            hour: h,
            label: `+${i}h`,
            pm25,
            category: pm25 < 12 ? "Good" : pm25 < 35 ? "Moderate" : "Poor",
        });
    }

    return {
        category: "Moderate",
        probabilities: { Good: 22.5, Moderate: 61.3, Poor: 16.2 },
        pm25_current: base,
        forecast_24h,
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
