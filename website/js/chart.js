/* ─── 24-Hour Average Forecast Panel ─────────────────────────────────────── */

function renderAvgForecast(avgPm25, category) {
    const section = document.getElementById("chartSection");
    const container = document.getElementById("avgForecastPanel");
    section.classList.remove("hidden");

    // Determine colors
    let color, bgColor, label;
    if (category === "Good" || avgPm25 < 12) {
        color = "#4ade80"; bgColor = "rgba(74,222,128,0.12)"; label = "Good";
    } else if (category === "Moderate" || avgPm25 < 35) {
        color = "#facc15"; bgColor = "rgba(250,204,21,0.12)"; label = "Moderate";
    } else {
        color = "#f87171"; bgColor = "rgba(248,113,113,0.12)"; label = "Poor";
    }

    // Gauge: max scale is 60 µg/m³ for visual clarity
    const SCALE_MAX = 60;
    const pctFill = Math.min(100, (avgPm25 / SCALE_MAX) * 100).toFixed(1);
    const pctGood = ((12 / SCALE_MAX) * 100).toFixed(1);      // 12 µg/m³ marker
    const pctMod = ((35 / SCALE_MAX) * 100).toFixed(1);      // 35 µg/m³ marker

    container.innerHTML = `
    <div class="avg-forecast-panel" style="--accent:${color}; --accent-bg:${bgColor};">
      <div class="avg-top">
        <div class="avg-metric">
          <span class="avg-label">Predicted 24h Average</span>
          <span class="avg-value">${avgPm25 !== null ? avgPm25.toFixed(1) : "—"}</span>
          <span class="avg-unit">µg/m³ PM2.5</span>
        </div>
        <div class="avg-badge" style="background:${bgColor}; color:${color}; border:1px solid ${color}40;">
          ${label}
        </div>
      </div>

      <!-- Gradient gauge bar -->
      <div class="gauge-wrap">
        <div class="gauge-track">
          <div class="gauge-fill" style="width:${pctFill}%; background:${color};"></div>
          <!-- Threshold tick: Good / Moderate -->
          <div class="gauge-tick" style="left:${pctGood}%;" title="12 µg/m³ — Good limit"></div>
          <div class="gauge-tick" style="left:${pctMod}%;"  title="35 µg/m³ — Moderate limit"></div>
        </div>
        <div class="gauge-labels">
          <span>0</span>
          <span style="left:${pctGood}%" class="gauge-threshold-label">12</span>
          <span style="left:${pctMod}%"  class="gauge-threshold-label">35</span>
          <span class="gauge-max">${SCALE_MAX}+</span>
        </div>
      </div>

      <p class="avg-note">
        Based on the model's <strong>target_24h_avg</strong> output —
        the expected mean PM2.5 concentration over the next 24 hours.
      </p>
    </div>
  `;

    // Animate fill after paint
    requestAnimationFrame(() => {
        const fill = container.querySelector(".gauge-fill");
        if (fill) {
            fill.style.transition = "width 1s cubic-bezier(0.34,1.1,0.64,1)";
        }
    });
}
