/* global Chart */

let forecastChartInstance = null;

function renderForecastChart(forecast24h) {
    const section = document.getElementById("chartSection");
    section.classList.remove("hidden");

    const labels = forecast24h.map((f) => f.label);
    const values = forecast24h.map((f) => f.pm25);
    const bgColors = forecast24h.map((f) => {
        if (f.category === "Good") return "rgba(74,222,128,0.75)";
        if (f.category === "Moderate") return "rgba(250,204,21,0.75)";
        return "rgba(248,113,113,0.75)";
    });
    const borderColors = forecast24h.map((f) => {
        if (f.category === "Good") return "#4ade80";
        if (f.category === "Moderate") return "#facc15";
        return "#f87171";
    });

    const ctx = document.getElementById("forecastChart").getContext("2d");

    // Destroy old chart if exists
    if (forecastChartInstance) {
        forecastChartInstance.destroy();
    }

    forecastChartInstance = new Chart(ctx, {
        type: "bar",
        data: {
            labels,
            datasets: [
                {
                    label: "PM2.5 (µg/m³)",
                    data: values,
                    backgroundColor: bgColors,
                    borderColor: borderColors,
                    borderWidth: 1.5,
                    borderRadius: 6,
                    borderSkipped: false,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 900, easing: "easeOutQuart" },
            interaction: { intersect: false, mode: "index" },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: "hsl(220,22%,12%)",
                    borderColor: "hsla(220,30%,55%,0.2)",
                    borderWidth: 1,
                    titleColor: "#f1f5f9",
                    bodyColor: "#94a3b8",
                    padding: 12,
                    cornerRadius: 10,
                    callbacks: {
                        title: (items) => `Forecast ${items[0].label}`,
                        label: (item) => ` PM2.5: ${item.raw} µg/m³`,
                        afterLabel: (item) => {
                            const d = forecast24h[item.dataIndex];
                            return ` Category: ${d.category}`;
                        },
                    },
                },
            },
            scales: {
                x: {
                    grid: { color: "hsla(220,30%,55%,0.08)" },
                    ticks: {
                        color: "#64748b",
                        font: { family: "Inter", size: 11 },
                        maxRotation: 45,
                    },
                },
                y: {
                    grid: { color: "hsla(220,30%,55%,0.08)" },
                    ticks: {
                        color: "#64748b",
                        font: { family: "Inter", size: 11 },
                        callback: (v) => `${v} µg/m³`,
                    },
                    beginAtZero: true,
                },
            },
        },
    });

    // Reference lines
    const thresholdPlugin = {
        id: "thresholds",
        afterDraw(chart) {
            const { ctx: c, chartArea: { left, right }, scales: { y } } = chart;
            [[12, "#4ade80"], [35, "#f87171"]].forEach(([val, color]) => {
                const yPos = y.getPixelForValue(val);
                c.save();
                c.beginPath();
                c.setLineDash([5, 4]);
                c.strokeStyle = color + "66";
                c.lineWidth = 1.5;
                c.moveTo(left, yPos);
                c.lineTo(right, yPos);
                c.stroke();
                c.restore();
            });
        },
    };

    forecastChartInstance.config.plugins = [thresholdPlugin];
    forecastChartInstance.update();
}
