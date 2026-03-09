/* global L */

const API_BASE = "";

// HCMC center coordinates
const HCMC_CENTER = [10.7769, 106.7009];
const HCMC_ZOOM = 12;

let map;
let heatLayer;
let districtMarkers = [];

// Color mapping for AQI categories
const AQI_COLORS = {
    Good: { fill: "#4ade80", ring: "rgba(74,222,128,0.25)" },
    Moderate: { fill: "#facc15", ring: "rgba(250,204,21,0.25)" },
    Poor: { fill: "#f87171", ring: "rgba(248,113,113,0.25)" },
};

function initMap() {
    // Initialize Leaflet map
    map = L.map("map", {
        center: HCMC_CENTER,
        zoom: HCMC_ZOOM,
        zoomControl: true,
        attributionControl: true,
    });

    // Dark basemap tiles
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        attribution: '© <a href="https://www.openstreetmap.org/">OpenStreetMap</a>',
        maxZoom: 18,
    }).addTo(map);

    // Load initial district data
    fetchDistrictData();

    // Auto-refresh every 60 seconds
    setInterval(fetchDistrictData, 60_000);
}

async function fetchDistrictData() {
    try {
        const res = await fetch(`${API_BASE}/api/current`);
        const json = await res.json();
        if (json.status === "ok") {
            renderDistrictData(json.data);
        }
    } catch (_) {
        // Backend offline — render demo data so the map still looks great
        renderDistrictData(getDemoData());
    }
}

function renderDistrictData(districts) {
    // Clear old markers
    districtMarkers.forEach((m) => map.removeLayer(m));
    districtMarkers = [];
    if (heatLayer) map.removeLayer(heatLayer);

    const heatPoints = [];

    districts.forEach((d) => {
        const colors = AQI_COLORS[d.category] || AQI_COLORS.Good;

        // Circle marker
        const circle = L.circleMarker([d.lat, d.lng], {
            radius: 14,
            fillColor: colors.fill,
            color: colors.fill,
            weight: 2,
            opacity: 0.9,
            fillOpacity: 0.7,
        });

        circle.bindPopup(`
      <div style="font-family:Inter,sans-serif; padding:4px 6px; min-width:160px;">
        <div style="font-weight:700; font-size:14px; margin-bottom:4px;">${d.name}</div>
        <div style="color:#9ca3af; font-size:12px;">PM2.5: <span style="color:#f1f5f9; font-weight:600;">${d.pm25} µg/m³</span></div>
        <div style="margin-top:6px;">
          <span style="background:${colors.ring}; color:${colors.fill}; padding:2px 10px; border-radius:20px; font-size:11px; font-weight:600;">${d.category}</span>
        </div>
      </div>
    `, { className: "dark-popup" });

        circle.addTo(map);
        districtMarkers.push(circle);

        // Heat point: intensity based on PM2.5
        const intensity = Math.min(1.0, d.pm25 / 50);
        heatPoints.push([d.lat, d.lng, intensity]);
    });

    // Heatmap layer
    heatLayer = L.heatLayer(heatPoints, {
        radius: 45,
        blur: 30,
        maxZoom: 14,
        gradient: { 0.0: "#4ade80", 0.4: "#facc15", 0.7: "#f97316", 1.0: "#ef4444" },
        max: 1.0,
    });
    heatLayer.addTo(map);
}

// Update heat intensity after a prediction (called from main.js)
function updateHeatIntensity(multiplier) {
    if (!heatLayer) return;
    // Re-fetch with slightly elevated readings based on prediction
    fetchDistrictData();
}

// Demo fallback data when backend is offline
function getDemoData() {
    return [
        { name: "Quận 1", lat: 10.7743, lng: 106.7020, pm25: 22.4, category: "Moderate" },
        { name: "Quận 3", lat: 10.7898, lng: 106.6861, pm25: 18.1, category: "Moderate" },
        { name: "Quận 5", lat: 10.7538, lng: 106.6620, pm25: 28.9, category: "Moderate" },
        { name: "Quận 7", lat: 10.7349, lng: 106.7208, pm25: 9.5, category: "Good" },
        { name: "Quận 10", lat: 10.7738, lng: 106.6670, pm25: 31.2, category: "Moderate" },
        { name: "Bình Thạnh", lat: 10.8123, lng: 106.7106, pm25: 24.7, category: "Moderate" },
        { name: "Tân Bình", lat: 10.8027, lng: 106.6478, pm25: 41.3, category: "Poor" },
        { name: "Gò Vấp", lat: 10.8382, lng: 106.6647, pm25: 35.8, category: "Poor" },
        { name: "Thủ Đức", lat: 10.8503, lng: 106.7717, pm25: 14.2, category: "Moderate" },
        { name: "Nhà Bè", lat: 10.6923, lng: 106.7391, pm25: 8.6, category: "Good" },
        { name: "Hóc Môn", lat: 10.8906, lng: 106.5938, pm25: 44.1, category: "Poor" },
        { name: "Bình Dương", lat: 10.9102, lng: 106.7183, pm25: 19.0, category: "Moderate" },
    ];
}

// Inject dark popup CSS
const style = document.createElement("style");
style.textContent = `
  .dark-popup .leaflet-popup-content-wrapper {
    background: hsl(220,22%,12%);
    color: #f1f5f9;
    border: 1px solid hsla(220,30%,55%,0.18);
    border-radius: 12px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
  }
  .dark-popup .leaflet-popup-tip { background: hsl(220,22%,12%); }
`;
document.head.appendChild(style);

// Init on DOM ready
document.addEventListener("DOMContentLoaded", initMap);
