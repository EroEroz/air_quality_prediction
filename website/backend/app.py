from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os
import predictor

# Serve the website/ folder as static files
WEBSITE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(
    __name__,
    static_folder=WEBSITE_DIR,
    static_url_path="",
)
CORS(app)


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/predict", methods=["POST", "GET"])
def predict():
    try:
        result = predictor.predict()
        return jsonify({"status": "ok", "data": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/current", methods=["GET"])
def current():
    try:
        result = predictor.current_district_data()
        return jsonify({"status": "ok", "data": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/day-period", methods=["POST"])
def day_period():
    try:
        body   = request.get_json(force=True) or {}
        date   = body.get("date", "")
        period = body.get("period", "morning")
        # Map old period inputs to new Shift expectations
        shift_map = {"morning": "Morning", "afternoon": "Afternoon", "evening": "Night"}
        shift_value = shift_map.get(period.lower(), "Morning")
        
        result = predictor.predict_shift(date, shift_value)
        
        # Map back to old UI format so grid lookup succeeds
        if result["period"] == "night":
            result["period"] = "evening"

        return jsonify({"status": "ok", "data": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Air Quality API is running"})


if __name__ == "__main__":
    print("=" * 50)
    print("  HCMC Air Quality Prediction API")
    print("  Running at: http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, port=5000)
