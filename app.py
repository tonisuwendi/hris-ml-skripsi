from flask import Flask, request, jsonify
import joblib
import pandas as pd
import re
import os
import hmac
from dotenv import load_dotenv
import shap

load_dotenv()

app = Flask(__name__)
ML_API_KEY = os.getenv("ML_API_KEY")
MODEL_PREPROCESSOR_URL = os.getenv("MODEL_PREPROCESSOR_URL")
MODEL_RANDOM_FOREST_URL = os.getenv("MODEL_RANDOM_FOREST_URL")

preprocessor = joblib.load(MODEL_PREPROCESSOR_URL)
model = joblib.load(MODEL_RANDOM_FOREST_URL)

def check_api_key(req):
  header = req.headers.get("x-api-key")
  if header is None:
    return False
  else:
    token = header.strip()
  return bool(ML_API_KEY) and hmac.compare_digest(token, ML_API_KEY)

@app.before_request
def require_api_key():
  if request.path in ("/"):
    return None
  if not check_api_key(request):
    return jsonify({"error": "Unauthorized"}), 401

@app.route("/", methods=["GET"])
def home():
  return jsonify({
    "status": "success",
    "message": "HRIS ML API Ready"
  })
  
@app.route("/predict", methods=["POST"])
def predict():
  try:
    data = request.get_json()

    if isinstance(data, dict):
      data = [data]
      
    key_mapping = {
      "work_mode": "Lokasi Kerja",
      "job_position": "Jabatan/Posisi",
      "performance_score": "Skor Kinerja",
      "attendance_count": "Kehadiran Digital",
      "project_completed": "Jumlah Proyek Selesai",
      "years_of_service": "Masa Kerja"
    }

    mapped_list = []
    for item in data:
      mapped = {key_mapping.get(k, k): v for k, v in item.items()}
      mapped_list.append(mapped)

    df = pd.DataFrame(mapped_list)
    processed = preprocessor.transform(df)
    prediction = model.predict(processed)

    return jsonify({
      "status": "success",
      "count": len(prediction),
      "predicted_salary": [round(float(p), 2) for p in prediction]
    })
  
  except Exception as e:
    return jsonify({
      "status": "error",
      "message": str(e)
    }), 400
  
@app.route("/insight", methods=["POST"])
def insight():
  try:
    data = request.get_json()

    key_mapping = {
      "work_mode": "Lokasi Kerja",
      "job_position": "Jabatan/Posisi",
      "performance_score": "Skor Kinerja",
      "attendance_count": "Kehadiran Digital",
      "project_completed": "Jumlah Proyek Selesai",
      "years_of_service": "Masa Kerja"
    }

    mapped_data = {key_mapping.get(k, k): v for k, v in data.items()}
    df = pd.DataFrame([mapped_data])

    processed = preprocessor.transform(df)
    predicted_salary = model.predict(processed)[0]

    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(model)
    explainer = shap.Explainer(model, algorithm="auto")
    shap_values = explainer.shap_values(processed)

    shap_df = pd.DataFrame({
      "feature": feature_names,
      "shap_value": shap_values[0]
    })
    shap_df["abs_value"] = shap_df["shap_value"].abs()
    shap_df["base_feature"] = shap_df["feature"].apply(lambda x: re.sub(r"cat__|num__|_.+$", "", x))

    agg = shap_df.groupby("base_feature")["abs_value"].sum().reset_index()
    agg["influence_percent"] = (agg["abs_value"] / agg["abs_value"].sum()) * 100
    agg = agg.sort_values("influence_percent", ascending=False)

    def describe_feature(feature_name, value, percent):
      desc_map = {
        "Jabatan/Posisi": f"Jabatan {value} memengaruhi gaji sebesar {percent:.1f}%.",
        "Lokasi Kerja": f"Mode kerja {value} memberikan pengaruh sekitar {percent:.1f}%.",
        "Skor Kinerja": f"Skor kinerja {value} berkontribusi {percent:.1f}% terhadap gaji.",
        "Masa Kerja": f"Masa kerja {value} tahun berpengaruh sekitar {percent:.1f}%.",
        "Jumlah Proyek Selesai": f"Jumlah proyek {value} memberikan kontribusi {percent:.1f}%.",
        "Kehadiran Digital": f"Kehadiran digital {value} memberikan dampak {percent:.1f}% terhadap gaji."
      }
      return desc_map.get(feature_name, f"Faktor {feature_name} berpengaruh {percent:.1f}% terhadap gaji.")

    insights = []
    for _, row in agg.iterrows():
      base = row["base_feature"]
      val = None
      if "Jabatan/Posisi" in base:
        val = data.get("job_position")
      elif "Lokasi Kerja" in base:
        val = data.get("work_mode")
      elif "Skor Kinerja" in base:
        val = data.get("performance_score")
      elif "Masa Kerja" in base:
        val = data.get("years_of_service")
      elif "Jumlah Proyek" in base:
        val = data.get("project_completed")
      elif "Kehadiran" in base:
        val = data.get("attendance_count")

      insights.append({
        "feature": base,
        "value": val,
        "influence_percent": round(row["influence_percent"], 2),
        "description": describe_feature(base, val, row["influence_percent"])
      })

    return jsonify({
      "status": "success",
      "predicted_salary": round(float(predicted_salary), 2),
      "feature_influence": insights
    })

  except Exception as e:
    return jsonify({
      "status": "error",
      "message": str(e)
    }), 400

if __name__ == "__main__":
  app.run(host="0.0.0.0", port=1010)
