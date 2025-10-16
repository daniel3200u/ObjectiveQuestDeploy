from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import numpy as np
import scipy.sparse as sp
import joblib
from catboost import CatBoostRegressor
import xgboost as xgb
import lightgbm as lgb
from preprocessing import preprocess_text as text_cleaner, extract_features

app = FastAPI(
    title="API Klasifikasi Dokumen Hukum",
    description="API ini melakukan preprocessing teks, menggabungkan fitur TF-IDF dan tabular, lalu melakukan prediksi lama hukuman dengan beberapa model ML.",
    version="1.0.0",
)

templates = Jinja2Templates(directory="templates")

model1 = CatBoostRegressor()
model1.load_model("catboost_model.cbm")

lgb_booster = lgb.Booster(model_file="lightgbm_model (1).txt")

model3 = xgb.XGBRegressor()
model3.load_model("xgboost_model.json")

vectorizer = joblib.load("vectorizer.pkl")

TABULAR_ORDER = ["num_unique_pasal", "hukuman_bulan", "num_pages", "pengurangan_tahanan"]

class TextInput(BaseModel):
    text: str

def create_feature_vector(raw_text: str):
    clean_text = text_cleaner(raw_text)
    tfidf_vector = vectorizer.transform([clean_text])
    if "id" in vectorizer.vocabulary_:
        id_index = vectorizer.vocabulary_["id"]
        mask = np.ones(tfidf_vector.shape[1], dtype=bool)
        mask[id_index] = False
        tfidf_vector = tfidf_vector[:, mask]
    feats = extract_features(raw_text)
    tabular_values = [float(feats.get(f, 0) or 0) for f in TABULAR_ORDER]
    tabular_array = np.array(tabular_values, dtype=np.float32).reshape(1, -1)
    combined = sp.hstack([tfidf_vector, tabular_array]).toarray().astype(np.float32)
    return combined

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict_hukuman(input: TextInput):
    X = create_feature_vector(input.text)
    y_catboost = float(model1.predict(X)[0])
    y_xgboost = float(model3.predict(X)[0])
    weights = {"catboost": 0.7268, "lightgbm": 0.5021, "xgboost": 0.2732}
    y_final = weights["catboost"] * y_catboost + weights["xgboost"] * y_xgboost
    return {
        "catboost_pred": y_catboost,
        "xgboost_pred": y_xgboost,
        "final_weighted_pred": round(float(y_final), 2)
    }

@app.post("/predict_file")
async def predict_from_file(file: UploadFile = File(...)):
    if not file or not file.filename or not file.filename.endswith(".txt"):
        return {"error": "Harap unggah file .txt yang valid."}
    content = await file.read()
    text = content.decode("utf-8")
    X = create_feature_vector(text)
    y_catboost = float(model1.predict(X)[0])
    y_xgboost = float(model3.predict(X)[0])
    weights = {"catboost": 0.7268, "lightgbm": 0.5021, "xgboost": 0.2732}
    y_final = weights["catboost"] * y_catboost + weights["xgboost"] * y_xgboost
    return {
        "filename": file.filename,
        "catboost_pred": y_catboost,
        "xgboost_pred": y_xgboost,
        "final_weighted_pred": round(float(y_final), 2)
    }

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
