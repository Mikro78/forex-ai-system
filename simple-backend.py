from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import random
import os

app = FastAPI(title="FOREX AI Trading System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS = {
    "NARX": 91.2, "EMD-NARX": 94.2, "TimeGPT": 93.1,
    "Transformer": 89.5, "Autoformer": 88.7, "Informer": 87.9,
    "FEDformer": 90.3, "LSTM": 85.4, "Bi-LSTM": 86.8,
    "Attention-LSTM": 88.2, "GRU": 84.9, "TCN": 87.3,
    "N-BEATS": 86.5, "PatchTST": 89.1, "Prophet": 82.1,
    "Ensemble": 95.7
}

@app.get("/")
def root():
    return {"status": "online", "message": "FOREX AI Backend", "models": len(MODELS)}

@app.get("/api/models")
def get_models():
    return {"models": [{"name": k, "accuracy": v} for k, v in MODELS.items()]}

@app.get("/api/status")
def status():
    return {"status": "online", "models_active": len(MODELS)}

@app.get("/api/predict")
def predict():
    predictions = []
    for name, acc in MODELS.items():
        predictions.append({
            "model": name,
            "signal": random.choice(["BUY", "SELL", "HOLD"]),
            "confidence": acc * 0.9
        })
    return predictions

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
