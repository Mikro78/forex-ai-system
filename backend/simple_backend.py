from fastapi import FastAPI
import uvicorn
import os

app = FastAPI(title="FOREX AI Trading System")

@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "FOREX AI Backend is running!",
        "version": "1.0"
    }

@app.get("/api/status")
async def status():
    return {"status": "online", "models": "simplified version"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
