# ChartAnalyst main
from fastapi import FastAPI, Request
import uvicorn

app = FastAPI(title="ChartAnalyst Agent")

@app.post("/detect_pattern")
async def detect_pattern(payload: dict):
    # TODO: integrate with Mistral Small 3.2 or mock output for now
    candles = payload.get("candles", [])
    # Stubbed response
    response = {
        "pattern": "Bullish Engulfing",
        "confidence": 0.85,
        "price_zones": {"support": 1970, "resistance": 1980}
    }
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
