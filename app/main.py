"""
STEP 6: FASTAPI SECURITY LAYER
================================
This is the ML-powered security API that:
- Receives network traffic data
- Runs it through the ML model
- Returns ALLOW or BLOCK decision

Run locally: uvicorn app.main:app --reload --port 8000
Then visit:  http://localhost:8000/docs  (Swagger UI)
"""

import time
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Make sure src/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from predict import predictor

# ── App Setup ───────────────────────────────────────────────────
app = FastAPI(
    title="ADAPT-SHIELD Security API",
    description="ML-powered Network Intrusion Detection System",
    version="1.0.0",
)

# Track request stats for monitoring
stats = {"total_requests": 0, "blocked": 0, "allowed": 0, "start_time": time.time()}


# ── Request/Response Schemas ─────────────────────────────────────
class TrafficFeatures(BaseModel):
    """Input: one network flow record (all features optional — missing = 0)."""
    Protocol: float = 0
    Flow_Duration: float = 0
    Total_Fwd_Packets: float = 0
    Total_Backward_Packets: float = 0
    Fwd_Packets_Length_Total: float = 0
    Bwd_Packets_Length_Total: float = 0
    Fwd_Packet_Length_Max: float = 0
    Fwd_Packet_Length_Min: float = 0
    Fwd_Packet_Length_Mean: float = 0
    Flow_Bytes_s: float = 0
    Flow_Packets_s: float = 0
    SYN_Flag_Count: float = 0
    ACK_Flag_Count: float = 0
    URG_Flag_Count: float = 0
    PSH_Flag_Count: float = 0
    FIN_Flag_Count: float = 0
    RST_Flag_Count: float = 0
    Packet_Length_Mean: float = 0
    Packet_Length_Std: float = 0

    class Config:
        # Allow extra fields (all 77 features)
        extra = "allow"


class PredictionResponse(BaseModel):
    prediction: str
    is_malicious: bool
    confidence: float
    action: str          # "ALLOW" or "BLOCK"
    processing_time_ms: float


# ── Startup: Load Model ──────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Load ML model when API starts."""
    print("🚀 ADAPT-SHIELD Security API starting...")
    success = predictor.load()
    if not success:
        print("⚠️  Model not loaded — run training first!")
    else:
        print("✅ Model ready for inference!")


# ── Endpoints ────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "service": "ADAPT-SHIELD Security Layer",
        "model_loaded": predictor._loaded,
    }


@app.get("/health", tags=["Health"])
def health():
    """Detailed health status."""
    uptime = time.time() - stats["start_time"]
    return {
        "status": "healthy",
        "model_loaded": predictor._loaded,
        "uptime_seconds": round(uptime, 1),
        "total_requests": stats["total_requests"],
        "blocked_requests": stats["blocked"],
        "allowed_requests": stats["allowed"],
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Detection"])
async def predict(features: TrafficFeatures):
    """
    **Main endpoint**: Analyze network traffic and detect attacks.
    
    - **Input**: Network flow features (Protocol, packet counts, flags, etc.)
    - **Output**: Attack type prediction + ALLOW/BLOCK decision
    
    Example attack that will be BLOCKED:
    ```json
    {"SYN_Flag_Count": 1000, "Flow_Packets_s": 50000, "Protocol": 6}
    ```
    """
    if not predictor._loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run training first: python src/train.py"
        )

    start_time = time.time()
    stats["total_requests"] += 1

    try:
        # Convert pydantic model to dict, handle field name mapping
        feature_dict = {}
        raw = features.dict()

        # Map API field names (underscores) back to dataset names (spaces/slashes)
        name_map = {
            "Protocol": "Protocol",
            "Flow_Duration": "Flow Duration",
            "Total_Fwd_Packets": "Total Fwd Packets",
            "Total_Backward_Packets": "Total Backward Packets",
            "Fwd_Packets_Length_Total": "Fwd Packets Length Total",
            "Bwd_Packets_Length_Total": "Bwd Packets Length Total",
            "Fwd_Packet_Length_Max": "Fwd Packet Length Max",
            "Fwd_Packet_Length_Min": "Fwd Packet Length Min",
            "Fwd_Packet_Length_Mean": "Fwd Packet Length Mean",
            "Flow_Bytes_s": "Flow Bytes/s",
            "Flow_Packets_s": "Flow Packets/s",
            "SYN_Flag_Count": "SYN Flag Count",
            "ACK_Flag_Count": "ACK Flag Count",
            "URG_Flag_Count": "URG Flag Count",
            "PSH_Flag_Count": "PSH Flag Count",
            "FIN_Flag_Count": "FIN Flag Count",
            "RST_Flag_Count": "RST Flag Count",
            "Packet_Length_Mean": "Packet Length Mean",
            "Packet_Length_Std": "Packet Length Std",
        }

        for api_name, dataset_name in name_map.items():
            if api_name in raw:
                feature_dict[dataset_name] = raw[api_name]

        result = predictor.predict(feature_dict)

        # Update stats
        if result["is_malicious"]:
            stats["blocked"] += 1
        else:
            stats["allowed"] += 1

        elapsed_ms = (time.time() - start_time) * 1000

        return PredictionResponse(
            prediction=result["prediction"],
            is_malicious=result["is_malicious"],
            confidence=result["confidence"],
            action=result["action"],
            processing_time_ms=round(elapsed_ms, 2),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", tags=["Monitoring"])
def get_stats():
    """View real-time API statistics."""
    total = stats["total_requests"]
    block_rate = (stats["blocked"] / total * 100) if total > 0 else 0
    return {
        **stats,
        "block_rate_percent": round(block_rate, 2),
    }


@app.get("/classes", tags=["Info"])
def get_classes():
    """List all attack classes the model can detect."""
    if predictor.label_encoder:
        return {"classes": list(predictor.label_encoder.classes_)}
    return {"classes": []}


# ── Run directly ─────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)