"""
STEP 6b: MAIN WEBSITE (simulated)
===================================
This simulates the real website that ADAPT-SHIELD protects.
In production, this would be your actual web server.

All traffic first goes through the Security Layer (port 8000).
If ALLOWED, security layer forwards to this website (port 9000).
If BLOCKED, security layer returns 403 Forbidden.
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
import httpx

SECURITY_LAYER_URL = "http://security-layer:8000"  # Docker service name

app = FastAPI(title="Main Website", description="Protected by ADAPT-SHIELD")


@app.get("/", response_class=HTMLResponse)
async def home():
    """Simulated website homepage."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Protected Website</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
            .shield { color: #1a73e8; font-size: 48px; }
            .status { background: #d4edda; border: 1px solid #c3e6cb; padding: 15px; border-radius: 8px; }
            h1 { color: #333; }
        </style>
    </head>
    <body>
        <div class="shield">🛡️</div>
        <h1>Welcome! You are protected by ADAPT-SHIELD</h1>
        <div class="status">
            ✅ Your connection has been verified as <strong>BENIGN</strong><br>
            🔒 ADAPT-SHIELD ML model is actively monitoring traffic
        </div>
        <p>This is the main website. Only traffic classified as BENIGN reaches here.</p>
        <p><a href="http://localhost:8000/docs">View Security Layer API →</a></p>
    </body>
    </html>
    """


@app.get("/health")
def health():
    return {"status": "healthy", "service": "main-website"}


@app.get("/api/data")
def get_data():
    """Sample protected API endpoint."""
    return {
        "message": "This is sensitive data — only accessible after ML verification",
        "records": [{"id": 1, "value": "secure_data"}]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)