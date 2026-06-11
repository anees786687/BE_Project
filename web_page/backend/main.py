"""
VISOR Camera Dashboard — FastAPI Backend

Serves MJPEG video stream from a ROS2 image topic (or skeleton demo).
Provides REST endpoints for status and snapshots.
"""

import os
import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from dotenv import load_dotenv

from ros2_subscriber import frame_buffer, start_subscriber, get_mode

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the ROS2 subscriber on startup."""
    logger.info("Starting VISOR backend…")
    start_subscriber()
    yield
    logger.info("Shutting down VISOR backend…")


app = FastAPI(
    title="VISOR Camera Dashboard API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# MJPEG Streaming
# ---------------------------------------------------------------------------

def generate_mjpeg():
    """Generator that yields MJPEG frames."""
    while True:
        frame = frame_buffer.frame
        if frame is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame
                + b"\r\n"
            )
        # Wait for next frame or timeout
        frame_buffer.wait(timeout=0.1)


@app.get("/api/stream")
async def video_stream():
    """MJPEG video stream endpoint."""
    return StreamingResponse(
        generate_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Connection": "close",
        },
    )


# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/status")
async def get_status():
    """Return current camera/stream status."""
    info = frame_buffer.info
    mode = get_mode()
    topic = os.getenv("ROS2_IMAGE_TOPIC", "/image")

    return JSONResponse({
        "connected": info["connected"],
        "width": info["width"],
        "height": info["height"],
        "encoding": info["encoding"],
        "frame_count": info["frame_count"],
        "mode": mode,
        "topic": topic,
        "resolution": f"{info['width']}×{info['height']}" if info["width"] > 0 else "—",
        "device": f"ROS2 {topic}" if mode == "ros2" else "SKELETON DEMO",
    })


@app.get("/api/snapshot")
async def snapshot():
    """Return a single JPEG frame."""
    frame = frame_buffer.frame
    if frame is None:
        return JSONResponse({"error": "No frame available"}, status_code=503)

    return Response(
        content=frame,
        media_type="image/jpeg",
        headers={
            "Content-Disposition": f"attachment; filename=visor_{int(time.time())}.jpg",
        },
    )


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "mode": get_mode()}


# ---------------------------------------------------------------------------
# Run with: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
