"""
api_v2.py  —  Car Dent Detection API  (Version 2 — Deep Learning)
==================================================================
Upgraded FastAPI server that uses DentDetectorDL (Tier 2 upgrade).

New in v2:
  • POST /v2/analyze-dent     — single image, full repairman-level JSON
  • POST /v2/analyze-vehicle  — multi-image (different angles), merged result
  • GET  /v2/health            — status + current detection mode
  • GET  /                     — serves frontend/index.html
  • GET  /docs                 — Swagger UI (built-in via FastAPI)

Run:
    python -m uvicorn api_v2:app --host 127.0.0.1 --port 8001 --reload
"""

import os
import shutil
import uuid
import base64
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from dent_detector_dl import DentDetectorDL

FRONTEND_DIR = Path(__file__).parent / "frontend"

# ── Single global model instance (loaded once on startup) ────────────────────
detector = DentDetectorDL()

app = FastAPI(
    title       = "Car Dent Detection API — v2 (Deep Learning)",
    description = (
        "Analyzes car images for dents using YOLOv8-seg segmentation. "
        "Falls back to enhanced classical CV when no trained weights are found. "
        "Returns detailed repairman-level JSON + an annotated image (base64)."
    ),
    version     = "2.0.0",
)

# ── CORS — allow browser requests from any origin ────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve frontend static files at /static ───────────────────────────────────
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ── Root — serve the frontend HTML ───────────────────────────────────────────
@app.get("/")
async def root():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index), media_type="text/html")
    return {
        "status":  "online",
        "api":     "Car Dent Detection API v2",
        "mode":    detector.mode,
        "docs":    "http://127.0.0.1:8001/docs",
    }


# ── Health Check ─────────────────────────────────────────────────────────────
@app.get("/v2/health")
async def health():
    return {
        "status":         "ok",
        "detection_mode": detector.mode,
        "model_weights":  "loaded" if detector.model is not None else "unavailable",
        "info": (
            "Running with fine-tuned YOLOv8 weights."
            if detector.mode == "trained"
            else "Running in fallback mode (CLAHE + multi-dent CV). "
                 "Place trained weights at models/dent_best.pt to upgrade."
        ),
    }


# ── Single Image Analysis ─────────────────────────────────────────────────────
@app.post("/v2/analyze-dent")
async def analyze_dent(file: UploadFile, sensitivity: float = 75.0):
    """
    Upload a single car photo. Returns:
    - Overall severity score (0–100) and severity band
    - Per-dent breakdown: type, depth, paint damage, PDR eligibility
    - Cost & time estimates per dent + total
    - Annotated image (base64 JPEG) with dents highlighted
    """
    _validate_image(file)
    temp_path = _save_temp(file)

    try:
        result = detector.analyze(temp_path, sensitivity=sensitivity)
        if "error" in result:
            raise HTTPException(status_code=400, detail=str(result["error"]))
        return result
    except Exception as e:
        # Catch internal crashes and return them cleanly
        raise HTTPException(status_code=500, detail=f"Engine Error: {str(e)}")
    finally:
        _cleanup(temp_path)


# ── Multi-Angle Vehicle Analysis ──────────────────────────────────────────────
@app.post("/v2/analyze-vehicle")
async def analyze_vehicle(files: List[UploadFile], sensitivity: float = 75.0):
    """
    Upload 2–6 photos of the same vehicle from different angles.
    """
    if len(files) < 1:
        raise HTTPException(status_code=400, detail="At least 1 image required")
    if len(files) > 6:
        raise HTTPException(status_code=400, detail="Maximum 6 images allowed")

    for f in files:
        _validate_image(f)

    temp_paths = [_save_temp(f) for f in files]
    per_image_results = []

    try:
        for path in temp_paths:
            result = detector.analyze(path, sensitivity=sensitivity)
            if "error" not in result:
                per_image_results.append(result)
    finally:
        for path in temp_paths:
            _cleanup(path)

    if not per_image_results:
        return JSONResponse(content={"error": "Could not analyze any of the provided images"})

    return JSONResponse(content=_merge_results(per_image_results))


# ══════════════════════════════════════════════════════════════════════════════
#  Internal Helpers
# ══════════════════════════════════════════════════════════════════════════════
def _validate_image(file: UploadFile):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"File '{file.filename}' must be an image (got {file.content_type})"
        )


def _save_temp(file: UploadFile) -> str:
    temp_path = os.path.join(os.getcwd(), f"{uuid.uuid4()}.jpg")
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return temp_path


def _cleanup(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def _merge_results(results: list) -> dict:
    """
    Merge per-image results from multi-angle upload.
    Strategy: collect all dents, re-number them, take max overall severity.
    """
    all_dents     = []
    max_severity  = 0.0
    modes         = set()
    all_b64       = []

    for i, res in enumerate(results):
        modes.add(res.get("mode", "fallback"))
        score = res.get("severity_score", 0)
        if score > max_severity:
            max_severity = score

        # Re-number dents with view prefix to avoid ID collision
        for dent in res.get("dents", []):
            dent_copy        = dict(dent)
            dent_copy["id"] = f"View{i+1}-#{dent['id']}"
            dent_copy["view"] = i + 1
            all_dents.append(dent_copy)

        b64 = res.get("annotated_image_base64")
        if b64:
            all_b64.append(b64)

    # Cost aggregation
    from dent_detector_dl import REPAIR_COSTS, _score_to_band
    totals = []
    for d in all_dents:
        rec = d["recommendation"]
        lo, hi, _ = REPAIR_COSTS[rec]
        totals.append((lo, hi))
    total_lo = sum(t[0] for t in totals) if totals else 0
    total_hi = sum(t[1] for t in totals) if totals else 0

    avg_conf = round(
        sum(d["confidence"] for d in all_dents) / len(all_dents)
        if all_dents else 0, 3
    )

    return {
        "mode":                     "trained" if "trained" in modes else "fallback",
        "views_analyzed":           len(results),
        "dents_found":              len(all_dents),
        "overall_severity":         _score_to_band(max_severity),
        "severity_score":           round(max_severity, 1),
        "estimated_total_cost_usd": f"${total_lo}-${total_hi}" if totals else "$0",
        "confidence":               avg_conf,
        "dents":                    all_dents,
        "annotated_images_base64":  all_b64,   # one per view
    }
