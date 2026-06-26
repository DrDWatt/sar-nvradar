"""
Auto-enhancement agent routes.
Exposes the NVRadar agent pipeline that automatically selects the best algorithm.
"""

import io
import os
import time
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image

from sar_processor import (
    GPU_AVAILABLE,
    load_target_disc_mat,
    load_gmti_phase_history,
    load_geotiff_sar,
)
from agent import auto_enhance
from data_sources import UPLOADS_DIR

router = APIRouter(prefix="/api", tags=["agent"])

DATA_DIR = Path(os.environ.get("SAR_DATA_DIR", "/data"))


def _resolve_base_dir(base_dir_str: str) -> Path:
    """Resolve a base directory path from dataset metadata."""
    if base_dir_str:
        return Path(base_dir_str)
    return DATA_DIR


def _load_data(data_type: str, data_root: Path, **kwargs) -> dict:
    """Load SAR data based on type and parameters."""
    if data_type == "target_discrimination":
        vehicle = kwargs["vehicle"]
        file_index = kwargs.get("file_index", 0)
        target_dir = data_root / "target-disc" / "Target-Discrimination-CP" / "Data" / vehicle
        if not target_dir.exists():
            raise HTTPException(status_code=404, detail=f"Vehicle '{vehicle}' not found")
        mat_files = sorted(target_dir.glob("PH_*.mat"))
        if file_index >= len(mat_files):
            raise HTTPException(status_code=404, detail="File index out of range")
        return load_target_disc_mat(str(mat_files[file_index]))

    elif data_type == "gmti":
        channel = kwargs["channel"]
        gmti_dir = data_root / "gmti" / "SAR-Based_GMTI_CP" / "SPIEchallengeData"
        ph_file = gmti_dir / channel
        aux_file = gmti_dir / f"{channel}_auxSaveData.mat"
        if not ph_file.exists():
            raise HTTPException(status_code=404, detail=f"Channel '{channel}' not found")
        if not aux_file.exists():
            raise HTTPException(status_code=404, detail=f"Aux file not found for '{channel}'")
        return load_gmti_phase_history(str(ph_file), str(aux_file))

    elif data_type == "geotiff":
        path = kwargs["path"]
        file_path = data_root / path
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {path}")
        raw_image = load_geotiff_sar(str(file_path))
        return {"raw_image": raw_image}

    raise HTTPException(status_code=400, detail=f"Unknown data type: {data_type}")


@router.post("/auto-enhance")
async def run_auto_enhance(
    data_type: str = Query(..., description="Dataset type: target_discrimination, gmti, geotiff"),
    image_size: int = Query(512, description="Output image size"),
    base_dir: str = Query("", description="Base directory for data"),
    vehicle: str = Query(None, description="Vehicle name (target_discrimination)"),
    file_index: int = Query(0, description="File index (target_discrimination)"),
    channel: str = Query(None, description="Channel name (gmti)"),
    path: str = Query(None, description="Relative file path (geotiff)"),
):
    """Run the NVRadar auto-enhancement agent.

    Automatically runs all applicable algorithms, computes quality metrics,
    and uses NVIDIA Nemotron LLM to recommend the best result.
    """
    data_root = _resolve_base_dir(base_dir)

    # Build kwargs based on data type
    load_kwargs = {}
    if data_type == "target_discrimination":
        if not vehicle:
            raise HTTPException(status_code=400, detail="vehicle parameter required")
        load_kwargs = {"vehicle": vehicle, "file_index": file_index}
    elif data_type == "gmti":
        if not channel:
            raise HTTPException(status_code=400, detail="channel parameter required")
        load_kwargs = {"channel": channel}
    elif data_type == "geotiff":
        if not path:
            raise HTTPException(status_code=400, detail="path parameter required")
        load_kwargs = {"path": path}

    # Load data
    start_total = time.time()
    data = _load_data(data_type, data_root, **load_kwargs)

    # Run agent pipeline
    result = await auto_enhance(data_type, data, image_size)
    total_time = time.time() - start_total

    if result["best_image"] is None:
        raise HTTPException(status_code=500, detail="No algorithms produced valid results")

    # Return JSON with analysis (image fetched separately)
    return {
        "best_algorithm": result["best_algorithm"],
        "best_algorithm_name": result["best_algorithm_name"],
        "best_metrics": result["best_metrics"],
        "analysis": result["analysis"],
        "all_results": result["all_results"],
        "total_time_ms": round(total_time * 1000, 1),
        "gpu_used": result["gpu_used"],
        "llm_model": result["llm_model"],
    }


@router.get("/auto-enhance/image")
async def get_auto_enhanced_image(
    data_type: str = Query(...),
    image_size: int = Query(512),
    base_dir: str = Query(""),
    vehicle: str = Query(None),
    file_index: int = Query(0),
    channel: str = Query(None),
    path: str = Query(None),
):
    """Get the best auto-enhanced image (PNG).

    Runs the full pipeline and returns the winning algorithm's image.
    """
    data_root = _resolve_base_dir(base_dir)

    load_kwargs = {}
    if data_type == "target_discrimination":
        if not vehicle:
            raise HTTPException(status_code=400, detail="vehicle parameter required")
        load_kwargs = {"vehicle": vehicle, "file_index": file_index}
    elif data_type == "gmti":
        if not channel:
            raise HTTPException(status_code=400, detail="channel parameter required")
        load_kwargs = {"channel": channel}
    elif data_type == "geotiff":
        if not path:
            raise HTTPException(status_code=400, detail="path parameter required")
        load_kwargs = {"path": path}

    data = _load_data(data_type, data_root, **load_kwargs)
    result = await auto_enhance(data_type, data, image_size)

    if result["best_image"] is None:
        raise HTTPException(status_code=500, detail="No algorithms produced valid results")

    # Convert to PNG
    img = Image.fromarray(result["best_image"], mode="L")
    if img.size[0] != image_size or img.size[1] != image_size:
        img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    response = StreamingResponse(buf, media_type="image/png")
    response.headers["X-Best-Algorithm"] = result["best_algorithm"]
    response.headers["X-Composite-Score"] = str(result["best_metrics"]["composite_score"])
    response.headers["X-GPU-Used"] = str(result["gpu_used"])
    return response
