"""
Image processing routes: original and enhanced SAR image generation.
Handles target discrimination, GMTI, and GeoTIFF data types.
"""

import io
import os
import time
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from PIL import Image

from sar_processor import (
    GPU_AVAILABLE,
    load_target_disc_mat,
    load_gmti_phase_history,
    load_geotiff_sar,
    form_range_doppler_image,
    form_backprojection_image,
    form_pfa_image,
    form_enhanced_gmti_image,
    enhance_geotiff_sar,
    _to_uint8_log,
)
from data_sources import UPLOADS_DIR

router = APIRouter(prefix="/api", tags=["processing"])

DATA_DIR = Path(os.environ.get("SAR_DATA_DIR", "/data"))


def _resolve_base_dir(base_dir_str: str) -> Path:
    """Resolve a base directory path from dataset metadata."""
    if base_dir_str:
        return Path(base_dir_str)
    return DATA_DIR


def _image_response(image_array: np.ndarray, image_size: int, proc_time: float = None) -> StreamingResponse:
    """Convert numpy array to PNG StreamingResponse with optional timing header."""
    img = Image.fromarray(image_array, mode="L")
    if img.size[0] != image_size or img.size[1] != image_size:
        img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    response = StreamingResponse(buf, media_type="image/png")
    if proc_time is not None:
        response.headers["X-Processing-Time-Ms"] = str(round(proc_time * 1000, 1))
    response.headers["X-GPU-Used"] = str(GPU_AVAILABLE)
    return response


# --- Target Discrimination Endpoints ---

@router.get("/image/target-disc/original")
def get_target_disc_original(
    vehicle: str = Query(...),
    file_index: int = Query(0),
    image_size: int = Query(512),
    base_dir: str = Query(""),
):
    """Get original (range-Doppler) image for target discrimination data."""
    data_root = _resolve_base_dir(base_dir)
    target_dir = data_root / "target-disc" / "Target-Discrimination-CP" / "Data" / vehicle
    if not target_dir.exists():
        raise HTTPException(status_code=404, detail=f"Vehicle '{vehicle}' not found")

    mat_files = sorted(target_dir.glob("PH_*.mat"))
    if file_index >= len(mat_files):
        raise HTTPException(status_code=404, detail="File index out of range")

    data = load_target_disc_mat(str(mat_files[file_index]))
    image = form_range_doppler_image(data["fq"].T)

    return _image_response(image, image_size)


@router.get("/image/target-disc/enhanced")
def get_target_disc_enhanced(
    vehicle: str = Query(...),
    file_index: int = Query(0),
    algorithm: str = Query("pfa"),
    image_size: int = Query(512),
    base_dir: str = Query(""),
):
    """Get NVRadar-enhanced (image-formed) image for target discrimination data."""
    data_root = _resolve_base_dir(base_dir)
    target_dir = data_root / "target-disc" / "Target-Discrimination-CP" / "Data" / vehicle
    if not target_dir.exists():
        raise HTTPException(status_code=404, detail=f"Vehicle '{vehicle}' not found")

    mat_files = sorted(target_dir.glob("PH_*.mat"))
    if file_index >= len(mat_files):
        raise HTTPException(status_code=404, detail="File index out of range")

    data = load_target_disc_mat(str(mat_files[file_index]))

    start = time.time()
    if algorithm == "bp":
        bp_size = min(image_size, 160)
        image = form_backprojection_image(data, image_size=bp_size)
    else:
        pfa_size = min(image_size, 256)
        image = form_pfa_image(data, image_size=pfa_size)
    proc_time = time.time() - start

    response = _image_response(image, image_size, proc_time)
    response.headers["X-Algorithm"] = algorithm
    return response


# --- GMTI Endpoints ---

@router.get("/image/gmti/original")
def get_gmti_original(
    channel: str = Query(...),
    image_size: int = Query(512),
    base_dir: str = Query(""),
):
    """Get original range-Doppler image for GMTI data."""
    data_root = _resolve_base_dir(base_dir)
    gmti_dir = data_root / "gmti" / "SAR-Based_GMTI_CP" / "SPIEchallengeData"
    ph_file = gmti_dir / channel
    aux_file = gmti_dir / f"{channel}_auxSaveData.mat"

    if not ph_file.exists():
        raise HTTPException(status_code=404, detail=f"Channel '{channel}' not found")
    if not aux_file.exists():
        raise HTTPException(status_code=404, detail=f"Aux file not found for '{channel}'")

    data = load_gmti_phase_history(str(ph_file), str(aux_file))
    image = form_range_doppler_image(data["phase_history"])

    return _image_response(image, image_size)


@router.get("/image/gmti/enhanced")
def get_gmti_enhanced(
    channel: str = Query(...),
    image_size: int = Query(512),
    base_dir: str = Query(""),
):
    """Get enhanced GMTI image with clutter suppression."""
    data_root = _resolve_base_dir(base_dir)
    gmti_dir = data_root / "gmti" / "SAR-Based_GMTI_CP" / "SPIEchallengeData"
    ph_file = gmti_dir / channel
    aux_file = gmti_dir / f"{channel}_auxSaveData.mat"

    if not ph_file.exists():
        raise HTTPException(status_code=404, detail=f"Channel '{channel}' not found")
    if not aux_file.exists():
        raise HTTPException(status_code=404, detail=f"Aux file not found for '{channel}'")

    start = time.time()
    data = load_gmti_phase_history(str(ph_file), str(aux_file))
    image = form_enhanced_gmti_image(data, image_size=image_size)
    proc_time = time.time() - start

    response = _image_response(image, image_size, proc_time)
    return response


# --- GeoTIFF Endpoints ---

@router.get("/image/geotiff/original")
def get_geotiff_original(
    path: str = Query(..., description="Relative path within base_dir"),
    image_size: int = Query(512),
    base_dir: str = Query(""),
):
    """Get original display of a GeoTIFF SAR image (log-scaled)."""
    data_root = _resolve_base_dir(base_dir)
    file_path = data_root / path

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    try:
        image = load_geotiff_sar(str(file_path))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load GeoTIFF: {e}")

    display = _to_uint8_log(image)
    return _image_response(display, image_size)


@router.get("/image/geotiff/enhanced")
def get_geotiff_enhanced(
    path: str = Query(..., description="Relative path within base_dir"),
    algorithm: str = Query("adaptive", description="Enhancement: lee, histogram_eq, adaptive"),
    image_size: int = Query(512),
    base_dir: str = Query(""),
):
    """Get enhanced GeoTIFF SAR image with speckle filtering and contrast enhancement."""
    data_root = _resolve_base_dir(base_dir)
    file_path = data_root / path

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    try:
        image = load_geotiff_sar(str(file_path))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load GeoTIFF: {e}")

    start = time.time()
    enhanced = enhance_geotiff_sar(image, method=algorithm)
    proc_time = time.time() - start

    response = _image_response(enhanced, image_size, proc_time)
    response.headers["X-Algorithm"] = algorithm
    return response
