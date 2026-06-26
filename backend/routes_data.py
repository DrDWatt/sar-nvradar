"""
Dataset management routes: listing, upload, download from remote sources, delete.
"""

import os
import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.io as sio
from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from fastapi.responses import JSONResponse

from data_sources import (
    UPLOADS_DIR,
    SAR_SOURCES,
    download_file,
    extract_zip,
    get_source_by_id,
    list_sources_metadata,
)

router = APIRouter(prefix="/api", tags=["data"])

DATA_DIR = Path(os.environ.get("SAR_DATA_DIR", "/data"))


def _extract_angle_from_filename(filename: str) -> str:
    """Extract azimuth angle from filename like PH_vehicle_0214.mat -> 214 deg."""
    parts = filename.replace(".mat", "").split("_")
    angle_str = parts[-1] if parts else ""
    try:
        return f"{int(angle_str)}\u00b0"
    except ValueError:
        return angle_str


def _find_best_file_index(mat_files: list) -> int:
    """Find the .mat file with highest signal energy (best SNR)."""
    best_idx = 0
    best_energy = 0.0

    for idx, f in enumerate(mat_files):
        try:
            raw = sio.loadmat(str(f))
            fq = raw["data"]["fq"][0][0]
            energy = float(np.sum(np.abs(fq) ** 2))
            if energy > best_energy:
                best_energy = energy
                best_idx = idx
        except Exception:
            continue

    return best_idx


def _scan_directory(base_dir: Path, source_label: str = "") -> list:
    """Scan a directory tree for SAR datasets. Returns dataset metadata list."""
    datasets = []
    if not base_dir.exists():
        return datasets

    # Pattern 1: Target discrimination .mat files (SPIE format)
    target_disc_dir = base_dir / "target-disc" / "Target-Discrimination-CP" / "Data"
    if target_disc_dir.exists():
        for vehicle_dir in sorted(target_disc_dir.iterdir()):
            if vehicle_dir.is_dir():
                mat_files = sorted(vehicle_dir.glob("PH_*.mat"))
                if mat_files:
                    file_details = []
                    for i, f in enumerate(mat_files):
                        angle = _extract_angle_from_filename(f.name)
                        file_details.append({"index": i, "name": f.name, "angle": angle})

                    best_idx = _find_best_file_index(mat_files)
                    prefix = f"{source_label} " if source_label else ""
                    datasets.append({
                        "id": f"target-disc/{vehicle_dir.name}",
                        "name": f"{prefix}Target Disc - {vehicle_dir.name}",
                        "type": "target_discrimination",
                        "file_count": len(mat_files),
                        "files": file_details,
                        "best_index": best_idx,
                        "base_dir": str(base_dir),
                    })

    # Pattern 2: GMTI binary phase history
    gmti_dir = base_dir / "gmti" / "SAR-Based_GMTI_CP" / "SPIEchallengeData"
    if gmti_dir.exists():
        ph_files = sorted(gmti_dir.glob("*_PH"))
        for ph_file in ph_files:
            aux_file = Path(str(ph_file) + "_auxSaveData.mat")
            if aux_file.exists():
                prefix = f"{source_label} " if source_label else ""
                datasets.append({
                    "id": f"gmti/{ph_file.name}",
                    "name": f"{prefix}GMTI - {ph_file.stem}",
                    "type": "gmti",
                    "file_count": 1,
                    "files": [{"index": 0, "name": ph_file.name, "angle": "N/A"}],
                    "best_index": 0,
                    "base_dir": str(base_dir),
                })

    # Pattern 3: GeoTIFF files (any .tif/.tiff in directory tree)
    tif_files = sorted(base_dir.rglob("*.tif")) + sorted(base_dir.rglob("*.tiff"))
    if tif_files:
        for tif_file in tif_files:
            rel_path = tif_file.relative_to(base_dir)
            prefix = f"{source_label} " if source_label else ""
            datasets.append({
                "id": f"geotiff/{rel_path}",
                "name": f"{prefix}{tif_file.stem}",
                "type": "geotiff",
                "file_count": 1,
                "files": [{"index": 0, "name": tif_file.name, "angle": "N/A"}],
                "best_index": 0,
                "base_dir": str(base_dir),
            })

    # Pattern 4: Loose .mat files (not in target-disc structure)
    loose_mats = [f for f in base_dir.rglob("*.mat")
                  if "Target-Discrimination-CP" not in str(f)
                  and "auxSaveData" not in f.name
                  and "SPIEchallengeData" not in str(f)]
    for mat_file in sorted(loose_mats):
        rel_path = mat_file.relative_to(base_dir)
        prefix = f"{source_label} " if source_label else ""
        datasets.append({
            "id": f"mat/{rel_path}",
            "name": f"{prefix}{mat_file.stem}",
            "type": "target_discrimination",
            "file_count": 1,
            "files": [{"index": 0, "name": mat_file.name, "angle": "N/A"}],
            "best_index": 0,
            "base_dir": str(base_dir),
        })

    return datasets


def scan_all_datasets() -> list:
    """Scan both mounted data and uploads directories."""
    datasets = []

    # Scan mounted data directory (if present)
    if DATA_DIR.exists() and any(DATA_DIR.iterdir()):
        datasets.extend(_scan_directory(DATA_DIR, source_label=""))

    # Scan each upload subdirectory
    if UPLOADS_DIR.exists():
        for upload_dir in sorted(UPLOADS_DIR.iterdir()):
            if upload_dir.is_dir():
                datasets.extend(_scan_directory(upload_dir, source_label=f"[{upload_dir.name}]"))

    return datasets


@router.get("/datasets")
def list_datasets():
    """List all available SAR datasets from mounted data and uploads."""
    datasets = scan_all_datasets()
    return {"datasets": datasets}


@router.get("/sources")
def list_sources():
    """List available remote SAR data sources."""
    return {"sources": list_sources_metadata()}


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a zip file containing SAR data. Extracts and makes available for processing."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Determine dataset name from filename
    dataset_name = file.filename.replace(".zip", "").replace(" ", "_")
    dest_dir = UPLOADS_DIR / dataset_name

    # Remove existing if re-uploading
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded file
    zip_path = dest_dir / file.filename
    content = await file.read()
    with open(zip_path, "wb") as f:
        f.write(content)

    # Extract if zip
    if file.filename.endswith(".zip"):
        try:
            extract_zip(zip_path, dest_dir)
            zip_path.unlink()  # Remove zip after extraction
        except Exception as e:
            shutil.rmtree(dest_dir)
            raise HTTPException(status_code=400, detail=f"Failed to extract zip: {e}")

    # Scan for datasets in the uploaded content
    new_datasets = _scan_directory(dest_dir, source_label=f"[{dataset_name}]")

    return {
        "message": f"Uploaded and extracted '{file.filename}'",
        "dataset_name": dataset_name,
        "datasets_found": len(new_datasets),
        "datasets": new_datasets,
    }


@router.post("/download")
async def download_from_url(
    url: str = Query(..., description="URL to download SAR data from"),
    source_id: Optional[str] = Query(None, description="Source catalog ID (optional)"),
    dataset_name: Optional[str] = Query(None, description="Name for the dataset"),
):
    """Download SAR data from a remote URL. Supports direct files and zip archives."""
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="URL must start with http:// or https://")

    # Derive dataset name
    if not dataset_name:
        if source_id:
            src = get_source_by_id(source_id)
            dataset_name = src["id"] if src else "download"
        else:
            dataset_name = url.split("/")[-1].split(".")[0].replace(" ", "_") or "download"

    dest_dir = UPLOADS_DIR / dataset_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    try:
        downloaded_path = await download_file(url, dest_dir)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Download failed: {e}")

    # If zip, extract
    if downloaded_path.suffix.lower() == ".zip":
        try:
            extract_zip(downloaded_path, dest_dir)
            downloaded_path.unlink()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Extraction failed: {e}")

    # Scan new datasets
    new_datasets = _scan_directory(dest_dir, source_label=f"[{dataset_name}]")

    return {
        "message": f"Downloaded from {url}",
        "dataset_name": dataset_name,
        "datasets_found": len(new_datasets),
        "datasets": new_datasets,
    }


@router.delete("/datasets/{dataset_name}")
def delete_dataset(dataset_name: str):
    """Delete an uploaded dataset by name."""
    dest_dir = UPLOADS_DIR / dataset_name
    if not dest_dir.exists():
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found in uploads")

    shutil.rmtree(dest_dir)
    return {"message": f"Deleted dataset '{dataset_name}'"}
