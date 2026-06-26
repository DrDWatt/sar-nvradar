"""
SAR remote data source catalog and download utilities.
Provides source metadata for the frontend and async download/extraction helpers.
"""

import os
import shutil
import zipfile
import logging
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

UPLOADS_DIR = Path(os.environ.get("SAR_UPLOADS_DIR", "/uploads"))

# Catalog of supported remote SAR data sources
SAR_SOURCES = [
    {
        "id": "sen1floods11",
        "name": "Sen1Floods11",
        "purpose": "SAR + EO layer pipeline, small chips, fastest development loop",
        "description": "Sentinel-1 SAR flood detection dataset. 512x512 GeoTIFF chips with VV/VH bands.",
        "format": "geotiff",
        "auth_required": False,
        "docs_url": "https://github.com/cloudtostreet/Sen1Floods11",
        "samples": [
            {"name": "Bolivia S1 Flood", "url": "https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand/Bolivia_103757.tif", "size_mb": 0.5},
            {"name": "India S1 Flood", "url": "https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand/India_222338.tif", "size_mb": 0.5},
            {"name": "USA Houston Flood", "url": "https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand/USA_Houston_104001.tif", "size_mb": 0.5},
        ],
    },
    {
        "id": "asf_sentinel1",
        "name": "ASF Sentinel-1 GRD",
        "purpose": "Authoritative original Sentinel-1 product handling",
        "description": "Alaska Satellite Facility Sentinel-1 GRD products in SAFE format. Full scenes.",
        "format": "safe_zip",
        "auth_required": True,
        "auth_instructions": "Requires NASA Earthdata Login. Register at https://urs.earthdata.nasa.gov/ then search at https://search.asf.alaska.edu/",
        "docs_url": "https://search.asf.alaska.edu/",
        "samples": [],
    },
    {
        "id": "planetary_computer_s1",
        "name": "Planetary Computer S1 RTC",
        "purpose": "STAC/COG cloud-native workflow",
        "description": "Microsoft Planetary Computer Sentinel-1 RTC as Cloud-Optimized GeoTIFFs via STAC.",
        "format": "cog",
        "auth_required": False,
        "auth_instructions": "Browse at https://planetarycomputer.microsoft.com/dataset/sentinel-1-rtc and copy COG URLs.",
        "docs_url": "https://planetarycomputer.microsoft.com/dataset/sentinel-1-rtc",
        "samples": [],
    },
    {
        "id": "capella_open",
        "name": "Capella Open Data",
        "purpose": "High-resolution commercial SAR COG enhancement",
        "description": "Capella Space open data. High-resolution spotlight SAR as Cloud-Optimized GeoTIFFs.",
        "format": "cog",
        "auth_required": False,
        "docs_url": "https://www.capellaspace.com/community/open-data/",
        "samples": [],
    },
    {
        "id": "umbra_open",
        "name": "Umbra Open Data",
        "purpose": "Very-high-resolution spotlight SAR and special format handling",
        "description": "Umbra Space open data. 25cm resolution spotlight SAR GeoTIFF imagery.",
        "format": "geotiff",
        "auth_required": False,
        "docs_url": "https://umbra.space/open-data",
        "samples": [],
    },
    {
        "id": "iceye_open",
        "name": "ICEYE Open Data",
        "purpose": "Commercial GRD/SLC/COG vendor diversity",
        "description": "ICEYE SAR satellite open datasets. GRD and SLC products.",
        "format": "geotiff",
        "auth_required": True,
        "auth_instructions": "Register at https://www.iceye.com/open-data to access download links.",
        "docs_url": "https://www.iceye.com/open-data",
        "samples": [],
    },
    {
        "id": "uavsar",
        "name": "UAVSAR",
        "purpose": "Advanced airborne/L-band/future phase-aware testing",
        "description": "NASA/JPL UAVSAR airborne L-band SAR. SLC and multi-look products.",
        "format": "binary",
        "auth_required": False,
        "docs_url": "https://uavsar.jpl.nasa.gov/",
        "samples": [],
    },
    {
        "id": "mstar_opensarship",
        "name": "MSTAR / OpenSARShip",
        "purpose": "Chip-level target enhancement and detection experiments",
        "description": "MSTAR target recognition and OpenSARShip ship detection datasets.",
        "format": "raw_chips",
        "auth_required": False,
        "docs_url": "https://www.sdms.afrl.af.mil/index.php?collection=mstar",
        "samples": [],
    },
]


async def download_file(url: str, dest_dir: Path, filename: Optional[str] = None) -> Path:
    """Download a file from URL via streaming. Returns path to saved file."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = url.split("/")[-1].split("?")[0]
        if not filename:
            filename = "download.dat"

    dest_path = dest_dir / filename

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=30.0, read=600.0, write=30.0, pool=30.0),
        follow_redirects=True,
    ) as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            with open(dest_path, "wb") as f:
                async for chunk in response.aiter_bytes(chunk_size=65536):
                    f.write(chunk)

    logger.info(f"Downloaded {url} -> {dest_path} ({dest_path.stat().st_size} bytes)")
    return dest_path


def extract_zip(zip_path: Path, dest_dir: Path) -> Path:
    """Extract a zip archive into dest_dir. Returns extraction root."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    logger.info(f"Extracted {zip_path} -> {dest_dir}")
    return dest_dir


def get_source_by_id(source_id: str) -> Optional[dict]:
    """Look up a data source definition by ID."""
    for src in SAR_SOURCES:
        if src["id"] == source_id:
            return src
    return None


def list_sources_metadata() -> list:
    """Return catalog metadata for the frontend."""
    return [
        {
            "id": s["id"],
            "name": s["name"],
            "purpose": s["purpose"],
            "description": s["description"],
            "format": s["format"],
            "auth_required": s.get("auth_required", False),
            "auth_instructions": s.get("auth_instructions", ""),
            "docs_url": s.get("docs_url", ""),
            "samples": s.get("samples", []),
        }
        for s in SAR_SOURCES
    ]
