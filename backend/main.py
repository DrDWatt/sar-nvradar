"""
SAR Processing Web API - FastAPI Backend
GPU-accelerated SAR image formation using NVRadar-style processing.
Data loaded on-demand via upload or remote source download.
"""

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from sar_processor import GPU_AVAILABLE
from data_sources import UPLOADS_DIR
from routes_data import router as data_router
from routes_processing import router as processing_router
from routes_agent import router as agent_router

app = FastAPI(
    title="SAR Processing API",
    description="GPU-accelerated SAR image formation with on-demand data loading",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include route modules
app.include_router(data_router)
app.include_router(processing_router)
app.include_router(agent_router)


@app.on_event("startup")
def ensure_uploads_dir():
    """Create uploads directory if it doesn't exist."""
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/api/health")
def health_check():
    """Health check endpoint with GPU status."""
    return {
        "status": "healthy",
        "gpu_available": GPU_AVAILABLE,
        "gpu_backend": "cupy" if GPU_AVAILABLE else "numpy (CPU fallback)",
    }
