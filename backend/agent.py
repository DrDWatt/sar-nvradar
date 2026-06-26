"""
NVRadar Auto-Enhancement Agent.
Runs all applicable algorithms, scores each result with image quality metrics,
and uses NVIDIA Nemotron LLM to analyze and recommend the best output.
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import httpx

from image_quality import compute_all_metrics
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

logger = logging.getLogger(__name__)

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
NVIDIA_NIM_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1"

# SAR domain expert system prompt
SAR_SYSTEM_PROMPT = """You are an expert SAR (Synthetic Aperture Radar) image processing analyst 
embedded in the NVIDIA NVRadar pipeline. Your role is to analyze image quality metrics from 
multiple processing algorithms and recommend the best one.

You understand:
- PFA (Polar Format Algorithm): Best for spotlight SAR with motion compensation. Produces 
  focused images from phase history data using polar-to-rectangular interpolation.
- Backprojection (BP): Most accurate image formation, handles arbitrary geometries. 
  Computationally intensive but produces the sharpest results for complex scenes.
- Enhanced GMTI: Specialized for ground moving target indication with clutter suppression.
- Lee Speckle Filter: Reduces multiplicative speckle noise while preserving edges.
- Histogram Equalization: Enhances contrast by flattening the intensity distribution.
- Adaptive Enhancement: Combines Lee filtering with histogram equalization for balanced results.

When analyzing metrics:
- SNR (dB): Higher is better. >20dB is good, >30dB is excellent for SAR.
- Contrast (0-1): Higher means better target-background separation.
- Sharpness (0-100): Higher means sharper edges and better spatial resolution.
- Entropy (bits): 5-7 bits is ideal — too low means information loss, too high may mean noise.
- Composite Score (0-100): Weighted overall quality measure.

Provide a concise recommendation (2-3 sentences) explaining which algorithm produced the best 
result and why, referencing specific metrics. Include the recommended algorithm name."""


def _get_applicable_algorithms(data_type: str) -> list:
    """Return list of algorithm configs applicable to the data type."""
    if data_type == "target_discrimination":
        return [
            {"id": "pfa", "name": "NVRadar Polar Format (PFA)"},
            {"id": "bp", "name": "NVRadar Backprojection (BP)"},
            {"id": "adaptive", "name": "GPU Adaptive Enhancement"},
            {"id": "lee", "name": "GPU Lee Speckle Filter"},
            {"id": "histogram_eq", "name": "GPU Histogram Equalization"},
        ]
    elif data_type == "gmti":
        return [
            {"id": "enhanced", "name": "NVRadar Enhanced GMTI"},
            {"id": "adaptive", "name": "GPU Adaptive Enhancement"},
            {"id": "lee", "name": "GPU Lee Speckle Filter"},
            {"id": "histogram_eq", "name": "GPU Histogram Equalization"},
        ]
    else:  # geotiff
        return [
            {"id": "adaptive", "name": "GPU Adaptive Enhancement"},
            {"id": "lee", "name": "GPU Lee Speckle Filter"},
            {"id": "histogram_eq", "name": "GPU Histogram Equalization"},
        ]


def _run_algorithm(data_type: str, data: dict, algorithm: str,
                   image_size: int = 512) -> Optional[np.ndarray]:
    """Run a single algorithm and return the result image array."""
    try:
        if data_type == "target_discrimination":
            if algorithm == "pfa":
                pfa_size = min(image_size, 256)
                return form_pfa_image(data, image_size=pfa_size)
            elif algorithm == "bp":
                bp_size = min(image_size, 160)
                return form_backprojection_image(data, image_size=bp_size)
            elif algorithm in ("adaptive", "lee", "histogram_eq"):
                # For enhancement algorithms on phase history, first form a basic image
                base_img = form_range_doppler_image(data["fq"].T)
                return enhance_geotiff_sar(base_img.astype(np.float32), method=algorithm)

        elif data_type == "gmti":
            if algorithm == "enhanced":
                return form_enhanced_gmti_image(data, image_size=image_size)
            elif algorithm in ("adaptive", "lee", "histogram_eq"):
                base_img = form_range_doppler_image(data["phase_history"])
                return enhance_geotiff_sar(base_img.astype(np.float32), method=algorithm)

        elif data_type == "geotiff":
            raw_image = data["raw_image"]
            if algorithm in ("adaptive", "lee", "histogram_eq"):
                return enhance_geotiff_sar(raw_image, method=algorithm)

    except Exception as e:
        logger.warning(f"Algorithm {algorithm} failed: {e}")
        return None

    return None


async def _llm_analyze(algorithm_results: list) -> str:
    """Use NVIDIA Nemotron LLM to analyze algorithm results and recommend the best one."""
    if not NVIDIA_API_KEY:
        return _fallback_analysis(algorithm_results)

    # Build the metrics summary for the LLM
    metrics_text = "Algorithm results for this SAR image:\n\n"
    for result in algorithm_results:
        m = result["metrics"]
        metrics_text += (
            f"Algorithm: {result['name']}\n"
            f"  SNR: {m['snr_db']} dB\n"
            f"  Contrast: {m['contrast']}\n"
            f"  Sharpness: {m['sharpness']}\n"
            f"  Entropy: {m['entropy_bits']} bits\n"
            f"  Composite Score: {m['composite_score']}/100\n"
            f"  Processing Time: {result['processing_time_ms']} ms\n\n"
        )

    metrics_text += "Which algorithm produced the best result and why? Be specific about the metrics."

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                NVIDIA_NIM_URL,
                headers={
                    "Authorization": f"Bearer {NVIDIA_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": NVIDIA_MODEL,
                    "messages": [
                        {"role": "system", "content": SAR_SYSTEM_PROMPT},
                        {"role": "user", "content": metrics_text},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 300,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        logger.warning(f"LLM analysis failed: {e}, using fallback")
        return _fallback_analysis(algorithm_results)


def _fallback_analysis(algorithm_results: list) -> str:
    """Rule-based fallback analysis when LLM is unavailable."""
    if not algorithm_results:
        return "No algorithms produced valid results."

    # Sort by composite score
    sorted_results = sorted(
        algorithm_results,
        key=lambda r: r["metrics"]["composite_score"],
        reverse=True,
    )

    best = sorted_results[0]
    m = best["metrics"]

    analysis = (
        f"Recommended: **{best['name']}** with composite score {m['composite_score']}/100. "
        f"This algorithm achieved SNR of {m['snr_db']} dB, "
        f"contrast of {m['contrast']:.3f}, "
        f"sharpness of {m['sharpness']:.1f}, "
        f"and entropy of {m['entropy_bits']:.2f} bits "
        f"in {best['processing_time_ms']} ms."
    )

    if len(sorted_results) > 1:
        runner_up = sorted_results[1]
        score_diff = best["metrics"]["composite_score"] - runner_up["metrics"]["composite_score"]
        analysis += (
            f" Runner-up: {runner_up['name']} "
            f"(score: {runner_up['metrics']['composite_score']}/100, "
            f"delta: {score_diff:.1f} pts)."
        )

    return analysis


async def auto_enhance(data_type: str, data: dict, image_size: int = 512) -> dict:
    """Run the full auto-enhancement agent pipeline.

    1. Runs all applicable algorithms
    2. Computes quality metrics for each
    3. Uses LLM to analyze and recommend
    4. Returns best result with analysis

    Returns dict with:
    - best_algorithm: str
    - best_image: np.ndarray
    - analysis: str (LLM recommendation text)
    - all_results: list of {algorithm, name, metrics, processing_time_ms}
    """
    algorithms = _get_applicable_algorithms(data_type)
    results = []

    for algo in algorithms:
        start = time.time()
        image = _run_algorithm(data_type, data, algo["id"], image_size)
        proc_time = time.time() - start

        if image is not None:
            metrics = compute_all_metrics(image)
            results.append({
                "algorithm": algo["id"],
                "name": algo["name"],
                "metrics": metrics,
                "processing_time_ms": round(proc_time * 1000, 1),
                "image": image,
            })

    if not results:
        return {
            "best_algorithm": None,
            "best_image": None,
            "analysis": "No algorithms produced valid results for this data.",
            "all_results": [],
        }

    # Get LLM analysis (or fallback)
    results_for_llm = [
        {k: v for k, v in r.items() if k != "image"}
        for r in results
    ]
    analysis = await _llm_analyze(results_for_llm)

    # Select best by composite score
    best = max(results, key=lambda r: r["metrics"]["composite_score"])

    # Build response (strip image arrays from all_results)
    all_results = [
        {
            "algorithm": r["algorithm"],
            "name": r["name"],
            "metrics": r["metrics"],
            "processing_time_ms": r["processing_time_ms"],
            "is_best": r["algorithm"] == best["algorithm"],
        }
        for r in results
    ]

    return {
        "best_algorithm": best["algorithm"],
        "best_algorithm_name": best["name"],
        "best_image": best["image"],
        "best_metrics": best["metrics"],
        "analysis": analysis,
        "all_results": all_results,
        "gpu_used": GPU_AVAILABLE,
        "llm_model": NVIDIA_MODEL if NVIDIA_API_KEY else "rule-based fallback",
    }
