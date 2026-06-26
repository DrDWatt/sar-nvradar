"""
SAR image quality metrics for auto-enhancement agent.
Computes SNR, contrast, sharpness, and entropy to evaluate
algorithm output quality.
"""

import numpy as np


def compute_snr(image: np.ndarray) -> float:
    """Estimate Signal-to-Noise Ratio in dB.

    Uses the ratio of mean signal power to noise variance.
    For SAR images, noise is estimated from the darkest 25% of pixels.
    """
    flat = image.astype(np.float64).flatten()
    threshold = np.percentile(flat, 25)
    noise_region = flat[flat <= threshold]
    signal_region = flat[flat > threshold]

    if len(noise_region) == 0 or len(signal_region) == 0:
        return 0.0

    noise_power = np.var(noise_region)
    signal_power = np.mean(signal_region ** 2)

    if noise_power < 1e-10:
        return 60.0  # Cap at 60 dB for near-zero noise

    snr = 10.0 * np.log10(signal_power / noise_power)
    return float(np.clip(snr, -20.0, 60.0))


def compute_contrast(image: np.ndarray) -> float:
    """Michelson contrast ratio (0-1 scale).

    Higher values mean better dynamic range between bright targets
    and dark background — desirable for SAR target discrimination.
    """
    flat = image.astype(np.float64).flatten()
    i_max = np.percentile(flat, 99)  # Robust max (ignore outliers)
    i_min = np.percentile(flat, 1)   # Robust min

    if (i_max + i_min) < 1e-10:
        return 0.0

    contrast = (i_max - i_min) / (i_max + i_min)
    return float(np.clip(contrast, 0.0, 1.0))


def compute_sharpness(image: np.ndarray) -> float:
    """Edge sharpness via Laplacian variance.

    Higher values indicate sharper edges and better spatial resolution.
    Normalized to 0-100 scale for readability.
    """
    img = image.astype(np.float64)

    # Laplacian kernel
    laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)

    # Manual 2D convolution (avoid scipy dependency for this simple op)
    from scipy.ndimage import convolve
    lap_img = convolve(img, laplacian)

    variance = np.var(lap_img)
    # Normalize: typical SAR images have laplacian variance 0-5000
    normalized = min(variance / 50.0, 100.0)
    return float(normalized)


def compute_entropy(image: np.ndarray) -> float:
    """Shannon entropy of pixel intensity distribution (bits).

    Higher entropy means more information content.
    Typical range: 0-8 bits for uint8 images.
    """
    flat = image.astype(np.uint8).flatten()
    hist, _ = np.histogram(flat, bins=256, range=(0, 256))
    probs = hist / hist.sum()
    probs = probs[probs > 0]

    entropy = -np.sum(probs * np.log2(probs))
    return float(entropy)


def compute_all_metrics(image: np.ndarray) -> dict:
    """Compute all quality metrics for a SAR image.

    Returns dict with snr_db, contrast, sharpness, entropy, and composite_score.
    """
    snr = compute_snr(image)
    contrast = compute_contrast(image)
    sharpness = compute_sharpness(image)
    entropy = compute_entropy(image)

    # Composite score: weighted combination (0-100 scale)
    # SNR: 30% weight (normalize from -20..60 dB to 0-1)
    # Contrast: 25% weight (already 0-1)
    # Sharpness: 25% weight (already 0-100, normalize to 0-1)
    # Entropy: 20% weight (normalize from 0-8 to 0-1)
    snr_norm = np.clip((snr + 20) / 80.0, 0, 1)
    sharpness_norm = np.clip(sharpness / 100.0, 0, 1)
    entropy_norm = np.clip(entropy / 8.0, 0, 1)

    composite = (
        0.30 * snr_norm +
        0.25 * contrast +
        0.25 * sharpness_norm +
        0.20 * entropy_norm
    ) * 100.0

    return {
        "snr_db": round(snr, 2),
        "contrast": round(contrast, 4),
        "sharpness": round(sharpness, 2),
        "entropy_bits": round(entropy, 3),
        "composite_score": round(composite, 2),
    }
