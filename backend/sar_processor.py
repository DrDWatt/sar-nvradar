"""
GPU-accelerated SAR image formation using CuPy.
Implements algorithms ported from NVRadar Holoscan operators:
- BackprojectionImageFormationOp (range-profile interpolation BP)
- PFAImageFormationOp (Polar Format Algorithm)
- Range-Doppler (windowed 2D FFT)
"""

import numpy as np

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = np
    GPU_AVAILABLE = False

import scipy.io as sio
from pathlib import Path

# NVRadar common constant
SPEED_OF_LIGHT = 299792458.0


def _get_xp():
    """Return the array module (cupy or numpy)."""
    return cp if GPU_AVAILABLE else np


def _to_uint8_image(complex_image, dynamic_range_db=40.0):
    """Convert complex SAR image to uint8 using NVRadar SARVisualizationOp logic.

    Uses log10(abs(image)) normalization matching NVRadar's visualization operator.
    """
    xp = _get_xp()
    magnitude = xp.abs(complex_image)
    magnitude = xp.where(magnitude > 0, magnitude, xp.finfo(xp.float32).tiny)
    log_mag = xp.log10(magnitude)

    # NVRadar normalization: subtract floor, divide by max
    floor_val = xp.max(log_mag) - dynamic_range_db / 10.0
    log_mag = xp.clip(log_mag - floor_val, 0, None)
    max_val = xp.max(log_mag)
    if float(max_val) > 0:
        normalized = log_mag / max_val
    else:
        normalized = xp.zeros_like(log_mag)

    result = (normalized * 255).astype(xp.uint8)
    if GPU_AVAILABLE:
        return result.get()
    return result


def load_target_disc_mat(filepath: str) -> dict:
    """Load target discrimination .mat file (SPIE challenge format).

    Returns dict with: fq (phase history K×Np), freq (K,),
    x, y, z, r0, th, phi (Np,)
    """
    raw = sio.loadmat(filepath)
    d = raw["data"]
    result = {}
    for name in d.dtype.names:
        val = d[name][0][0]
        if val.ndim == 2 and val.shape[0] == 1:
            result[name] = val.flatten()
        else:
            result[name] = val
    return result


def load_gmti_phase_history(ph_file: str, aux_file: str) -> dict:
    """Load GMTI phase history binary file with aux metadata.

    Binary format: big-endian interleaved float32 pairs (I,Q),
    384 range bins per pulse.
    """
    aux_data = sio.loadmat(aux_file)
    aux = aux_data["aux_SPIE"]

    fc = float(aux["fc"][0][0].flatten()[0])
    bw = float(aux["BW"][0][0].flatten()[0])

    read_prm = aux["readPrm"][0][0]
    num_pulses = int(read_prm["numPulses"][0][0].flatten()[0])
    num_samples = int(read_prm["numSamples"][0][0].flatten()[0])
    start_pulse = int(read_prm["startPulse"][0][0].flatten()[0])

    max_pulses = min(num_pulses, 5000)

    with open(ph_file, "rb") as f:
        f.seek((start_pulse - 1) * num_samples * 8)
        raw = np.fromfile(f, dtype=">f4", count=max_pulses * num_samples * 2)

    raw = raw.reshape(max_pulses, num_samples * 2)
    phase_history = raw[:, 0::2] + 1j * raw[:, 1::2]

    return {
        "phase_history": phase_history.astype(np.complex64),
        "fc": fc,
        "bw": bw,
        "num_pulses": max_pulses,
        "num_samples": num_samples,
    }


def _select_aperture(data: dict, aperture_deg: float = 10.0) -> dict:
    """Select a contiguous sub-aperture from the full 360° collection.

    Matches AFRL RunScript.m default of using a small angular aperture
    centered on the mid-collection point. This keeps processing time
    reasonable and matches spotlight SAR assumptions.
    """
    x = data["x"].astype(np.float64)
    y = data["y"].astype(np.float64)

    # Compute azimuth angle for each pulse
    az = np.unwrap(np.arctan2(y, x))
    total_az_deg = np.degrees(az.max() - az.min())

    # If already a small aperture, return as-is
    if total_az_deg <= aperture_deg * 1.5:
        return data

    # Select centered sub-aperture
    mid_az = (az.max() + az.min()) / 2.0
    half_ap = np.radians(aperture_deg / 2.0)
    mask = (az >= mid_az - half_ap) & (az <= mid_az + half_ap)
    indices = np.where(mask)[0]

    if len(indices) < 10:
        # Fallback: use first N pulses corresponding to aperture_deg
        pulses_per_deg = len(x) / total_az_deg
        n = max(int(pulses_per_deg * aperture_deg), 50)
        n = min(n, len(x))
        mid = len(x) // 2
        start = max(0, mid - n // 2)
        indices = np.arange(start, min(start + n, len(x)))

    result = dict(data)
    result["fq"] = data["fq"][:, indices]
    for key in ["x", "y", "z", "r0"]:
        result[key] = data[key][indices]
    if "th" in data:
        result["th"] = data["th"][indices]
    if "phi" in data:
        result["phi"] = data["phi"][indices]
    return result


def form_range_doppler_image(phase_history: np.ndarray) -> np.ndarray:
    """Form a basic range-Doppler map via windowed 2D FFT.

    This is the 'original' unprocessed view — no geometric correction.
    """
    xp = _get_xp()
    ph = xp.asarray(phase_history)

    win_range = xp.hamming(ph.shape[1]).astype(xp.float32)
    win_azimuth = xp.hamming(ph.shape[0]).astype(xp.float32)
    ph = ph * win_range[xp.newaxis, :] * win_azimuth[:, xp.newaxis]

    rd_image = xp.fft.fftshift(xp.fft.fft2(ph))
    return _to_uint8_image(rd_image)


def form_backprojection_image(data: dict, image_size: int = 160) -> np.ndarray:
    """GPU-accelerated backprojection following NVRadar BackprojectionImageFormationOp.

    Algorithm (matching NVRadar + AFRL bpBasic_v2.m reference):
    1. For each pulse, form oversampled range profile via IFFT
    2. Compute differential range from each pixel to antenna phase center
    3. Interpolate range profile at differential range
    4. Apply matched-filter phase correction: exp(j*4*pi*minF/c * dR)
    5. Coherently accumulate across pulses
    """
    xp = _get_xp()

    # Select sub-aperture (AFRL default: 5-10 degrees from full 360°)
    data = _select_aperture(data, aperture_deg=10.0)

    # Load data arrays to GPU
    # phdata: (K freqs, Np pulses) — freq-domain phase history
    phdata = xp.asarray(data["fq"].astype(np.complex128))
    freq = data["freq"].astype(np.float64)
    ant_x = xp.asarray(data["x"].astype(np.float64))
    ant_y = xp.asarray(data["y"].astype(np.float64))
    ant_z = xp.asarray(data["z"].astype(np.float64))
    r0 = xp.asarray(data["r0"].astype(np.float64))

    K = phdata.shape[0]     # number of frequency bins
    Np = phdata.shape[1]    # number of pulses

    delta_f = float(freq[1] - freq[0]) if K > 1 else 1.0
    min_f = float(freq[0])

    # FFT oversampling factor (matching AFRL default Nfact=4)
    Nfact = 4
    Nfft = Nfact * K

    # Maximum range swath and range bin spacing
    max_wr = SPEED_OF_LIGHT / (2.0 * delta_f)
    dr = max_wr / Nfft

    # Range resolution and cross-range resolution
    range_res = SPEED_OF_LIGHT / (2.0 * delta_f * K)

    # Azimuth geometry for cross-range resolution
    ant_az = xp.unwrap(xp.arctan2(ant_y, ant_x))
    delta_az = float(xp.abs(xp.mean(xp.diff(ant_az))))
    total_az = float(xp.max(ant_az) - xp.min(ant_az))
    cross_range_res = SPEED_OF_LIGHT / (2.0 * total_az * min_f) if total_az > 0 else range_res

    # Range vector for interpolation
    r_vec = xp.linspace(-Nfft / 2, Nfft / 2 - 1, Nfft) * max_wr / Nfft

    # Build image grid (scene_size × scene_size meters, centered at origin)
    scene_size = 10.0  # meters (AFRL default)
    pixel_spacing = scene_size / (image_size - 1)
    grid_1d = xp.linspace(-scene_size / 2, scene_size / 2, image_size)
    x_mat, y_mat = xp.meshgrid(grid_1d, grid_1d)
    z_mat = xp.zeros_like(x_mat)

    # Coherent image accumulator
    im_final = xp.zeros((image_size, image_size), dtype=xp.complex128)

    # Per-pulse backprojection loop
    r_min = float(r_vec[0])
    r_max = float(r_vec[-1])

    for ii in range(Np):
        # Form range profile: zero-padded IFFT (NVRadar DataConditionerOp)
        rc = xp.fft.fftshift(xp.fft.ifft(phdata[:, ii], n=Nfft))

        # Differential range to each pixel
        dR = xp.sqrt(
            (ant_x[ii] - x_mat) ** 2
            + (ant_y[ii] - y_mat) ** 2
            + (ant_z[ii] - z_mat) ** 2
        ) - r0[ii]

        # Phase correction (matched filter)
        ph_corr = xp.exp(1j * 4.0 * xp.pi * min_f / SPEED_OF_LIGHT * dR)

        # Find pixels within range swath
        valid = (dR > r_min) & (dR < r_max)

        if xp.any(valid):
            # Linear interpolation of range profile at differential range
            # Map dR to fractional bin index
            bin_idx = (dR[valid] - r_min) / (r_max - r_min) * (Nfft - 1)
            b1 = xp.floor(bin_idx).astype(xp.int64)
            b1 = xp.clip(b1, 0, Nfft - 2)
            w = bin_idx - b1.astype(xp.float64)
            interp_vals = rc[b1] * (1.0 - w) + rc[b1 + 1] * w

            # Accumulate with phase correction
            contribution = xp.zeros_like(im_final)
            contribution[valid] = interp_vals * ph_corr[valid]
            im_final += contribution

    return _to_uint8_image(im_final)


def form_pfa_image(data: dict, image_size: int = 256) -> np.ndarray:
    """GPU-accelerated Polar Format Algorithm from NVRadar PFAImageFormationOp.

    Ported directly from NVRadar custom_operators.py PFAImageFormationOp.compute().
    Resamples from polar k-space to Cartesian grid, then 2D FFT.
    """
    xp = _get_xp()

    # Select sub-aperture for spotlight-mode PFA
    data = _select_aperture(data, aperture_deg=10.0)

    # phdata: (K, Np) freq-domain
    phdata = xp.asarray(data["fq"].astype(np.complex64))
    freq = data["freq"].astype(np.float64)

    K = phdata.shape[0]
    Np = phdata.shape[1]

    # Platform positions (Np, 3)
    platform_pos = xp.stack([
        xp.asarray(data["x"].astype(np.float64)),
        xp.asarray(data["y"].astype(np.float64)),
        xp.asarray(data["z"].astype(np.float64)),
    ], axis=1)

    delta_f = float(freq[1] - freq[0]) if K > 1 else 1.0
    min_f = float(freq[0])
    f0 = float(min_f + delta_f * K / 2)
    freqs = xp.asarray(np.linspace(min_f, min_f + delta_f * K, K, endpoint=False))

    # Compute pixel spacing from bandwidth
    bandwidth = delta_f * (K - 1)
    range_res = SPEED_OF_LIGHT / (2.0 * bandwidth) if bandwidth > 0 else 0.5
    pixel_spacing = range_res

    nu = image_size
    nv = image_size
    du = pixel_spacing
    dv = du

    n_hat = xp.array([0.0, 0.0, 1.0])
    R_c = platform_pos[Np // 2]

    # NVRadar PFA: compute image plane basis vectors
    v_hat = xp.cross(n_hat, R_c)
    v_norm = xp.linalg.norm(v_hat)
    if float(v_norm) > 1e-10:
        v_hat = v_hat / v_norm
    else:
        v_hat = xp.array([0.0, 1.0, 0.0])

    u_hat = xp.cross(v_hat, n_hat)
    u_norm = xp.linalg.norm(u_hat)
    if float(u_norm) > 1e-10:
        u_hat = u_hat / u_norm
    else:
        u_hat = xp.array([1.0, 0.0, 0.0])

    # Recompute u_hat as in NVRadar source (projection onto ground plane)
    u_hat = R_c - xp.dot(R_c, n_hat) * n_hat
    u_norm = xp.linalg.norm(u_hat)
    if float(u_norm) > 1e-10:
        u_hat = u_hat / u_norm
    v_hat = xp.cross(u_hat, n_hat)

    # Grazing angle
    psi = xp.pi / 2 - xp.arccos(
        xp.dot(R_c, n_hat) / max(float(xp.linalg.norm(R_c)), 1e-10)
    )

    # Output k-space grids
    kui = 2 * xp.pi * xp.linspace(-1.0 / (2 * du), 1.0 / (2 * du), nu)
    kvi = 2 * xp.pi * xp.linspace(-1.0 / (2 * dv), 1.0 / (2 * dv), nv)
    kui = kui + 4 * xp.pi * f0 / SPEED_OF_LIGHT * xp.cos(psi)

    # Per-pulse direction vectors
    r_norm = xp.linalg.norm(platform_pos, axis=1)
    r_hat = platform_pos / r_norm[:, None]

    # Wavenumber array (NVRadar: K = 4*pi/c * freq)
    K_wave = 4 * xp.pi / SPEED_OF_LIGHT * freqs

    # Spatial frequency coordinates: ku, kv per pulse per freq
    # NVRadar: ku = (r_hat . u_hat) * K, kv = (r_hat . v_hat) * K
    ku = xp.matmul(xp.matmul(r_hat, u_hat[:, None]), K_wave[None, :])  # (Np, K)
    kv = xp.matmul(xp.matmul(r_hat, v_hat[:, None]), K_wave[None, :])  # (Np, K)

    # Phase history transposed for indexing: phdata[:, i] = freq samples for pulse i
    # Interpolate from polar to Cartesian k-space
    rad_interp = xp.zeros((Np, nu), dtype=xp.complex64)
    ky_new = xp.zeros((Np, nu), dtype=xp.float64)

    for i in range(Np):
        rad_interp[i, :] = xp.interp(kui, ku[i, :], phdata[:, i], left=0, right=0)
        ky_new[i, :] = xp.interp(kui, ku[i, :], kv[i, :], left=0, right=0)

    # Azimuth interpolation (NVRadar PFA logic)
    polar = xp.zeros((nv, nu), dtype=xp.complex64)
    mid = Np // 2
    is_sorted = bool(ky_new[mid, nu // 2] < ky_new[mid + 1, nu // 2])

    if is_sorted:
        for i in range(nu):
            polar[:, i] = xp.interp(kvi, ky_new[:, i], rad_interp[:, i], left=0, right=0)
    else:
        for i in range(nu):
            col = ky_new[::-1, i]
            if float(xp.max(col)) == 0:
                continue
            nz = xp.nonzero(col)[0]
            if len(nz) > 0:
                first = int(nz[0])
                vals = rad_interp[::-1, i]
                tmp1 = col[first:int(nz[-1]) + 1]
                tmp2 = vals[first:first + len(tmp1)]
                polar[:, i] = xp.interp(kvi, tmp1, tmp2, left=0, right=0)

    polar = xp.nan_to_num(polar)

    # 2D FFT to form image (NVRadar: fftshift(fft2(fftshift(polar))))
    image = xp.fft.fftshift(xp.fft.fft2(xp.fft.fftshift(polar)))

    return _to_uint8_image(image)


def load_geotiff_sar(filepath: str) -> np.ndarray:
    """Load a GeoTIFF SAR image file.

    Handles single-band and multi-band (VV/VH) GeoTIFFs,
    as well as complex-valued SLC products.
    Returns a 2D float32 array.
    """
    import tifffile

    img = tifffile.imread(filepath)

    # Handle complex data (SLC products)
    if np.iscomplexobj(img):
        img = np.abs(img).astype(np.float32)

    # Handle multi-band (take first band)
    if img.ndim == 3:
        img = img[0].astype(np.float32)
    elif img.ndim == 2:
        img = img.astype(np.float32)
    else:
        raise ValueError(f"Unexpected image dimensions: {img.ndim}")

    return img


def _lee_filter(image: np.ndarray, window_size: int = 7) -> np.ndarray:
    """Lee speckle filter for SAR imagery.

    Vectorized implementation using uniform filter for local statistics.
    Reduces speckle while preserving edges.
    """
    from scipy.ndimage import uniform_filter

    img = image.astype(np.float64)
    img = np.maximum(img, 0)

    # Local mean and variance via uniform filter
    local_mean = uniform_filter(img, size=window_size)
    local_sq_mean = uniform_filter(img ** 2, size=window_size)
    local_var = np.maximum(local_sq_mean - local_mean ** 2, 0)

    # Overall noise variance estimate
    noise_var = np.mean(local_var)

    # Lee filter weight: W = max(0, 1 - noise_var / local_var)
    weight = np.where(
        local_var > 0,
        np.clip(1.0 - noise_var / local_var, 0, 1),
        0,
    )

    result = local_mean + weight * (img - local_mean)
    return result.astype(np.float32)


def _histogram_equalize(image: np.ndarray) -> np.ndarray:
    """Apply histogram equalization to a uint8 image for contrast enhancement."""
    hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
    cdf = hist.cumsum()
    cdf_min = cdf[cdf > 0].min()
    total = cdf[-1]

    if total - cdf_min > 0:
        cdf_norm = ((cdf - cdf_min) * 255 / (total - cdf_min)).astype(np.uint8)
    else:
        cdf_norm = np.zeros(256, dtype=np.uint8)

    return cdf_norm[image.astype(np.uint8)]


def _to_uint8_log(image: np.ndarray, dynamic_range_db: float = 40.0) -> np.ndarray:
    """Convert float SAR image to uint8 with log-scale normalization."""
    magnitude = np.abs(image)
    magnitude = np.where(magnitude > 0, magnitude, np.finfo(np.float32).tiny)
    log_mag = np.log10(magnitude)

    floor_val = np.max(log_mag) - dynamic_range_db / 10.0
    log_mag = np.clip(log_mag - floor_val, 0, None)
    max_val = np.max(log_mag)
    if max_val > 0:
        normalized = log_mag / max_val
    else:
        normalized = np.zeros_like(log_mag)

    return (normalized * 255).astype(np.uint8)


def enhance_geotiff_sar(image: np.ndarray, method: str = "lee") -> np.ndarray:
    """Enhance a pre-formed SAR GeoTIFF image.

    Methods:
    - lee: Lee speckle filter + log display
    - histogram_eq: Log display + histogram equalization
    - adaptive: Lee filter + histogram equalization (best overall)
    """
    if method == "lee":
        filtered = _lee_filter(image)
        return _to_uint8_log(filtered)
    elif method == "histogram_eq":
        log_img = _to_uint8_log(image)
        return _histogram_equalize(log_img)
    elif method == "adaptive":
        filtered = _lee_filter(image)
        log_img = _to_uint8_log(filtered)
        return _histogram_equalize(log_img)
    else:
        return _to_uint8_log(image)


def form_enhanced_gmti_image(data: dict, image_size: int = 512) -> np.ndarray:
    """Enhanced GMTI processing using NVRadar-style range-Doppler with
    zero-padding and Kaiser windowing for improved resolution.
    """
    xp = _get_xp()
    ph = xp.asarray(data["phase_history"])

    num_pulses, num_samples = ph.shape

    # Apply Kaiser window for better sidelobe control (NVRadar uses windowing)
    win_range = xp.asarray(np.kaiser(num_samples, 6.0).astype(np.float32))
    win_azimuth = xp.asarray(np.kaiser(num_pulses, 4.0).astype(np.float32))
    ph_windowed = ph * win_range[xp.newaxis, :] * win_azimuth[:, xp.newaxis]

    # Zero-pad for higher resolution (NVRadar DataConditioner oversample)
    pad_factor = 2
    padded = xp.zeros(
        (num_pulses * pad_factor, num_samples * pad_factor), dtype=xp.complex64
    )
    padded[:num_pulses, :num_samples] = ph_windowed

    # 2D FFT
    image = xp.fft.fftshift(xp.fft.fft2(padded))

    # Crop to image_size from center
    cy, cx = image.shape[0] // 2, image.shape[1] // 2
    half = image_size // 2
    cropped = image[cy - half : cy + half, cx - half : cx + half]

    return _to_uint8_image(cropped)
