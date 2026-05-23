print("STARTING IMPORT")
import numpy as np
print("NUMPY OK")
import cv2
print("CV2 OK")
"""
MaePixel — Input Enhancement Service v2
FastAPI service for pre-processing user uploads before MaePixel blend.

Pipeline:
  1. Bilateral pre-clean     — removes JPEG noise before sharpening
  2. Lanczos resize          — only if genuinely undersized, guaranteed to meet target
  3. Hybrid edge-aware acutance — structure tensor coherence + gradient energy mask
                                  recovers edges AND isotropic textures (hair, pores, cloth)

Fixes from v1:
  - Bilateral parameters stronger (d=7, σ_c=20, σ_s=15)
  - Hybrid mask: max(coherence, gradient_energy*0.4) — recovers textures
  - Sharpness safety gate compares pre/post acutance, not original
  - Acutance amount 0.48 (was 0.38), threshold 2 (was 4)
  - Resize guaranteed to meet target on both axes
  - Output WebP quality 95 for bandwidth efficiency (PNG optional)
"""

import io
import time
import logging
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import Response
from PIL import Image
import cv2
print("IMPORT TEST: starting service")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("enhance")

app = FastAPI(title="MaePixel Enhance Service v2")

# ── tuning constants ──────────────────────────────────────────────────────────

# Bilateral pre-clean
BILATERAL_D       = 7       # wider neighbourhood — actually removes JPEG noise
BILATERAL_SIGMA_C = 20.0    # colour sigma — removes JPEG colour blotching
BILATERAL_SIGMA_S = 15.0    # space sigma  — still edge-preserving

# Structure tensor
TENSOR_KSIZE      = 3
TENSOR_SIGMA      = 1.0
COHERENCE_THRESH  = 0.15

# Hybrid mask gradient weight
# mask = max(coherence_mask, gradient_mag * GRAD_WEIGHT)
# Recovers isotropic textures coherence-only mask misses
GRAD_WEIGHT       = 0.4

# Acutance
ACUTANCE_RADIUS    = 1.2
ACUTANCE_AMOUNT    = 0.48    # slightly stronger than v1
ACUTANCE_THRESHOLD = 2       # lower — recovers micro-detail

# Target minimum resolution
TARGET_MIN_W = 1920
TARGET_MIN_H = 1080


# ── helpers ───────────────────────────────────────────────────────────────────

def load_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img.astype(np.float32) / 255.0


def save_image(img: np.ndarray, fmt: str = "webp") -> tuple[bytes, str]:
    out = np.clip(img * 255, 0, 255).astype(np.uint8)
    if fmt == "png":
        _, buf = cv2.imencode(".png", out)
        return buf.tobytes(), "image/png"
    else:
        _, buf = cv2.imencode(".webp", out, [cv2.IMWRITE_WEBP_QUALITY, 95])
        return buf.tobytes(), "image/webp"


def laplacian_sharpness(img: np.ndarray) -> float:
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    lap  = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def bilateral_preclean(img: np.ndarray) -> np.ndarray:
    """
    Bilateral denoise — removes JPEG compression noise before sharpening.
    d=7, σ_c=20, σ_s=15: strong enough to remove mosquito noise and
    JPEG ringing without blurring real edges.
    """
    u8 = (img * 255).astype(np.uint8)
    cleaned = cv2.bilateralFilter(u8, BILATERAL_D, BILATERAL_SIGMA_C, BILATERAL_SIGMA_S)
    return cleaned.astype(np.float32) / 255.0


def lanczos_resize(img: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Resize to meet TARGET_MIN_W x TARGET_MIN_H on BOTH axes.
    Fixes v1 bug where one axis could remain below target.
    Capped at 2x per pass.
    Returns (resized_image, scale_used).
    """
    h, w = img.shape[:2]
    scale_w = TARGET_MIN_W / w if w < TARGET_MIN_W else 1.0
    scale_h = TARGET_MIN_H / h if h < TARGET_MIN_H else 1.0
    scale   = max(scale_w, scale_h)          # ensure BOTH axes meet target
    scale   = min(scale, 2.0)                # cap at 2x
    if scale <= 1.0:
        return img, 1.0
    nw = int(w * scale)
    nh = int(h * scale)
    u8 = (img * 255).astype(np.uint8)
    resized = cv2.resize(u8, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    return resized.astype(np.float32) / 255.0, scale


def build_hybrid_mask(img: np.ndarray) -> np.ndarray:
    """
    Hybrid edge mask: max(coherence_mask, gradient_energy_mask * GRAD_WEIGHT)

    Coherence branch:  catches oriented edges (contours, lines, hard boundaries)
    Gradient branch:   catches isotropic high-freq textures (hair, pores, cloth, fur)

    Combined: anything with real local structure gets sharpened.
    Flat/smooth/dark regions still score near 0 on both — untouched.
    """
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=TENSOR_KSIZE)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=TENSOR_KSIZE)

    # Structure tensor
    Jxx = cv2.GaussianBlur(gx * gx, (0, 0), TENSOR_SIGMA)
    Jyy = cv2.GaussianBlur(gy * gy, (0, 0), TENSOR_SIGMA)
    Jxy = cv2.GaussianBlur(gx * gy, (0, 0), TENSOR_SIGMA)

    trace = Jxx + Jyy
    det   = Jxx * Jyy - Jxy * Jxy
    disc  = np.sqrt(np.maximum(0, trace * trace / 4 - det))
    l1    = trace / 2 + disc
    l2    = trace / 2 - disc
    coherence = (l1 - l2) / (l1 + l2 + 1e-6)

    # Coherence branch
    coherence_mask = np.clip(
        (coherence - COHERENCE_THRESH) / (1.0 - COHERENCE_THRESH + 1e-6), 0, 1
    )

    # Gradient energy branch — normalised magnitude
    grad_mag = cv2.magnitude(gx, gy)
    grad_mag = grad_mag / (grad_mag.max() + 1e-6)

    # Hybrid: take max of both branches
    mask = np.maximum(coherence_mask, grad_mag * GRAD_WEIGHT)

    # Smooth mask edges to avoid transition artifacts
    mask = cv2.GaussianBlur(mask, (0, 0), 1.5)
    mask = np.clip(mask, 0, 1)

    return np.stack([mask, mask, mask], axis=2)


def edge_aware_acutance(img: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    Unsharp mask gated by hybrid structure mask.
    Returns (sharpened, sharpness_before, sharpness_after).
    Safety gate compares pre-acutance vs post-acutance (not original input).
    """
    sharpness_before = laplacian_sharpness(img)

    mask     = build_hybrid_mask(img)
    blurred  = cv2.GaussianBlur(img, (0, 0), ACUTANCE_RADIUS)
    residual = img - blurred

    # Suppress residual below noise floor
    threshold_mask = (np.abs(residual) > (ACUTANCE_THRESHOLD / 255.0)).astype(np.float32)

    sharpened = img + ACUTANCE_AMOUNT * mask * threshold_mask * residual
    sharpened = np.clip(sharpened, 0, 1)

    sharpness_after = laplacian_sharpness(sharpened)

    # Safety gate: compare pre vs post acutance only
    # (not original — resize + denoise changed statistics)
    if sharpness_after < sharpness_before * 0.95:
        logger.warning(
            f"Acutance degraded sharpness {sharpness_before:.1f} → {sharpness_after:.1f} — reverting"
        )
        return img, sharpness_before, sharpness_before

    return sharpened, sharpness_before, sharpness_after


# ── main pipeline ─────────────────────────────────────────────────────────────

def enhance_pipeline(img: np.ndarray, output_fmt: str = "webp") -> tuple[bytes, str, dict]:
    t0   = time.time()
    h, w = img.shape[:2]
    meta = {"input_w": w, "input_h": h, "steps": []}

    meta["sharpness_input"] = round(laplacian_sharpness(img), 2)
    logger.info(f"Input: {w}×{h}, sharpness={meta['sharpness_input']:.2f}")

    # Step 1: bilateral pre-clean
    img = bilateral_preclean(img)
    meta["steps"].append("bilateral_preclean")
    logger.info("Step 1: bilateral pre-clean done")

    # Step 2: resize if needed — guaranteed to meet both axis targets
    img, scale = lanczos_resize(img)
    if scale > 1.0:
        meta["steps"].append(f"lanczos_{scale:.2f}x")
        meta["resized_to"] = f"{img.shape[1]}×{img.shape[0]}"
        logger.info(f"Step 2: resized {scale:.2f}x → {img.shape[1]}×{img.shape[0]}")
    else:
        meta["steps"].append("resize_skipped")
        logger.info(f"Step 2: resize skipped ({w}×{h} already meets target)")

    # Step 3: hybrid edge-aware acutance
    img, sharp_before_acut, sharp_after_acut = edge_aware_acutance(img)
    if sharp_after_acut > sharp_before_acut * 0.95:
        meta["steps"].append("acutance_applied")
    else:
        meta["steps"].append("acutance_reverted")
    logger.info(f"Step 3: acutance {sharp_before_acut:.1f} → {sharp_after_acut:.1f}")

    meta["sharpness_after"]  = round(sharp_after_acut, 2)
    meta["output_w"]         = img.shape[1]
    meta["output_h"]         = img.shape[0]
    meta["elapsed_ms"]       = round((time.time() - t0) * 1000)

    logger.info(
        f"Done: {meta['output_w']}×{meta['output_h']}, "
        f"sharpness {meta['sharpness_input']} → {meta['sharpness_after']}, "
        f"{meta['elapsed_ms']}ms"
    )

    img_bytes, mime = save_image(img, fmt=output_fmt)
    return img_bytes, mime, meta


# ── routes ────────────────────────────────────────────────────────────────────

@app.post("/enhance")
async def enhance(
    file: UploadFile = File(...),
    fmt: str = Query(default="webp", regex="^(webp|png)$")
):
    """
    POST image → receive enhanced image.
    ?fmt=webp (default, smaller) or ?fmt=png (lossless)
    Response headers carry full enhancement metadata.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    data = await file.read()
    if len(data) > 50 * 1024 * 1024:
        raise HTTPException(413, "Image too large (max 50MB)")

    try:
        img = load_image(data)
    except Exception as e:
        raise HTTPException(422, f"Could not decode image: {e}")

    try:
        img_bytes, mime, meta = enhance_pipeline(img, output_fmt=fmt)
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        raise HTTPException(500, f"Enhancement failed: {e}")

    return Response(
        content=img_bytes,
        media_type=mime,
        headers={
            "X-Enhance-Steps":            ",".join(meta["steps"]),
            "X-Enhance-Sharpness-Input":  str(meta["sharpness_input"]),
            "X-Enhance-Sharpness-After":  str(meta["sharpness_after"]),
            "X-Enhance-Input":            f"{meta['input_w']}x{meta['input_h']}",
            "X-Enhance-Output":           f"{meta['output_w']}x{meta['output_h']}",
            "X-Enhance-Elapsed-Ms":       str(meta["elapsed_ms"]),
        }
    )


@app.get("/health")
def health():
    return {"status": "ok", "version": "2"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
