# inference_utils.py
import os
import json
import numpy as np
import pandas as pd
import cv2

from config import (
    PATCH_SIZE, STRIDE,
    RGB_TO_THERMAL_ZOOM,
    THERM_P01, THERM_P99,
    CANOPY_THR
)

# =========================================================
# IO
# =========================================================
def load_rgb(rgb_path: str) -> np.ndarray:
    img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"RGB image cannot be read: {rgb_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def load_thermal(th_path: str):
    """
    Support:
    - .npy : expected 2D float array (often Celsius)
    - image (.png/.jpg): loaded as grayscale float (NOT Celsius)
    """
    ext = os.path.splitext(th_path.lower())[1]
    if ext == ".npy":
        arr = np.load(th_path)
        if arr.ndim == 3:
            arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise ValueError(f"Thermal .npy must be 2D. Got shape={arr.shape}")
        return arr.astype(np.float32), "celsius_or_float"
    else:
        im = cv2.imread(th_path, cv2.IMREAD_GRAYSCALE)
        if im is None:
            raise ValueError(f"Thermal image cannot be read: {th_path}")
        return im.astype(np.float32), "grayscale_raw"

# =========================================================
# Basic utils
# =========================================================
def percentile_minmax(x: np.ndarray, p01: float, p99: float):
    x = x.astype(np.float32)
    lo = np.nanpercentile(x, p01)
    hi = np.nanpercentile(x, p99)
    if not np.isfinite(lo): lo = 0.0
    if not np.isfinite(hi): hi = lo + 1e-6
    if hi <= lo:
        hi = lo + 1e-6
    xn = (x - lo) / (hi - lo)
    xn = np.clip(xn, 0.0, 1.0)
    return xn, float(lo), float(hi)

def center_crop(img, crop_h: int, crop_w: int):
    H, W = img.shape[:2]
    crop_h = min(crop_h, H)
    crop_w = min(crop_w, W)
    y0 = (H - crop_h) // 2
    x0 = (W - crop_w) // 2
    return img[y0:y0+crop_h, x0:x0+crop_w], (x0, y0, crop_w, crop_h)

def rgb_to_thermal_grid(rgb_rgb: np.ndarray, th_shape_hw, zoom: float):
    """
    RGB wide -> crop center (zoom-in) -> resize to thermal grid size.
    th_shape_hw : (Hth, Wth)
    Return: rgb_aligned_rgb, roi
    """
    Hth, Wth = th_shape_hw
    Hr, Wr = rgb_rgb.shape[:2]

    # guard
    if zoom is None or float(zoom) <= 0:
        zoom = 1.0

    crop_h = int(Hr / float(zoom))
    crop_w = int(Wr / float(zoom))
    rgb_crop, roi = center_crop(rgb_rgb, crop_h, crop_w)
    rgb_aligned = cv2.resize(rgb_crop, (Wth, Hth), interpolation=cv2.INTER_LINEAR)
    return rgb_aligned, roi

# =========================================================
# Canopy mask (RGB-based)
# =========================================================
def build_canopy_mask_full(rgb_img: np.ndarray) -> np.ndarray:
    """
    Return vegetation mask (H,W) uint8 {0,1} from RGB image (the image you will patch on).
    """
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)

    lower = np.array([35, 60, 60], dtype=np.uint8)
    upper = np.array([90, 255, 255], dtype=np.uint8)
    m_hsv = (cv2.inRange(hsv, lower, upper) > 0)

    R = rgb_img[..., 0].astype(np.int16)
    G = rgb_img[..., 1].astype(np.int16)
    B = rgb_img[..., 2].astype(np.int16)
    m_gdom = (G > R + 15) & (G > B + 15) & (G > 60)

    m = m_hsv & m_gdom
    return m.astype(np.uint8)

def keep_largest_component(mask01: np.ndarray, min_area: int = 1000) -> np.ndarray:
    mask = (mask01 > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = int(np.argmax(areas) + 1)
    if stats[idx, cv2.CC_STAT_AREA] < int(min_area):
        return np.zeros_like(mask)
    return (labels == idx).astype(np.uint8)

def canopy_frac_from_fullmask(main_mask01: np.ndarray, x: int, y: int, patch: int) -> float:
    roi = main_mask01[y:y+patch, x:x+patch]
    if roi.size == 0:
        return 0.0
    return float(np.mean(roi > 0))

# =========================================================
# Patch extraction
# =========================================================
def extract_grid_patches(rgb_img: np.ndarray, patch: int, stride: int):
    H, W = rgb_img.shape[:2]
    if H < patch or W < patch:
        raise ValueError(f"Image too small for patch={patch}: got {(H,W)}")

    rows = (H - patch) // stride + 1
    cols = (W - patch) // stride + 1

    patches, meta = [], []
    for r in range(rows):
        y = r * stride
        for c in range(cols):
            x = c * stride
            crop = rgb_img[y:y+patch, x:x+patch, :]
            patches.append(crop)
            meta.append((r, c, x, y))
    return np.stack(patches, axis=0), meta, rows, cols, H, W

def extract_grid_patches_thermal(th_arr: np.ndarray, patch: int, stride: int):
    """
    NO implicit resize here. Assume th_arr is already in the target grid.
    """
    H, W = th_arr.shape[:2]
    if H < patch or W < patch:
        raise ValueError(f"Thermal too small for patch={patch}: got {(H,W)}")

    rows = (H - patch) // stride + 1
    cols = (W - patch) // stride + 1

    patches, means = [], []
    for r in range(rows):
        y = r * stride
        for c in range(cols):
            x = c * stride
            crop = th_arr[y:y+patch, x:x+patch]
            patches.append(crop)
            means.append(float(np.nanmean(crop)))
    return np.stack(patches, axis=0), np.array(means, dtype=np.float32), rows, cols

# =========================================================
# Predict: RGB-only (FORCE thermal-like grid)
# =========================================================
def predict_grid_rgb_only(
    model_rgb,
    rgb_path: str,
    th_path_or_none: str = None,
    patch: int = PATCH_SIZE,
    stride: int = STRIDE,
    threshold: float = 0.5,
    p01: float = THERM_P01,
    p99: float = THERM_P99,
    zoom: float = RGB_TO_THERMAL_ZOOM,
    thermal_target_hw=(512, 640),  # fallback grid kalau thermal tidak ada
):
    """
    RGB-only inference tapi GRID DIPAKSA ke thermal-like grid:
    - crop+zoom+resize ke (H,W) target supaya patch/overlay konsisten dengan fusion.

    Jika thermal disediakan walaupun mode rgb_only:
    - temp_mean tetap dihitung untuk hover (opsional).
    """

    rgb = load_rgb(rgb_path)

    thermal_kind = None
    roi = None

    # -----------------------------
    # Tentukan target grid
    # -----------------------------
    th_used = None
    if th_path_or_none is not None and str(th_path_or_none).strip() != "":
        th_used, thermal_kind = load_thermal(th_path_or_none)
        target_hw = th_used.shape[:2]  # (H,W)
    else:
        target_hw = tuple(thermal_target_hw)

    # -----------------------------
    # Align RGB -> target grid (SAMA KAYAK FUSION)
    # -----------------------------
    rgb_used, roi = rgb_to_thermal_grid(rgb, target_hw, zoom=zoom)

    # -----------------------------
    # Canopy mask computed on aligned rgb (konsisten dengan fusion)
    # -----------------------------
    veg_mask = build_canopy_mask_full(rgb_used)               # 0/1
    main_mask = keep_largest_component(veg_mask, min_area=5000)

    # -----------------------------
    # Patch extraction pada rgb_used
    # -----------------------------
    rgb_patches, meta, rows, cols, H, W = extract_grid_patches(rgb_used, patch, stride)
    x_rgb = rgb_patches.astype(np.float32) / 255.0

    probs = model_rgb.predict(x_rgb, batch_size=64, verbose=0).reshape(-1)
    pred = (probs >= float(threshold)).astype(int)

    # -----------------------------
    # Kalau thermal ada: hitung temp_mean per patch (untuk hover)
    # -----------------------------
    temp_mean = None
    if th_used is not None:
        # Pastikan thermal size sama dengan rgb_used (harusnya sama karena target_hw = th.shape)
        if th_used.shape[0] != rgb_used.shape[0] or th_used.shape[1] != rgb_used.shape[1]:
            # guard: resize thermal ke grid (last resort)
            th_used_rs = cv2.resize(th_used, (rgb_used.shape[1], rgb_used.shape[0]), interpolation=cv2.INTER_NEAREST)
            th_used = th_used_rs.astype(np.float32)

        _, th_means, r2, c2 = extract_grid_patches_thermal(th_used, patch, stride)
        temp_mean = th_means
    else:
        temp_mean = None

    # -----------------------------
    # Records
    # -----------------------------
    records = []
    for i, (r, c, x, y) in enumerate(meta):
        cf = canopy_frac_from_fullmask(main_mask, int(x), int(y), int(patch))

        tmean = None
        if temp_mean is not None and i < len(temp_mean) and np.isfinite(temp_mean[i]):
            tmean = float(temp_mean[i])

        records.append({
            "patch_id": f"grid_{r}_{c}",
            "r": int(r), "c": int(c),
            "x": int(x), "y": int(y),
            "canopy_frac": float(cf),
            "temp_mean": tmean,
            "y_prob": float(probs[i]),
            "pred_stress": int(pred[i]),
        })

    df = pd.DataFrame(records)

    # -----------------------------
    # Summary (WAJIB aman untuk Jinja)
    # -----------------------------
    temp_mean_global = None
    if temp_mean is not None and len(temp_mean) > 0 and np.isfinite(np.nanmean(temp_mean)):
        temp_mean_global = float(np.nanmean(temp_mean))

    summary = {
        "mode": "rgb_only",
        "threshold": float(threshold),
        "grid_space": "thermal_grid_like",
        "align_zoom": float(zoom),
        "align_roi": roi,
        "num_patches": int(len(df)),
        "stress_count": int(df["pred_stress"].sum()) if len(df) else 0,
        "normal_count": int((df["pred_stress"] == 0).sum()) if len(df) else 0,
        "thermal_kind": thermal_kind,
        "temp_mean_global": temp_mean_global,   # <-- penting biar jinja ga error
        "img_w": int(W),
        "img_h": int(H),
        "grid_rows": int(rows),
        "grid_cols": int(cols),
        "patch": int(patch),
        "stride": int(stride),
    }

    return df, summary, rgb_used

# =========================================================
# Predict: Fusion (RGB + Thermal) — ALWAYS thermal grid
# =========================================================
def predict_grid_fusion(
    model_fusion,
    rgb_path: str,
    th_path: str,
    patch: int = PATCH_SIZE,
    stride: int = STRIDE,
    threshold: float = 0.5,
    p01: float = THERM_P01,
    p99: float = THERM_P99,
    zoom: float = RGB_TO_THERMAL_ZOOM,
):
    rgb = load_rgb(rgb_path)
    th, thermal_kind = load_thermal(th_path)

    # 1) Align RGB to thermal grid (Tahap 7)
    rgb_aligned, roi = rgb_to_thermal_grid(rgb, th.shape[:2], zoom=zoom)

    # 2) Canopy mask computed on aligned rgb
    veg_mask = build_canopy_mask_full(rgb_aligned)                  # 0/1
    main_mask = keep_largest_component(veg_mask, min_area=5000)      # focus field

    # 3) Normalize thermal
    th_norm, lo, hi = percentile_minmax(th, p01, p99)

    # 4) Extract paired patches
    Ht, Wt = th.shape[:2]
    rgb_patches, meta, rows, cols, _, _ = extract_grid_patches(rgb_aligned, patch, stride)
    _, th_means, _, _ = extract_grid_patches_thermal(th, patch, stride)
    th_patches_norm, _, _, _ = extract_grid_patches_thermal(th_norm, patch, stride)

    # 5) Build model inputs
    x_rgb = rgb_patches.astype(np.float32) / 255.0
    x_th  = th_patches_norm.astype(np.float32)[..., None]

    in_shapes = [tuple(inp.shape) for inp in model_fusion.inputs]
    n_inputs = len(in_shapes)

    print("[MODEL] fusion n_inputs =", n_inputs, "input_shapes =", in_shapes)

    if n_inputs == 2:
        ch0 = int(in_shapes[0][-1]) if in_shapes[0][-1] is not None else None
        ch1 = int(in_shapes[1][-1]) if in_shapes[1][-1] is not None else None

        if ch0 == 3 and ch1 == 1:
            probs = model_fusion.predict([x_rgb, x_th], batch_size=64, verbose=0).reshape(-1)
        elif ch0 == 1 and ch1 == 3:
            probs = model_fusion.predict([x_th, x_rgb], batch_size=64, verbose=0).reshape(-1)
        else:
            probs = model_fusion.predict([x_rgb, x_th], batch_size=64, verbose=0).reshape(-1)

    elif n_inputs == 1:
        ch = int(in_shapes[0][-1]) if in_shapes[0][-1] is not None else None

        if ch == 4:
            x_fused = np.concatenate([x_rgb, x_th], axis=-1)
            probs = model_fusion.predict(x_fused, batch_size=64, verbose=0).reshape(-1)
        elif ch == 3:
            probs = model_fusion.predict(x_rgb, batch_size=64, verbose=0).reshape(-1)
        elif ch == 1:
            probs = model_fusion.predict(x_th, batch_size=64, verbose=0).reshape(-1)
        else:
            raise ValueError(
                f"Unsupported single-input model channel={ch}. "
                f"Expected 4 (RGB+TH), 3 (RGB), or 1 (TH). Got input_shapes={in_shapes}"
            )
    else:
        raise ValueError(
            f"Unsupported fusion model with {n_inputs} inputs. input_shapes={in_shapes}"
        )

    pred = (probs >= float(threshold)).astype(int)

    # 6) Records
    records = []
    for i, (r, c, x, y) in enumerate(meta):
        cf = canopy_frac_from_fullmask(main_mask, int(x), int(y), int(patch))
        tmean = float(th_means[i]) if np.isfinite(th_means[i]) else None

        records.append({
            "patch_id": f"grid_{r}_{c}",
            "r": int(r), "c": int(c),
            "x": int(x), "y": int(y),
            "canopy_frac": float(cf),
            "temp_mean": tmean,
            "y_prob": float(probs[i]),
            "pred_stress": int(pred[i]),
        })

    df = pd.DataFrame(records)

    summary = {
        "mode": "fusion",
        "threshold": float(threshold),
        "grid_space": "thermal_grid",
        "align_zoom": float(zoom),
        "align_roi": roi,
        "num_patches": int(len(df)),
        "stress_count": int(df["pred_stress"].sum()),
        "normal_count": int((df["pred_stress"] == 0).sum()),
        "temp_mean_global": float(np.nanmean(th_means)),
        "thermal_kind": thermal_kind,
        "img_w": int(Wt),
        "img_h": int(Ht),
        "grid_rows": int(rows),
        "grid_cols": int(cols),
        "patch": int(patch),
        "stride": int(stride),
        "therm_norm_lo": float(lo),
        "therm_norm_hi": float(hi),
    }

    return df, summary, rgb_aligned

# =========================================================
# Save outputs
# =========================================================
def save_outputs(df: pd.DataFrame, out_dir: str, threshold: float, mode: str):
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "audit_output.csv")
    df2 = df.copy()
    df2["threshold"] = float(threshold)
    df2["mode"] = mode
    df2.to_csv(out_csv, index=False)
    return out_csv

def save_grid_json(
    df: pd.DataFrame,
    out_dir: str,
    img_w: int,
    img_h: int,
    patch: int,
    stride: int,
    mode: str,
    threshold: float
):
    os.makedirs(out_dir, exist_ok=True)

    cols = ["r", "c", "x", "y", "temp_mean", "pred_stress", "y_prob"]

    if "canopy_frac" in df.columns:
        cols.append("canopy_frac")

    if "is_canopy" in df.columns:
        cols.append("is_canopy")
    if "pred_display" in df.columns:
        cols.append("pred_display")

    if "pred_level" in df.columns:
        cols.append("pred_level")
    if "stress_level" in df.columns:
        cols.append("stress_level")

    grid_json = {
        "img_w": int(img_w),
        "img_h": int(img_h),
        "patch": int(patch),
        "stride": int(stride),
        "mode": mode,
        "threshold": float(threshold),
        "cells": df[cols].to_dict(orient="records"),
    }

    out_path = os.path.join(out_dir, "grid.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(grid_json, f, ensure_ascii=False)

    return out_path
