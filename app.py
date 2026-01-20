# app.py
import os
import uuid

import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, send_file, flash

import inference_utils as iu
from overlay_utils import build_overlay_normal_vs_stress
from config import (
    MODEL_FUSION_PATH,
    PATCH_SIZE, STRIDE,
    THERM_P01, THERM_P99,
    DEFAULT_THRESHOLD_FUSION,
    CANOPY_THR,
    UPLOAD_DIR, OUTPUT_DIR,
)

# =====================================================
# Flask App Init
# =====================================================
app = Flask(__name__)
app.secret_key = "dev_key_change_me"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALLOWED_RGB = {".jpg", ".jpeg", ".png", ".JPG"}
ALLOWED_TH  = {".npy", ".png", ".jpg", ".jpeg"}  # .npy recommended

# =====================================================
# DSS Severity (PATCH-LEVEL)
# - LOW/MED/HIGH untuk patch CANOPY yang pred_stress=1
# - berbasis confidence model (y_prob), bukan klaim biologis
# - RELATIF terhadap threshold agar "high" tidak mustahil
# =====================================================
SEV_LOW_DELTA = 0.10   # threshold .. threshold+0.10  => LOW
SEV_MED_DELTA = 0.25   # threshold+0.10 .. threshold+0.25 => MED
# >= threshold+0.25 => HIGH


# =====================================================
# Utils
# =====================================================
def _ext(fn: str) -> str:
    return os.path.splitext(fn.lower())[1]


def _load_fusion_model():
    print(f"[LOAD] FUSION MODEL : {MODEL_FUSION_PATH}")
    if not os.path.exists(MODEL_FUSION_PATH):
        print("[WARN] Fusion model NOT found")
        return None
    model = tf.keras.models.load_model(MODEL_FUSION_PATH)
    print("[OK] Fusion model loaded")
    return model


# =====================================================
# Load model once (global)
# =====================================================
model_fusion = _load_fusion_model()


# =====================================================
# Routes
# =====================================================
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/run", methods=["POST"])
def run_infer():
    global model_fusion

    # -----------------------------
    # Fusion-only: mode dikunci
    # -----------------------------
    mode = "fusion"
    threshold = float(DEFAULT_THRESHOLD_FUSION)

    rgb_f = request.files.get("rgb_file")
    th_f  = request.files.get("th_file")

    # -----------------------------
    # Validate RGB
    # -----------------------------
    if (not rgb_f) or (rgb_f.filename.strip() == ""):
        flash("Harus upload file RGB.")
        return redirect(url_for("index"))

    if _ext(rgb_f.filename) not in ALLOWED_RGB:
        flash("RGB harus .jpg / .jpeg / .png")
        return redirect(url_for("index"))

    # -----------------------------
    # Validate Thermal (WAJIB)
    # -----------------------------
    if (not th_f) or (th_f.filename.strip() == ""):
        flash("Thermal wajib diupload (.npy disarankan).")
        return redirect(url_for("index"))

    if _ext(th_f.filename) not in ALLOWED_TH:
        flash("Thermal harus .npy / .png / .jpg / .jpeg")
        return redirect(url_for("index"))

    # -----------------------------
    # Check model
    # -----------------------------
    if model_fusion is None:
        flash("Model fusion tidak ditemukan. Cek folder models / path MODEL_FUSION_PATH.")
        return redirect(url_for("index"))

    # -----------------------------
    # Prepare job folders
    # -----------------------------
    job_id = str(uuid.uuid4())[:8]
    job_up = os.path.join(UPLOAD_DIR, job_id)
    job_out = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(job_up, exist_ok=True)
    os.makedirs(job_out, exist_ok=True)

    rgb_path = os.path.join(job_up, "rgb" + _ext(rgb_f.filename))
    th_path  = os.path.join(job_up, "thermal" + _ext(th_f.filename))
    rgb_f.save(rgb_path)
    th_f.save(th_path)

    # -----------------------------
    # Run inference (fusion)
    # -----------------------------
    df, summary, rgb_img = iu.predict_grid_fusion(
        model_fusion=model_fusion,
        rgb_path=rgb_path,
        th_path=th_path,
        patch=PATCH_SIZE,
        stride=STRIDE,
        threshold=threshold,
        p01=THERM_P01,
        p99=THERM_P99,
    )

    # -----------------------------
    # Hardening summary for Jinja
    # -----------------------------
    summary = summary or {}
    summary.setdefault("mode", mode)
    summary.setdefault("temp_mean_global", None)

    if "img_w" not in summary or "img_h" not in summary:
        h, w = rgb_img.shape[:2]
        summary["img_w"] = int(w)
        summary["img_h"] = int(h)

    # -----------------------------
    # DSS: Canopy Gate (RULE-BASED)
    # -----------------------------
    if "canopy_frac" in df.columns:
        df["is_canopy"] = df["canopy_frac"].fillna(0.0) >= float(CANOPY_THR)
    else:
        df["is_canopy"] = True

    # Display label:
    # -1 = NON_CANOPY | 0 = NORMAL | 1 = STRESS
    df["pred_display"] = df["pred_stress"].astype(int)
    df.loc[~df["is_canopy"], "pred_display"] = -1

    # Recompute base summary (canopy-only)
    df_can = df[df["is_canopy"]].copy()
    stress_can = int((df_can["pred_stress"] == 1).sum())
    normal_can = int((df_can["pred_stress"] == 0).sum())

    summary.update({
        "num_patches_total": int(len(df)),
        "num_patches_canopy": int(len(df_can)),
        "stress_count_canopy": stress_can,
        "normal_count_canopy": normal_can,
        "canopy_thr": float(CANOPY_THR),

        # UI tampilkan canopy-only
        "num_patches": int(len(df_can)),
        "stress_count": stress_can,
        "normal_count": normal_can,

        "threshold_used": float(threshold),
    })

    # =====================================================
    # DSS: PATCH-LEVEL Severity (LOW/MED/HIGH)
    # pred_level: -1 non_canopy, 0 normal, 1 low, 2 medium, 3 high
    # =====================================================
    t = float(threshold)
    low_max = min(1.0, t + float(SEV_LOW_DELTA))
    med_max = min(1.0, t + float(SEV_MED_DELTA))

    df["stress_level"] = "normal"
    df["pred_level"] = 0

    is_stress_canopy = (df["is_canopy"] == True) & (df["pred_stress"] == 1)

    if "y_prob" in df.columns and int(is_stress_canopy.sum()) > 0:
        df.loc[is_stress_canopy & (df["y_prob"] < low_max), "stress_level"] = "low"
        df.loc[is_stress_canopy & (df["y_prob"] >= low_max) & (df["y_prob"] < med_max), "stress_level"] = "medium"
        df.loc[is_stress_canopy & (df["y_prob"] >= med_max), "stress_level"] = "high"
    else:
        # fallback minimal (harusnya jarang terjadi)
        df.loc[is_stress_canopy, "stress_level"] = "low"

    df.loc[~df["is_canopy"], "stress_level"] = "non_canopy"

    df.loc[df["stress_level"] == "low", "pred_level"] = 1
    df.loc[df["stress_level"] == "medium", "pred_level"] = 2
    df.loc[df["stress_level"] == "high", "pred_level"] = 3
    df.loc[df["stress_level"] == "non_canopy", "pred_level"] = -1

    # Level counts (canopy-only)
    df_can2 = df[df["is_canopy"]].copy()
    stress_low_n  = int((df_can2["stress_level"] == "low").sum())
    stress_med_n  = int((df_can2["stress_level"] == "medium").sum())
    stress_high_n = int((df_can2["stress_level"] == "high").sum())

    summary.update({
        "stress_low": int(stress_low_n),
        "stress_medium": int(stress_med_n),
        "stress_high": int(stress_high_n),
        "sev_low_max": float(low_max),
        "sev_med_max": float(med_max),
    })

    # =====================================================
    # DSS: AREA COVERAGE Summary (LOW/MED/HIGH)
    # - LOW  : stress sedikit (normal dominan)
    # - MED  : seimbang
    # - HIGH : stress dominan
    # =====================================================
    total_can = int(len(df_can2))
    stress_n = int(stress_low_n + stress_med_n + stress_high_n)
    normal_n = int(total_can - stress_n)
    stress_ratio = (stress_n / total_can) if total_can > 0 else 0.0

    if stress_ratio < 0.10:
        area_status = "normal"
    elif stress_ratio < 0.33:
        area_status = "low"
    elif stress_ratio <= 0.66:
        area_status = "medium"
    else:
        area_status = "high"

    dominant_patch_level = "normal"
    if stress_n > 0:
        lev_counts = {"low": stress_low_n, "medium": stress_med_n, "high": stress_high_n}
        dominant_patch_level = max(lev_counts, key=lev_counts.get)

    summary.update({
        "stress_total": int(stress_n),
        "normal_total": int(normal_n),
        "stress_ratio": float(stress_ratio),
        "area_status": area_status,
        "dominant_patch_level": dominant_patch_level,
    })

    # -----------------------------
    # Save outputs
    # -----------------------------
    out_csv = iu.save_outputs(df, job_out, threshold=threshold, mode=summary.get("mode", mode))

    overlay_path = os.path.join(job_out, "overlay.png")
    build_overlay_normal_vs_stress(
        rgb_img=rgb_img,
        df=df,
        out_path=overlay_path,
        alpha=0.35,
        patch=PATCH_SIZE,
    )

    grid_json_path = iu.save_grid_json(
        df=df,
        out_dir=job_out,
        img_w=int(summary["img_w"]),
        img_h=int(summary["img_h"]),
        patch=PATCH_SIZE,
        stride=STRIDE,
        mode=summary.get("mode", mode),
        threshold=threshold,
    )

    # -----------------------------
    # Render result
    # -----------------------------
    return render_template(
        "result.html",
        job_id=job_id,
        summary=summary,
        csv_name=os.path.basename(out_csv),
        overlay_name=os.path.basename(overlay_path),
        grid_json_name=os.path.basename(grid_json_path),
    )


@app.route("/download/<job_id>/<fname>")
def download(job_id, fname):
    path = os.path.join(OUTPUT_DIR, job_id, fname)
    if not os.path.exists(path):
        return "File not found", 404
    return send_file(path, as_attachment=True)


@app.route("/view/<job_id>/<fname>")
def view_file(job_id, fname):
    path = os.path.join(OUTPUT_DIR, job_id, fname)
    if not os.path.exists(path):
        return "File not found", 404
    return send_file(path, as_attachment=False)


if __name__ == "__main__":
    app.run(debug=True)
