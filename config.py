# config.py
import os

# =========================================================
# Base paths
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR  = os.path.join(BASE_DIR, "models")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# pastikan semua folder ada
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# Model paths (DEPLOY MODELS ONLY)
# =========================================================
# RGB-only deploy model (fallback jika thermal tidak ada)
MODEL_RGB_PATH = os.path.join(MODEL_DIR, "model_final_deploy_rgb_only.keras")

# Fusion deploy model (RGB + Thermal) - mode utama
MODEL_FUSION_PATH = os.path.join(MODEL_DIR, "model_final_deploy_flask.keras")

# =========================================================
# Patch / grid inference params
# (HARUS sama dengan Tahap 7 Colab)
# =========================================================
PATCH_SIZE = 128
STRIDE = 128

# =========================================================
# Target grid (untuk konsistensi overlay saat thermal tidak ada)
# - DJI thermal umumnya 512x640 (H,W)
# - dipakai hanya untuk fallback RGB (tanpa thermal)
# =========================================================
THERMAL_TARGET_H = 512
THERMAL_TARGET_W = 640

# =========================================================
# RGB → Thermal grid alignment
# (center crop + zoom, lalu resize ke thermal grid)
# =========================================================
RGB_TO_THERMAL_ZOOM = 2.2

# =========================================================
# Thermal normalization (untuk inferensi & overlay)
# =========================================================
THERM_P01 = 1.0
THERM_P99 = 99.0

# =========================================================
# DSS & classification threshold (0..1)
# =========================================================
DEFAULT_THRESHOLD_RGB = 0.22
DEFAULT_THRESHOLD_FUSION = 0.30

# =========================================================
# (Optional) Canopy masking
# =========================================================
CANOPY_THR = 0.30
