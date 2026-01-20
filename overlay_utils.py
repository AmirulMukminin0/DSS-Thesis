# overlay_utils.py
import os
import cv2

def build_overlay_normal_vs_stress(rgb_img, df, out_path, alpha=0.35, patch=128):
    """
    Draw overlay rectangles over `rgb_img` (must be the same grid image used for patch extraction).

    Priority label sources:
      1) pred_level  : -1 non_canopy, 0 normal, 1 low, 2 medium, 3 high
      2) pred_display: -1 non_canopy, 0 normal, 1 stress
      3) pred_stress : 0 normal, 1 stress

    Coloring (RGB):
      - normal : green
      - low    : yellow
      - medium : orange
      - high   : red
      - non_canopy : skipped (no fill, no border)
    """
    img = rgb_img.copy()
    overlay = img.copy()

    has_pred_level = ("pred_level" in df.columns)
    has_pred_display = ("pred_display" in df.columns)
    

    def level_to_color_rgb(lv: int):
        # RGB tuples
        if lv == 3:   # high
            return (255, 0, 0)       # red
        if lv == 2:   # medium
            return (255, 165, 0)     # orange
        if lv == 1:   # low
            return (255, 255, 0)     # yellow
        # lv == 0 normal
        return (0, 255, 0)           # green

    # -----------------------------
    # Fill blocks
    # -----------------------------
    for _, row in df.iterrows():
        x = int(row["x"]); y = int(row["y"])

        if has_pred_level:
            lv = int(row["pred_level"])
            if lv == -1:
                continue
        elif has_pred_display:
            pd = int(row["pred_display"])
            if pd == -1:
                continue
            lv = 3 if pd == 1 else 0
        else:
            lv = 3 if int(row["pred_stress"]) == 1 else 0

        color_rgb = level_to_color_rgb(lv)
        cv2.rectangle(overlay, (x, y), (x + patch, y + patch), color_rgb, thickness=-1)

    blended = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    # Borders
    for _, row in df.iterrows():
        x = int(row["x"]); y = int(row["y"])
        if has_pred_level and int(row["pred_level"]) == -1:
            continue
        if (not has_pred_level) and has_pred_display and int(row["pred_display"]) == -1:
            continue
        cv2.rectangle(blended, (x, y), (x + patch, y + patch), (60, 60, 60), thickness=1)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, out_bgr)
    return out_path
