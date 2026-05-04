import cv2
import numpy as np

# ── Thresholds ──────────────────────────────────────────
GAUSSIAN_SIGMA       = 2
CANNY_LOW            = 13      
CANNY_HIGH           = 51
MORPH_DILATE_RADIUS  = 6
MORPH_ERODE_RADIUS   = 3
MIN_REGION_AREA      = 500
SURROUND_RING_RADIUS = 20
DENT_RATIO_THRESH    = 1.8
PAINT_RATIO_THRESH   = 1.5

def analyze_dent(image_path):
    # Load image
    img_original = cv2.imread(image_path)
    if img_original is None:
        return {"error": "Invalid image file"}

    # Preprocessing
    img_blurred = cv2.GaussianBlur(img_original, (5, 5), GAUSSIAN_SIGMA)
    img_gray = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2GRAY)
    img_eq = cv2.equalizeHist(img_gray)

    # Edge Detection & Morphology
    img_edges = cv2.Canny(img_eq, CANNY_LOW, CANNY_HIGH)
    
    kernel_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*MORPH_DILATE_RADIUS+1, 2*MORPH_DILATE_RADIUS+1))
    kernel_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*MORPH_ERODE_RADIUS+1, 2*MORPH_ERODE_RADIUS+1))

    img_dilated = cv2.dilate(img_edges, kernel_d)
    img_filled = flood_fill(img_dilated)
    img_eroded = cv2.erode(img_filled, kernel_e)

    # Region Extraction
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img_eroded)
    valid = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= MIN_REGION_AREA]

    if not valid:
        return {
            "dent_detected": False,
            "recommendation": "No significant dent detected."
        }

    largest_label = max(valid, key=lambda x: x[1])[0]
    dent_mask = (labels == largest_label)

    # Surrounding area for comparison
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*SURROUND_RING_RADIUS+1, 2*SURROUND_RING_RADIUS+1))
    surround_mask = cv2.dilate(dent_mask.astype(np.uint8), ring_kernel).astype(bool)
    surround_mask = surround_mask & ~dent_mask

    # Feature Calculation
    dent_pixels = img_gray[dent_mask].astype(float)
    surr_pixels = img_gray[surround_mask].astype(float)

    if len(dent_pixels) == 0 or len(surr_pixels) == 0:
        return {"dent_detected": False, "recommendation": "Inconclusive scan"}

    dent_var = np.var(dent_pixels)
    surround_var = np.var(surr_pixels)
    dent_ratio = float(dent_var / (surround_var + 1e-6))

    b, g, r = cv2.split(img_original.astype(float))
    dent_paint = np.mean([np.std(c[dent_mask]) for c in [r,g,b]])
    surr_paint = np.mean([np.std(c[surround_mask]) for c in [r,g,b]])
    paint_ratio = float(dent_paint / (surr_paint + 1e-6))

    # Decision Logic
    is_deep = bool(dent_ratio >= DENT_RATIO_THRESH)
    is_damaged = bool(paint_ratio >= PAINT_RATIO_THRESH)

    if not is_damaged and not is_deep:
        rec = "Paintless Dent Repair (PDR)"
    elif is_damaged and not is_deep:
        rec = "PDR + Touch-up Paint"
    elif not is_damaged and is_deep:
        rec = "Standard Dent Repair"
    else:
        rec = "Full Bodywork + Respray"

    return {
        "dent_detected": True,
        "deep_dent": is_deep,
        "damaged_paint": is_damaged,
        "recommendation": rec
    }

def flood_fill(img):
    df_out = img.copy()
    h, w = img.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(df_out, mask, (0, 0), 255)
    return img | cv2.bitwise_not(df_out)
