"""
dent_detector_dl.py  —  Tier 2 Deep Learning Damage Detector
=============================================================
Detects BOTH dents and scratches.

Two modes:
  • TRAINED MODE   — YOLOv8-seg fine-tuned weights (supports dent/scratch/crack classes)
  • FALLBACK MODE  — Enhanced classical CV:
      - Dents   : blob analysis via CLAHE + Canny + connected components
      - Scratches: linear analysis via HoughLinesP + color channel disruption

Output per image
----------------
{
  "mode": "trained" | "fallback",
  "damages_found": int,
  "dents_found": int,
  "scratches_found": int,
  "overall_severity": "None"|"Minor"|"Moderate"|"Severe"|"Critical",
  "severity_score": float (0-100),
  "estimated_total_cost_usd": str,
  "confidence": float (0-1),
  "damages": [ { per-damage details } ],
  "annotated_image_base64": str   # JPEG base64
}
"""

import base64
from pathlib import Path

import cv2
import numpy as np

# ── Model path ───────────────────────────────────────────────────────────────
CUSTOM_WEIGHTS_PATH = Path(__file__).parent / "models" / "dent_best.pt"

# ── Dent CV thresholds ───────────────────────────────────────────────────────
_GAUSSIAN_SIGMA        = 2
_CANNY_LOW             = 20       # raised to reduce noise
_CANNY_HIGH            = 60
_MORPH_DILATE_RADIUS   = 5
_MORPH_ERODE_RADIUS    = 3
_MIN_REGION_AREA       = 800
_MAX_REGION_AREA_RATIO = 0.22     # blobs > 22% of image = car body, ignore
_SURROUND_RING_RAD     = 20
_DENT_RATIO_THRESH     = 1.8
_PAINT_RATIO_THRESH    = 1.5

# ── Shadow-highlight dent detection (large smooth dents) ─────────────────────
_SH_LOCAL_BLUR         = 91       # larger blur for smoother dents
_SH_SHADOW_THRESH      = -18      # more sensitive
_SH_HIGHLIGHT_THRESH   = 18       
_SH_MIN_SHADOW_AREA    = 1500     # lower area to catch peaks of large dents
_SH_MIN_HIGHLIGHT_PX   = 400      

# ── Scratch CV thresholds ────────────────────────────────────────────────────
_SCRATCH_CANNY_LOW        = 25    # even lower for scuffs
_SCRATCH_CANNY_HIGH       = 95
_SCRATCH_HOUGH_THRESHOLD  = 30    
_SCRATCH_MIN_LENGTH       = 28    
_SCRATCH_MAX_GAP          = 15
_SCRATCH_PAINT_THRESH     = 1.55  # slightly looser for scuffs
_SCRATCH_THICKNESS        = 6
_SCRATCH_SURROUND_WIDTH   = 22
# Panel-gap rejection filters
_SCRATCH_DARK_MASK_THRESH = 45    # stricter masking of gaps
_SCRATCH_MAX_DARKNESS     = 75    # higher min brightness for scratches
_SCRATCH_MAX_LEN_RATIO    = 0.35  
_SCRATCH_MIN_INTEN_VAR    = 9.0   # higher variation for real scratches
_SCRATCH_REL_DARK_RATIO   = 0.70  # gap must be within 70% of paint brightness

# ── Repair cost lookup (USD) — covers dents AND scratches ────────────────────
REPAIR_COSTS = {
    # Dent recommendations
    "Paintless Dent Repair (PDR)":  (80,   200,  "1-2"),
    "PDR + Touch-up Paint":          (150,  350,  "2-4"),
    "Standard Dent Repair":          (250,  500,  "4-6"),
    "Full Bodywork + Respray":        (600, 2000,  "8-16"),
    # Scratch recommendations
    "Light Scratch Polish":           (50,   120,  "0.5-1"),
    "Scratch Touch-up Paint":         (100,  250,  "1-3"),
    "Scratch Panel Repaint":          (300,  700,  "4-8"),
}

# ── Severity bands ───────────────────────────────────────────────────────────
def _score_to_band(score: float) -> str:
    if score < 15:  return "None"
    if score < 35:  return "Minor"
    if score < 60:  return "Moderate"
    if score < 80:  return "Severe"
    return "Critical"


# ════════════════════════════════════════════════════════════════════════════
#  Main Detector Class
# ════════════════════════════════════════════════════════════════════════════
class DentDetectorDL:
    """
    Instantiate once and call analyze() for each image.
    Thread-safe in Ultralytics >= 8.x.
    """

    def __init__(self):
        self.mode  = "fallback"
        self.model = None
        self._load_model()

    # ── Model Loading ─────────────────────────────────────────────────────
    def _load_model(self):
        try:
            from ultralytics import YOLO
            if CUSTOM_WEIGHTS_PATH.exists():
                print(f"[Detector] Loading TRAINED weights: {CUSTOM_WEIGHTS_PATH}")
                self.model = YOLO(str(CUSTOM_WEIGHTS_PATH))
                self.mode  = "trained"
            else:
                print("[Detector] No trained weights — YOLOv8n-seg + enhanced CV (fallback)")
                self.model = YOLO("yolov8n-seg.pt")
                self.mode  = "fallback"
        except Exception as e:
            print(f"[Detector] YOLO unavailable ({e}), pure classical CV")
            self.model = None
            self.mode  = "fallback"

    # ── Public Entry Point ────────────────────────────────────────────────
    def analyze(self, image_path: str, sensitivity: float = 75.0) -> dict:
        img = cv2.imread(image_path)
        if img is None:
            return {"error": "Invalid image file"}
        if self.mode == "trained" and self.model is not None:
            return self._analyze_trained(img)
        return self._analyze_fallback(img, sensitivity=sensitivity)

    # ════════════════════════════════════════════════════════════════════════
    #  TRAINED MODE
    # ════════════════════════════════════════════════════════════════════════
    def _analyze_trained(self, img: np.ndarray) -> dict:
        results = self.model.predict(img, conf=0.25, verbose=False)
        result  = results[0]

        damages   = []
        annotated = img.copy()
        color_map = {
            "dent":    (0, 255, 100),
            "scratch": (255, 180, 0),
            "crack":   (0,  80, 255),
        }

        masks = result.masks.data.cpu().numpy() if result.masks else []
        boxes = result.boxes

        for idx, (mask, box) in enumerate(zip(masks, boxes)):
            label_id   = int(box.cls[0])
            label_name = self.model.names[label_id]
            conf       = float(box.conf[0])

            mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
            mask_bool    = mask_resized > 0.5
            area_px      = int(mask_bool.sum())
            gray         = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ring_k   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
            surround = cv2.dilate(mask_bool.astype(np.uint8), ring_k).astype(bool) & ~mask_bool

            dent_var    = np.var(gray[mask_bool].astype(float))
            surr_var    = np.var(gray[surround].astype(float)) if surround.any() else 1
            dent_ratio  = float(dent_var / (surr_var + 1e-6))

            b, g, r = cv2.split(img.astype(float))
            dp = np.mean([np.std(c[mask_bool]) for c in [r, g, b]])
            sp = np.mean([np.std(c[surround])  for c in [r, g, b]])
            paint_ratio = float(dp / (sp + 1e-6))

            severity   = self._compute_severity(dent_ratio, paint_ratio, area_px, conf)
            is_deep    = dent_ratio  >= _DENT_RATIO_THRESH
            is_damaged = paint_ratio >= _PAINT_RATIO_THRESH

            if label_name == "scratch":
                rec = self._recommend_scratch(paint_ratio, area_px)
            else:
                rec = self._recommend_dent(is_deep, is_damaged)

            cost_lo, cost_hi, time_r = REPAIR_COSTS[rec]
            shape = self._classify_shape(mask_bool) if label_name != "scratch" else "Scratch"

            damages.append({
                "id":                 idx + 1,
                "damage_class":       label_name,
                "type":               shape,
                "severity_score":     round(severity, 1),
                "severity_band":      _score_to_band(severity),
                "area_px":            area_px,
                "deep_dent":          is_deep if label_name == "dent" else None,
                "paint_damage":       is_damaged,
                "pdr_eligible":       (not is_deep and not is_damaged) if label_name == "dent" else False,
                "recommendation":     rec,
                "estimated_cost_usd": f"${cost_lo}–${cost_hi}",
                "repair_time_hours":  time_r,
                "confidence":         round(conf, 3),
            })

            color   = color_map.get(label_name, (255, 200, 0))
            overlay = annotated.copy()
            overlay[mask_bool] = color
            annotated = cv2.addWeighted(annotated, 0.6, overlay, 0.4, 0)

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            self._draw_label(annotated, f"#{idx+1} {label_name} ({severity:.0f})", (x1, y1-10), color)

        overall  = max((d["severity_score"] for d in damages), default=0)
        dents    = [d for d in damages if d["damage_class"] == "dent"]
        scratches= [d for d in damages if d["damage_class"] == "scratch"]

        return {
            "mode":                     "trained",
            "damages_found":            len(damages),
            "dents_found":              len(dents),
            "scratches_found":          len(scratches),
            "overall_severity":         _score_to_band(overall),
            "severity_score":           round(overall, 1),
            "estimated_total_cost_usd": self._aggregate_cost(damages),
            "confidence":               round(float(np.mean([d["confidence"] for d in damages])) if damages else 0, 3),
            "damages":                  damages,
            "annotated_image_base64":   self._to_b64(annotated),
        }

    # ════════════════════════════════════════════════════════════════════════
    #  FALLBACK MODE  — Dents + Scratches via classical CV
    # ════════════════════════════════════════════════════════════════════════
    def _analyze_fallback(self, img: np.ndarray, sensitivity: float = 75.0) -> dict:
        h, w = img.shape[:2]
        img_area = h * w
        
        # ── Dynamic Sensitivity Scaling ──
        # factor < 1.0 = more sensitive (lower thresholds)
        # factor > 1.0 = stricter (higher thresholds)
        factor = 1.0 + (75.0 - sensitivity) / 50.0
        factor = max(0.2, min(3.0, factor))

        # 1. Bilateral Filter: Smooths reflections while preserving damage edges
        # Very effective for improving accuracy on glossy cars.
        denoised = cv2.bilateralFilter(img, 9, 75, 75)
        
        # ── 0. Intelligent Masking (Body Only) ───────────────────────────
        # Use YOLO even in fallback to find the car body and ignore windows/wheels
        ignore_mask = np.zeros((h, w), np.uint8)
        
        if self.model is not None:
            try:
                # Detect 'car', 'bus', 'truck' (COCO 2, 5, 7)
                res = self.model.predict(img, classes=[2, 5, 7], conf=0.3, verbose=False)[0]
                if res.boxes:
                    body_mask = np.zeros((h, w), np.uint8)
                    for box in res.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        bw, bh = x2 - x1, y2 - y1
                        # 1. Include the car body
                        cv2.rectangle(body_mask, (x1, y1), (x2, y2), 255, -1)
                        
                        # 2. Exclude the "Window Zone" (top 42% of car height)
                        win_h = int(bh * 0.42)
                        cv2.rectangle(ignore_mask, (x1, y1), (x2, y1 + win_h), 255, -1)
                        
                        # 3. Exclude the "Bottom Edge" (Side-skirts/Ground) - Aggressive
                        ground_h = int(bh * 0.16)
                        cv2.rectangle(ignore_mask, (x1, y2 - ground_h), (x2, y2), 255, -1)

                        # 4. HEURISTIC WHEEL EXCLUSION (Bottom-left and Bottom-right areas)
                        # Most car photos have wheels in these zones relative to the car box
                        wheel_r = int(bh * 0.18)
                        # Front wheel area
                        cv2.circle(ignore_mask, (x1 + int(bw * 0.22), y2 - int(bh * 0.15)), wheel_r, 255, -1)
                        # Rear wheel area
                        cv2.circle(ignore_mask, (x1 + int(bw * 0.78), y2 - int(bh * 0.15)), wheel_r, 255, -1)

                    roi_mask = cv2.bitwise_and(body_mask, cv2.bitwise_not(ignore_mask))
                    ignore_mask = cv2.bitwise_not(roi_mask)
            except Exception as e:
                print(f"[Detector] YOLO ROI failed: {e}")

        # Ensure the absolute bottom of the image is always ignored (Ground Guard)
        cv2.rectangle(ignore_mask, (0, int(h * 0.94)), (w, h), 255, -1)
        
        # 6. EXCLUDE TAIL LIGHTS (Red/Orange regions)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Red spans 0-10 and 170-180 in OpenCV HSV
        red1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([12, 255, 255]))
        red2 = cv2.inRange(hsv, np.array([165, 100, 100]), np.array([180, 255, 255]))
        orange = cv2.inRange(hsv, np.array([13, 100, 100]), np.array([25, 255, 255]))
        lights_mask = cv2.dilate(red1 | red2 | orange, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        ignore_mask = cv2.bitwise_or(ignore_mask, lights_mask)

        blurred = cv2.GaussianBlur(denoised, (5, 5), _GAUSSIAN_SIGMA)
        gray    = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        eq      = clahe.apply(gray)

        annotated = img.copy()
        damages   = []

        # ── 1. Dent Detection (blob analysis) ─────────────────────────────
        dent_items = self._detect_dents(img, gray, eq, annotated, ignore_mask, factor=factor)
        damages.extend(dent_items)

        # ── 2. Scratch Detection (line analysis) ──────────────────────────
        scratch_items = self._detect_scratches(eq, gray, img, annotated, factor=factor)
        damages.extend(scratch_items)

        # Re-number IDs sequentially
        for i, d in enumerate(damages):
            d["id"] = i + 1

        if not damages:
            self._draw_label(annotated, "No significant damage detected", (20, 40), (0, 220, 100))

        overall    = max((d["severity_score"] for d in damages), default=0)
        dents      = [d for d in damages if d["damage_class"] == "dent"]
        scratches  = [d for d in damages if d["damage_class"] == "scratch"]
        avg_conf   = round(float(np.mean([d["confidence"] for d in damages])) if damages else 0, 3)

        return {
            "mode":                     "fallback",
            "damages_found":            len(damages),
            "dents_found":              len(dents),
            "scratches_found":          len(scratches),
            "overall_severity":         _score_to_band(overall),
            "severity_score":           round(overall, 1),
            "estimated_total_cost_usd": self._aggregate_cost(damages),
            "confidence":               avg_conf,
            "dents":                    damages,
            "annotated_image_base64":   self._to_b64(annotated),
        }

    # ── Dent sub-detector ─────────────────────────────────────────────────────
    def _detect_dents(self, img: np.ndarray, gray: np.ndarray, eq: np.ndarray, annotated: np.ndarray, 
                      ignore_mask: np.ndarray = None, factor: float = 1.0) -> list:
        h_img, w_img = gray.shape[:2]
        img_area     = h_img * w_img
        results      = []
        colors       = [(0,255,100),(0,180,255),(0,80,255),(255,180,0),(255,80,200)]

        # ── Strategy A: Edge blob analysis (sharp small dents) ────────────────
        edges   = cv2.Canny(eq, _CANNY_LOW, _CANNY_HIGH)
        if ignore_mask is not None:
            edges[ignore_mask > 0] = 0   # Block edges on wheels

        ker_d   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*_MORPH_DILATE_RADIUS+1,)*2)
        ker_e   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*_MORPH_ERODE_RADIUS+1,)*2)
        eroded  = cv2.erode(cv2.dilate(self._flood_fill(cv2.dilate(edges, ker_d)), ker_d), ker_e)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(eroded)
        valid = [(i, stats[i, cv2.CC_STAT_AREA])
                 for i in range(1, num_labels)
                 if _MIN_REGION_AREA <= stats[i, cv2.CC_STAT_AREA] <= img_area * _MAX_REGION_AREA_RATIO]
        top   = sorted(valid, key=lambda x: x[1], reverse=True)[:4]

        # ── Global Ground Guard ──
        # Reject any blobs in the bottom 15% of the TOTAL image height
        # that are very wide or touching the bottom edge.
        ground_limit = int(h_img * 0.85)
        
        for idx, (label_id, area_px) in enumerate(top):
            mask     = (labels == label_id)
            x, y, w, h = stats[label_id, :4]
            
            # ── Ground Plane Rejection ──
            # Floor/Ground is usually low, wide, and has low contrast vs surroundings
            if (y + h) > h_img * 0.94 and w > w_img * 0.2:
                continue # Hard floor reject
            if y > h_img * 0.72 and w > w_img * 0.5:
                # Large wide blob in bottom part is ground
                continue
            
            # ── Handle & Wheel Component Rejection ──
            x, y, w, h = stats[label_id, :4]
            aspect     = w / (h + 1e-6)
            
            # 1. Door handles
            if 1500 < area_px < 8000 and 1.8 < aspect < 5.0:
                continue 
            
            # 2. Wheel Spokes / Lug Nuts / Circular Rim parts
            # Circularity check for Strategy A
            cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                cnt = cnts[0]
                circ = 4 * np.pi * cv2.contourArea(cnt) / (cv2.arcLength(cnt, True)**2 + 1e-6)
                if circ > 0.78 and y > h_img * 0.5:
                    continue # Reject circular blobs in bottom half (wheels/rims)
            
            # 3. Intensity Check (Rubber/Gaps vs Paint)
            # Tire rubber and wheel-well shadows are very dark
            if np.mean(gray[mask]) < 75: # Raised from 65
                continue

            # 4. Texture Density Filter (Lettering/Tire tread)
            # Tires have high edge density but low color variation.
            b, g, r = cv2.split(img.astype(float))
            dp      = np.mean([np.std(c[mask]) for c in [r, g, b]])
            if dp < 4.5 and y > h_img * 0.6: 
                continue # Likely tire rubber/lettering (uniform color, high texture)
            
            color    = colors[idx]
            ring_k   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*_SURROUND_RING_RAD+1,)*2)
            surround = cv2.dilate(mask.astype(np.uint8), ring_k).astype(bool) & ~mask
            dent_pix = gray[mask].astype(float)
            surr_pix = gray[surround].astype(float)
            if len(dent_pix) == 0 or len(surr_pix) == 0:
                continue
            dent_ratio  = float(np.var(dent_pix) / (np.var(surr_pix) + 1e-6))
            b, g, r     = cv2.split(img.astype(float))
            dp          = np.mean([np.std(c[mask])     for c in [r, g, b]])
            sp          = np.mean([np.std(c[surround]) for c in [r, g, b]])
            paint_ratio = float(dp / (sp + 1e-6))
            conf        = float(min(0.72, 0.28 + (dent_ratio / 6.0) * 0.44))
            severity    = self._compute_severity(dent_ratio, paint_ratio, int(area_px), conf)
            is_deep     = dent_ratio  >= _DENT_RATIO_THRESH
            is_damaged  = paint_ratio >= _PAINT_RATIO_THRESH
            shape       = self._classify_shape(mask)
            rec         = self._recommend_dent(is_deep, is_damaged)
            lo, hi, tr  = REPAIR_COSTS[rec]

            results.append({
                "id":                 0,
                "damage_class":       "dent",
                "type":               shape,
                "severity_score":     round(severity, 1),
                "severity_band":      _score_to_band(severity),
                "area_px":            int(area_px),
                "deep_dent":          is_deep,
                "paint_damage":       is_damaged,
                "pdr_eligible":       not is_deep and not is_damaged,
                "recommendation":     rec,
                "estimated_cost_usd": f"${lo}-${hi}",
                "repair_time_hours":  tr,
                "confidence":         round(conf, 3),
            })

            overlay = annotated.copy()
            overlay[mask] = color
            cv2.addWeighted(annotated, 0.55, overlay, 0.45, 0, annotated)
            x = stats[label_id, cv2.CC_STAT_LEFT];  w = stats[label_id, cv2.CC_STAT_WIDTH]
            y = stats[label_id, cv2.CC_STAT_TOP];   h = stats[label_id, cv2.CC_STAT_HEIGHT]
            cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)
            self._draw_label(annotated, f"DENT | {shape} | Sev:{severity:.0f}", (x, y-10), color)

        # ── Strategy B: Shadow-highlight analysis (large smooth dents) ────────
        # Large collision dents have no sharp edges — Canny misses them.
        # They DO create a dark shadow adjacent to a bright highlight.
        local_mean  = cv2.GaussianBlur(gray.astype(float), (_SH_LOCAL_BLUR, _SH_LOCAL_BLUR), 25)
        deviation   = gray.astype(float) - local_mean
        # Stricter shadow threshold for large panels to avoid arch-shadows
        shadow_mask = (deviation < ((_SH_SHADOW_THRESH - 5) * factor)).astype(np.uint8) * 255
        
        if ignore_mask is not None:
            shadow_mask[ignore_mask > 0] = 0 # Block shadows on wheels/windows

        hi_mask     = (deviation > ((_SH_HIGHLIGHT_THRESH + 5) * factor)).astype(np.uint8) * 255
        k_cl        = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        shadow_cl   = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, k_cl)
        hi_cl       = cv2.morphologyEx(hi_mask,     cv2.MORPH_CLOSE, k_cl)

        ns, sl, ss, _ = cv2.connectedComponentsWithStats(shadow_cl)
        used_color_idx = len(results)
        for i in range(1, ns):
            if len(results) >= 5: break
            area = ss[i, cv2.CC_STAT_AREA]
            if not (_SH_MIN_SHADOW_AREA <= area <= img_area * 0.30): continue
            s_mask   = (sl == i)
            # ── Dark Void/Window Filter ──
            # Windows/Gaps are very dark. Real dented paint is brighter.
            if np.mean(gray[s_mask]) < 50: # lowered from 65 to see darker paint
                continue
            
            vicinity = cv2.dilate(s_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(150,150))).astype(bool) & ~s_mask
            if int(hi_cl[vicinity].sum() // 255) < 300: continue

            ring_k   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(41,41))
            surround = cv2.dilate(s_mask.astype(np.uint8),ring_k).astype(bool) & ~s_mask
            dent_pix = gray[s_mask].astype(float)
            surr_pix = gray[surround].astype(float)
            if len(dent_pix)==0 or len(surr_pix)==0: continue

            dent_ratio  = float(np.var(dent_pix)/(np.var(surr_pix)+1e-6))
            b,g,r       = cv2.split(img.astype(float))
            dp          = np.mean([np.std(c[s_mask])   for c in [r,g,b]])
            sp          = np.mean([np.std(c[surround]) for c in [r,g,b]])
            paint_ratio = float(dp/(sp+1e-6))
            conf        = 0.60
            severity    = self._compute_severity(max(dent_ratio,1.9), paint_ratio, int(area), conf)
            is_deep     = True
            is_damaged  = paint_ratio >= _PAINT_RATIO_THRESH
            shape       = "Large Panel Dent"
            rec         = self._recommend_dent(is_deep, is_damaged)
            lo, hi, tr  = REPAIR_COSTS[rec]
            color       = colors[min(used_color_idx, len(colors)-1)]
            used_color_idx += 1

            results.append({"id":0,"damage_class":"dent","type":shape,
                "severity_score":round(severity,1),"severity_band":_score_to_band(severity),
                "area_px":int(area),"deep_dent":True,"paint_damage":is_damaged,
                "pdr_eligible":False,"recommendation":rec,
                "estimated_cost_usd":f"${lo}\u2013${hi}","repair_time_hours":tr,
                "confidence":round(conf,3)})

            overlay = annotated.copy(); overlay[s_mask] = color
            cv2.addWeighted(annotated,0.50,overlay,0.50,0,annotated)
            x=ss[i,cv2.CC_STAT_LEFT]; w=ss[i,cv2.CC_STAT_WIDTH]
            y=ss[i,cv2.CC_STAT_TOP];  h=ss[i,cv2.CC_STAT_HEIGHT]
            cv2.rectangle(annotated,(x,y),(x+w,y+h),color,2)
            self._draw_label(annotated,f"DENT | {shape} | Sev:{severity:.0f}",(x,y-10),color)

        return results

    # ── Scratch sub-detector ─────────────────────────────────────────────
    def _detect_scratches(self, eq: np.ndarray, gray: np.ndarray, img: np.ndarray, annotated: np.ndarray, 
                         factor: float = 1.0) -> list:
        """
        Panel-gap proof scratch detector with Scuff/Ground protection.
        """
        h_img, w_img = gray.shape[:2]
        min_dim      = min(h_img, w_img)

        # ── Step 1: mask out dark pixels so panel gaps are invisible to Canny ──
        eq_masked              = eq.copy()
        eq_masked[gray < (_SCRATCH_DARK_MASK_THRESH * factor)] = 0   # black out gap regions

        edges = cv2.Canny(eq_masked, int(_SCRATCH_CANNY_LOW * factor), int(_SCRATCH_CANNY_HIGH * factor))
        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 180,
            threshold=_SCRATCH_HOUGH_THRESHOLD,
            minLineLength=_SCRATCH_MIN_LENGTH,
            maxLineGap=_SCRATCH_MAX_GAP,
        )
        
        results       = []
        used_regions  = set()
        scratch_color = (0, 200, 255)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
                if length < _SCRATCH_MIN_LENGTH: continue

                # ── Filter A: structural line check ──
                if length > _SCRATCH_MAX_LEN_RATIO * min_dim: continue
                
                # Filter A3: Linearity / Body Seam rejection
                # Structural gaps are extremely straight (low variance of distance to line)
                # This rejects bumper-to-fender seams
                points = []
                # Simple straightness check via endpoints (already done by P)
                # but we'll check the brightness profile consistency.

                s_mask = np.zeros((h_img, w_img), np.uint8)
                r_mask = np.zeros((h_img, w_img), np.uint8)
                cv2.line(s_mask, (x1, y1), (x2, y2), 255, thickness=_SCRATCH_THICKNESS)
                cv2.line(r_mask, (x1, y1), (x2, y2), 255, thickness=_SCRATCH_SURROUND_WIDTH)
                s_bool = s_mask > 0
                r_bool = (r_mask > 0) & ~s_bool

                s_pix, r_pix = gray[s_bool].astype(float), gray[r_bool].astype(float)
                if len(s_pix) == 0 or len(r_pix) == 0: continue

                s_mean, r_mean = float(np.mean(s_pix)), float(np.mean(r_pix))
                if s_mean < _SCRATCH_MAX_DARKNESS: continue
                if r_mean > 1 and (s_mean / r_mean) < _SCRATCH_REL_DARK_RATIO: continue
                if float(np.std(s_pix)) < _SCRATCH_MIN_INTEN_VAR: continue

                b, g, r_ch = cv2.split(img.astype(float))
                sc = np.mean([np.std(c[s_bool]) for c in [r_ch, g, b]])
                sr = np.mean([np.std(c[r_bool]) for c in [r_ch, g, b]])
                paint_disruption = float(sc / (sr + 1e-6))
                if paint_disruption < (_SCRATCH_PAINT_THRESH * factor): continue

                cell = ((x1 + x2) // 2 // 45, (y1 + y2) // 2 // 45)
                if cell in used_regions: continue
                used_regions.add(cell)

                conf     = float(min(0.72, 0.25 + (paint_disruption / 4.0) * 0.47))
                severity = float(min(100.0, (length/250.0)*45 + (paint_disruption/3.5)*45 + conf*10))
                rec      = self._recommend_scratch(paint_disruption, int(s_bool.sum()))
                lo, hi, tr = REPAIR_COSTS[rec]

                results.append({
                    "id": 0, "damage_class": "scratch", "type": self._classify_scratch(length, paint_disruption),
                    "severity_score": round(severity, 1), "severity_band": _score_to_band(severity),
                    "area_px": int(s_bool.sum()), "length_px": int(length), "deep_dent": None,
                    "paint_damage": True, "pdr_eligible": False, "recommendation": rec,
                    "estimated_cost_usd": f"${lo}\u2013${hi}", "repair_time_hours": tr, "confidence": round(conf, 3)
                })
                cv2.line(annotated, (x1, y1), (x2, y2), scratch_color, 3)
                self._draw_label(annotated, f"SCRATCH | Sev:{severity:.0f}", (min(x1,x2), min(y1,y2)-10), scratch_color)

        # ── Step 3: Scuff / Bumper Cluster Detection ──────────────────────
        # Detects dense areas of small scratches (scuffs) using lower thresholds
        edges_low = cv2.Canny(eq_masked, 15, 60) # Super sensitive for scuffs
        kernel_scuff = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        scuff_mask   = cv2.morphologyEx(edges_low, cv2.MORPH_CLOSE, kernel_scuff)
        num_sc, labels_sc, stats_sc, _ = cv2.connectedComponentsWithStats(scuff_mask)
        
        for i in range(1, num_sc):
            area = stats_sc[i, cv2.CC_STAT_AREA]
            if area < 600 or area > 30000: continue 
            s_mask = (labels_sc == i)
            x, y, w, h = stats_sc[i, :4]
            
            # Ground Filter
            if y > h_img * 0.88: continue

            skip = False
            if lines is not None:
                for line in lines:
                    lx1, ly1, lx2, ly2 = line[0]
                    mx, my = (lx1+lx2)//2, (ly1+ly2)//2
                    if 0 <= my < h_img and 0 <= mx < w_img:
                        if s_mask[my, mx]: skip = True; break
            if skip: continue

            ring_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
            surr   = cv2.dilate(s_mask.astype(np.uint8), ring_k).astype(bool) & ~s_mask
            s_mean, r_mean = np.mean(gray[s_mask]), (np.mean(gray[surr]) if surr.any() else 100)
            is_bright = (s_mean > r_mean + 8) # lowered threshold

            b, g, r_ch = cv2.split(img.astype(float))
            sc = np.mean([np.std(c[s_mask]) for c in [r_ch, g, b]])
            sr = np.mean([np.std(c[surr])   for c in [r_ch, g, b]]) if surr.any() else 0.5
            pd = float(sc / (sr + 1e-6))
            
            if pd > 1.12 or is_bright: # lowered from 1.22
                conf = 0.60 if is_bright else 0.50
                severity = float(min(100.0, (area/6000.0)*40 + (pd/2.5)*50 + (10 if is_bright else 0)))
                results.append({
                    "id": 0, "damage_class": "scratch", "type": "Bumper Scuff / Cluster",
                    "severity_score": round(severity, 1), "severity_band": _score_to_band(severity),
                    "area_px": int(area), "length_px": int(np.sqrt(area)), "deep_dent": None,
                    "paint_damage": True, "pdr_eligible": False, "recommendation": "Scratch Touch-up Paint",
                    "estimated_cost_usd": "$150-$350", "repair_time_hours": "2-4", "confidence": conf
                })
                overlay = annotated.copy(); overlay[s_mask] = scratch_color
                cv2.addWeighted(annotated, 0.6, overlay, 0.4, 0, annotated)
                self._draw_label(annotated, f"SCUFF | Sev:{severity:.0f}", (x, y-10), scratch_color)

        return results

    # ════════════════════════════════════════════════════════════════════════
    #  Helper Methods
    # ════════════════════════════════════════════════════════════════════════
    @staticmethod
    def _flood_fill(img: np.ndarray) -> np.ndarray:
        out  = img.copy()
        h, w = img.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(out, mask, (0, 0), 255)
        return img | cv2.bitwise_not(out)

    @staticmethod
    def _compute_severity(dent_ratio: float, paint_ratio: float,
                          area_px: int, confidence: float) -> float:
        depth_score = min(50.0, (dent_ratio  / 4.0) * 50.0)
        paint_score = min(30.0, (paint_ratio  / 3.0) * 30.0)
        size_score  = min(20.0, (area_px      / 50_000) * 20.0)
        return round((depth_score + paint_score + size_score) * confidence, 1)

    @staticmethod
    def _classify_shape(mask: np.ndarray) -> str:
        contours, _ = cv2.findContours(mask.astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return "Irregular Dent"
        cnt         = max(contours, key=cv2.contourArea)
        area        = cv2.contourArea(cnt)
        perimeter   = cv2.arcLength(cnt, True)
        if perimeter == 0:
            return "Irregular Dent"
        circularity  = 4 * np.pi * area / (perimeter ** 2)
        x, y, w, h  = cv2.boundingRect(cnt)
        aspect_ratio = w / (h + 1e-6)
        if circularity > 0.65:             return "Round Impact Dent"
        if aspect_ratio > 3.0 or aspect_ratio < 0.33: return "Crease / Line Dent"
        if area > 20_000:                  return "Large Panel Dent"
        return "Irregular Impact Dent"

    @staticmethod
    def _classify_scratch(length_px: float, paint_disruption: float) -> str:
        if length_px < 80 and paint_disruption < 1.7:
            return "Light Surface Scratch"
        if length_px < 160:
            return "Medium Scratch"
        if paint_disruption >= 2.5:
            return "Deep Scratch (Paint Through)"
        return "Long Surface Scratch"

    @staticmethod
    def _recommend_dent(is_deep: bool, is_damaged: bool) -> str:
        if not is_deep and not is_damaged:  return "Paintless Dent Repair (PDR)"
        if     is_deep and not is_damaged:  return "Standard Dent Repair"
        if not is_deep and     is_damaged:  return "PDR + Touch-up Paint"
        return "Full Bodywork + Respray"

    @staticmethod
    def _recommend_scratch(paint_disruption: float, area_px: int) -> str:
        if paint_disruption < 1.7 and area_px < 3000:
            return "Light Scratch Polish"
        if paint_disruption < 2.5:
            return "Scratch Touch-up Paint"
        return "Scratch Panel Repaint"

    @staticmethod
    def _aggregate_cost(damages: list) -> str:
        if not damages:
            return "$0"
        totals = []
        for d in damages:
            rec = d["recommendation"]
            if rec in REPAIR_COSTS:
                lo, hi, _ = REPAIR_COSTS[rec]
                totals.append((lo, hi))
        if not totals:
            return "$0"
        return f"${sum(t[0] for t in totals)}-${sum(t[1] for t in totals)}"

    @staticmethod
    def _draw_label(img: np.ndarray, text: str, pos: tuple, color: tuple) -> None:
        font      = cv2.FONT_HERSHEY_SIMPLEX
        scale     = 0.47
        thickness = 1
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        x, y = pos
        y    = max(y, th + 6)
        cv2.rectangle(img, (x-2, y-th-4), (x+tw+4, y+2), (18, 18, 18), cv2.FILLED)
        cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

    @staticmethod
    def _get_wheel_mask(gray: np.ndarray) -> np.ndarray:
        """
        Detects circular wheels/rims using HoughCircles and returns a mask.
        Also identifies dark circular regions in the bottom half of the image.
        """
        h, w = gray.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        
        # 1. Circle Detection
        # Reduce noise for circle detection
        blurred = cv2.medianBlur(gray, 7)
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=w//5,
            param1=50, param2=35, minRadius=int(h*0.08), maxRadius=int(h*0.25)
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # Draw wheel ignore area (slightly larger than detected circle)
                cv2.circle(mask, (i[0], i[1]), int(i[2] * 1.15), 255, -1)
        
        # 2. Bottom-half Tire/Void Filter
        # Wheels are dark and in the bottom 60%
        bottom_half = gray[int(h*0.4):, :]
        # Tire rubber is very dark (intensity < 60)
        _, dark     = cv2.threshold(bottom_half, 60, 255, cv2.THRESH_BINARY_INV)
        # Clean up noise
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(dark)
        
        for j in range(1, num):
            area = stats[j, cv2.CC_STAT_AREA]
            if area < 4000: continue
            
            wid = stats[j, cv2.CC_STAT_WIDTH]
            hei = stats[j, cv2.CC_STAT_HEIGHT]
            ar  = wid / hei
            
            # Wheels/Tires have aspect ratios close to 1.0
            if 0.5 < ar < 2.0:
                y_off = int(h*0.4)
                # Mask the entire bounding box area to be safe
                cv2.rectangle(mask, (stats[j, cv2.CC_STAT_LEFT], stats[j, cv2.CC_STAT_TOP] + y_off),
                                   (stats[j, cv2.CC_STAT_LEFT] + wid, stats[j, cv2.CC_STAT_TOP] + hei + y_off), 255, -1)
        
        return mask

    @staticmethod
    def _to_b64(img: np.ndarray) -> str:
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return base64.b64encode(buf).decode("utf-8")
