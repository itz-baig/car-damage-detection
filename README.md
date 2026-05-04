# DentScan AI: Professional Car Damage Detection & Repair Estimator

**DentScan AI** is a state-of-the-art vehicle inspection system that combines Deep Learning (YOLOv8) with advanced Computer Vision to detect, classify, and estimate repair costs for automotive cosmetic damage.

![DentScan AI](https://img.shields.io/badge/AI-Computer%20Vision-blueviolet)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688)
![OpenCV](https://img.shields.io/badge/Processing-OpenCV-green)

---

## 🚀 Key Features

- **Hybrid Detection Engine**: Uses YOLOv8-segmentation for vehicle body identification and an enhanced Fallback CV engine for micro-damage detection.
- **Intelligent ROI Masking**: Automatically identifies and ignores non-paint regions such as:
  - Windows and Windshields
  - Tires and Rims
  - Tail Lights and Headlights
  - Ground/Floor shadows
- **Dynamic Sensitivity Slider**: Real-time precision control to tune the AI for different lighting conditions and paint finishes (Gloss vs. Matte).
- **Multi-Angle Support**: Batch process up to 6 photos of a vehicle to generate a comprehensive 360° damage report.
- **Automated Repair Estimator**: Calculates repair costs and labor hours based on damage area, type (dent vs. scratch), and severity.
- **Glassmorphism UI**: A premium, responsive web interface for easy analysis.

---

## 🛠️ Technical Stack

- **Backend**: Python, FastAPI, Uvicorn
- **AI/ML**: Ultralytics YOLOv8 (Segmentation), NumPy
- **Image Processing**: OpenCV (Bilateral Filtering, CLAHE, Hough Transforms, Canny Analysis)
- **Frontend**: Vanilla JS, Modern CSS (Glassmorphism), HTML5

---

## 📦 Installation & Setup

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd dent_detection
```

### 2. Install Dependencies

Ensure you have Python 3.8+ installed.

```bash
pip install fastapi uvicorn opencv-python numpy ultralytics pillow
```

### 3. Run the Server

```bash
python -m uvicorn api_v2:app --host 127.0.0.1 --port 8001
```

### 4. Access the App

Open your browser and navigate to:
**[http://127.0.0.1:8001](http://127.0.0.1:8001)**

---

## ⚙️ How it Works

1. **Preprocessing**: The system applies a **Bilateral Filter** to smooth out reflections while preserving the sharp edges of dents and scratches.
2. **ROI Extraction**: YOLOv8 segments the car body. Any region outside the car (or identified as a window/wheel) is masked out.
3. **Multi-Pass Detection**:
   - **Strategy A (Sharp Lines)**: Detects creases and sharp scratches using Canny edge analysis.
   - **Strategy B (Surface Distortion)**: Detects smooth collision dents by analyzing shadow-highlight pairs relative to local paint intensity.
4. **Scuff Analysis**: A sensitive secondary pass identifies clusters of micro-scratches typical on bumpers.
5. **Report Generation**: Data is merged, severity is calculated (0-100), and a repairman-level JSON report is returned to the UI.

---

## 📊 Detection Tuning

| Sensitivity                       | Best For...                | Description                                              |
| :-------------------------------- | :------------------------- | :------------------------------------------------------- |
| **10% - 40% (Strict)**      | Glossy Cars, High Sunlight | Filters out building/sky reflections and panel gaps.     |
| **70% - 85% (Optimal)**     | Standard Use               | Balanced detection for most dents and scuffs.            |
| **90% - 100% (Aggressive)** | Matte Paint, Indoor        | Finds very faint metal deformations and micro-scratches. |
