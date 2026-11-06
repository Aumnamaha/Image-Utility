# 🛰️ Image Utility — Water Body Segmentation

> **Modular Python pipeline for automated water body segmentation from multi-spectral satellite imagery using NDWI, Otsu thresholding, and morphological filtering — packaged as a reproducible CLI tool with 98 automated tests.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4_x-5C3EE2?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![pytest](https://img.shields.io/badge/tests-98_passed-brightgreen?style=for-the-badge&logo=pytest&logoColor=white)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

---

## 📋 Executive Summary

A modular, open-source Python pipeline for automated water body segmentation from multi-spectral satellite imagery. Combines **NDWI** (Normalized Difference Water Index), **Otsu's automatic thresholding**, and **morphological filtering** — packaged as a reproducible CLI tool with 98 automated tests.

---

<div align="center">

### 🛰️ Water Body Segmentation from Satellite Imagery
**Topic 60 — CSE/AIML Final-Year Project**  
[![GitHub](https://img.shields.io/badge/GitHub-View_Repo-181717?logo=github)](https://github.com/Aumnamaha/Image-Utility)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-98_passed-brightgreen)]()

</div>

A modular, open-source Python pipeline for automated water body segmentation from multi-spectral satellite imagery. Combines **NDWI** (Normalized Difference Water Index), **Otsu's automatic thresholding**, and **morphological filtering** — packaged as a reproducible CLI tool with 98 automated tests.

---

## 🚀 Quick Start (5-Minute Tutorial)

Clone, install, and run the synthetic demo in under five minutes:

```bash
# 1. Clone the repository
git clone https://github.com/Aumnamaha/Image-Utility.git
cd Image-Utility

# 2. Create a virtual environment & install dependencies
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Windows PowerShell
pip install -r requirements.txt        # or: pip install -e ".[dev]"

# 3. Run the instant synthetic demonstration
python main.py --demo --save-plots

# Output: NDWI range, Otsu threshold, water coverage %, JSON metrics report,
#          and a 6-panel dashboard.png saved to ./output/
```

**Expected output:**

```
=== DEMO MODE ===

NDWI range: [-0.4217, 0.5995]
Otsu threshold: 138.00
Water coverage : 50.0%

--- JSON Report ---
{
  "threshold_val": 138.0,
  "water_coverage_pct": 50.0,
  "metrics": {
    "iou": 0.9412,
    "dice": 0.9683,
    "precision": 0.9756,
    "recall": 0.9615
  }
}
```

---

## 📂 Custom Dataset Guide

Replace the synthetic demo with your own Green and NIR spectral bands:

### Prerequisites

- Your data must be co-registered **Green** and **NIR** bands as NumPy `.npy` arrays.
- Both arrays should have the same shape `(H, W)` and contain positive float values (reflectance or DN).
- Optional: a ground-truth binary mask (`gt.npy`, uint8, 0 = non-water / 255 = water) for evaluation.

### Running with custom data

```bash
# Basic pipeline — no evaluation
python main.py --green green_band.npy --nir nir_band.npy \
               --out-dir ./results --save-plots

# With ground-truth evaluation
python main.py --green green_band.npy --nir nir_band.npy --gt gt_mask.npy \
               --out-dir ./results --save-plots
```

### Data preparation tips

| Step | Command / Note |
|---|---|
| **Download** bands from USGS Earth Explorer, Copernicus Open Access Hub, or GEE | GeoTIFF format |
| **Extract** Green (Band 3 for Landsat 8/9) and NIR (Band 5) | Use `rasterio` or GDAL |
| **Convert** to float64 reflectance values [0, 1] | Divide DN by 10000 if using SR products |
| **Save** as `.npy` | `np.save('green_band.npy', green_array)` |

---

## 🏗️ Architecture & Pipeline Explanation

```
┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  Green Band  │────▶│ Gaussian Smooth │────▶│   NDWI Computation│
│  (H×W, float)│     │  (optional)      │     │                  │
└──────────────┘     └─────────────────┘     │  NDWI = (G-N)/(G+N)│
                                               └────────┬─────────┘
                                                        │
                                               ┌────────▼─────────┐
                                               │ Otsu Thresholding │◀── automatic, no manual tuning
                                               └────────┬─────────┘
                                                        │
                                               ┌────────▼─────────┐
                                               │ Morphological     │
                                               │ Opening + Closing │
                                               └────────┬─────────┘
                                                        │
                                               ┌────────▼─────────┐
                                               │ Evaluation &      │
                                               │ Visualization     │
                                               │ (IoU/Dice/P/R)    │
                                               └──────────────────┘
```

### Pipeline stages in detail

| Stage | Method | Key parameter | Purpose |
|---|---|---|---|
| **1. Smoothing** | Gaussian blur (`skimage.filters.gaussian`) | `kernel_size`, `sigma` | Suppress sensor noise & high-frequency texture |
| **2. NDWI Index** | $(G - \text{NIR}) / (G + \text{NIR} + \varepsilon)$ | $\varepsilon = 10^{-5}$ | Spectral separation: water → positive, land → negative |
| **3. Thresholding** | Otsu's method (`cv2.THRESH_OTSU`) | automatic | Global threshold maximising inter-class variance |
| **4. Morphology** | Opening (erosion→dilation) + Closing (dilation→erosion) | Ellipse `(3, 3)` | Remove noise specks & fill interior gaps |
| **5. Evaluation** | IoU / Dice / Precision / Recall | — | Quantitative accuracy against ground truth |

---

## 🐍 Python API Usage Examples

Use the pipeline as a library in your own scripts:

### Basic segmentation

```python
import numpy as np
from src.water_segmentation import WaterSegmentationPipeline

# Load your Green and NIR bands (float64, same shape)
green = np.load("green_band.npy")
nir   = np.load("nir_band.npy")

# Compute NDWI
ndwi = WaterSegmentationPipeline.compute_ndwi(green, nir)
print(f"NDWI range: [{ndwi.min():.3f}, {ndwi.max():.3f}]")

# Segment water (returns dict with raw_ndwi, threshold_val, binary_mask, cleaned_mask)
pipeline = WaterSegmentationPipeline(kernel_size=(3, 3))
result = pipeline.segment_water(ndwi)

cleaned_mask = result["cleaned_mask"]   # uint8, 0=non-water / 255=water
threshold    = result["threshold_val"]  # Otsu threshold on [0, 255] scale

print(f"Otsu threshold : {threshold:.1f}")
print(f"Water coverage : {cleaned_mask[cleaned_mask > 0].size / cleaned_mask.size * 100:.1f}%")
```

### Evaluation against ground truth

```python
from src.metrics import evaluate_segmentation

ground_truth = np.load("gt_mask.npy")          # uint8, same shape as NDWI map
metrics      = evaluate_segmentation(ground_truth, cleaned_mask)

for name, value in metrics.items():
    print(f"  {name:>10s}: {value:.4f}")
# Output:
#         iou: 0.9412
#        dice: 0.9683
#   precision: 0.9756
#      recall: 0.9615
```

### Visual dashboard export

```python
from src.visualizer import plot_segmentation_results

rgb_dummy = np.stack([green * 255] * 3, axis=-1).astype(np.uint8)
plot_segmentation_results(
    rgb_image=rgb_dummy,
    ndwi_map=ndwi,
    ground_truth=ground_truth,
    thresholded=result["binary_mask"],
    cleaned=cleaned_mask,
    save_path="dashboard.png",   # omit to display interactively
)
```

---

## 🧪 Testing & Verification

The project ships with **98 automated tests** covering every module. Run the full suite:

```bash
# Activate your virtual environment first (if not already active):
# .\.venv\Scripts\Activate.ps1

pytest tests/ -v
```

Expected output: `98 passed in ~7s` with zero failures and one minor Pydantic deprecation warning (unrelated to project code).

Individual test files:

| File | Scope |
|---|---|
| `tests/test_smoke.py` | Synthetic image fixture validation |
| `tests/test_filters.py` | Gaussian blur, median filter, CLAHE, contrast stretching |
| `tests/test_color_processing.py` | RGB↔HSV/LAB/GRAY conversions, colour masking, region extraction |
| `tests/test_water_segmentation.py` | NDWI computation, Otsu thresholding, morphological cleanup |
| `tests/test_metrics.py` | IoU, Dice, Precision, Recall — perfect / zero / partial overlap |
| `tests/test_e2e_pipeline.py` | Full end-to-end integration + CLI demo mode |

---

## 📖 Documentation

| Resource | Description |
|---|---|
| [**CASE_STUDY_REPORT.md**](docs/CASE_STUDY_REPORT.md) | Full 6-section academic case study (Problem, Methodology, Implementation, Results, Conclusion, References) |
| [**CASE_STUDY_REPORT.pdf**](docs/CASE_STUDY_REPORT.pdf) | PDF version of the case study report |

---

## 📁 Project Structure

```
Image-Utility/
├── main.py                      # CLI entry point (argparse: --demo, --green, --nir, --out-dir, --save-plots)
├── pyproject.toml               # Build config + pytest / ruff / black settings
├── requirements.txt             # Runtime dependencies
├── README.md                    # This file
├── docs/
│   ├── CASE_STUDY_REPORT.md     # Full academic case study (Markdown)
│   └── CASE_STUDY_REPORT.pdf    # PDF version of the report
├── src/
│   ├── __init__.py              # Package init, exports ImageConfig
│   ├── config.py                # Pydantic-settings global configuration
│   ├── filters.py               # Gaussian blur, median filter, CLAHE, contrast stretching
│   ├── color_processing.py      # RGB↔HSV/LAB/GRAY conversions & colour masking
│   ├── water_segmentation.py    # NDWI + Otsu thresholding + morphological cleanup
│   ├── metrics.py               # IoU / Dice / Precision / Recall evaluation
│   └── visualizer.py            # 6-panel Matplotlib dashboard renderer
└── tests/
    ├── conftest.py              # Synthetic image fixtures (RGB, multi-spectral)
    ├── test_smoke.py            # Fixture validation smoke tests
    ├── test_filters.py          # Spatial filter unit tests (32 tests)
    ├── test_color_processing.py # Colour conversion & masking tests (20 tests)
    ├── test_water_segmentation.py # NDWI + Otsu + morphology tests (19 tests)
    ├── test_metrics.py          # Segmentation metrics tests (10 tests)
    └── test_e2e_pipeline.py     # Full integration + CLI demo tests (10 tests)
```

---

## 📜 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

*Built with NumPy, OpenCV, scikit-image, Matplotlib & Pydantic. 98 tests · Zero external data dependency.*
