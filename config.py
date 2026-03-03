"""
DrawQuantPDF - Configuration
Color ranges (HSV) and application settings.
"""

import os

APP_NAME = "DrawQuantPDF"
APP_VERSION = "1.0.0"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

SCALE = 100  # 1:100

# HSV color ranges for detection
# Format: (H_min, S_min, V_min), (H_max, S_max, V_max)
COLOR_RANGES = {
    "facade": {
        "cyan": {"lower": (80, 80, 80), "upper": (105, 255, 255)},
        "blue": {"lower": (100, 80, 80), "upper": (130, 255, 255)},
    },
    "window": {
        "yellow": {"lower": (18, 80, 80), "upper": (35, 255, 255)},
        "green": {"lower": (35, 60, 60), "upper": (85, 255, 255)},
    },
    "door": {
        "orange": {"lower": (5, 100, 100), "upper": (20, 255, 255)},
        "red_low": {"lower": (0, 100, 100), "upper": (8, 255, 255)},
        "red_high": {"lower": (165, 100, 100), "upper": (180, 255, 255)},
        "magenta": {"lower": (140, 60, 60), "upper": (170, 255, 255)},
    },
}

# Minimum contour area (in pixels) to be considered a valid region
MIN_FACADE_AREA_PX = 5000
MIN_WINDOW_AREA_PX = 800
MIN_DOOR_AREA_PX = 800

# OCR settings
OCR_LANGUAGES = ["ro", "en"]
OCR_GPU = False

# Excel export
EXCEL_TEMPLATE_HEADERS = [
    "Nr. Crt.",
    "Fatada",
    "Element",
    "Tip",
    "Latime (m)",
    "Inaltime (m)",
    "Suprafata (m²)",
]

# PDF conversion DPI
PDF_DPI = 254  # ~1:100 on A3 paper
