"""
Generate fixture images for evaluation.
Uses HSV ranges from config to create detectable regions.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
import numpy as np

import config

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images")
GT_DIR = os.path.join(os.path.dirname(__file__), "gt")


def _hsv_to_bgr(h, s, v):
    """Convert HSV to BGR for cv2."""
    arr = np.array([[[h, s, v]]], dtype=np.uint8)
    bgr = cv2.cvtColor(arr, cv2.COLOR_HSV2BGR)
    return tuple(int(x) for x in bgr[0, 0])


def make_fixture_facade():
    """Create simple facade + window image. 400x300."""
    h, w = 300, 400
    img = np.ones((h, w, 3), dtype=np.uint8) * 200  # light gray bg

    # Cyan facade (HSV ~90, 150, 200) - inside cyan range
    cyan = _hsv_to_bgr(90, 150, 200)
    cv2.rectangle(img, (40, 40), (360, 260), cyan, -1)

    # Yellow window inside facade (HSV ~25, 200, 230)
    yellow = _hsv_to_bgr(25, 200, 230)
    cv2.rectangle(img, (140, 110), (260, 190), yellow, -1)

    os.makedirs(IMAGES_DIR, exist_ok=True)
    path = os.path.join(IMAGES_DIR, "fixture_facade.png")
    cv2.imwrite(path, img)
    return path, (40, 40, 320, 220), (140, 110, 120, 80)


def make_fixture_window_on_roof():
    """
    Create image with window in facade + window in roof zone.
    Roof window should be filtered (top 12% of image).
    """
    h, w = 400, 500
    img = np.ones((h, w, 3), dtype=np.uint8) * 200

    # Facade: large cyan region (lower 75% of image)
    cyan = _hsv_to_bgr(90, 150, 200)
    cv2.rectangle(img, (50, 80), (450, 380), cyan, -1)

    # Valid window inside facade (center)
    yellow = _hsv_to_bgr(25, 200, 230)
    cv2.rectangle(img, (200, 180), (300, 260), yellow, -1)

    # Roof window: top 10% of image (y=20..60)
    cv2.rectangle(img, (350, 25), (420, 55), yellow, -1)

    os.makedirs(IMAGES_DIR, exist_ok=True)
    path = os.path.join(IMAGES_DIR, "fixture_window_on_roof.png")
    cv2.imwrite(path, img)
    return path


if __name__ == "__main__":
    make_fixture_facade()
    make_fixture_window_on_roof()
    print("Fixtures created in", IMAGES_DIR)
