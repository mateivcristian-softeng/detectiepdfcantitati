"""
DrawQuantPDF - PDF Converter
Converts PNG/JPG images to PDF maintaining 1:100 scale.
"""

import os
from PIL import Image
import img2pdf
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas as rl_canvas


def png_to_pdf_simple(input_path: str, output_path: str) -> str:
    """Convert an image to PDF preserving original dimensions using img2pdf."""
    with open(output_path, "wb") as f:
        f.write(img2pdf.convert(input_path))
    return output_path


def png_to_pdf_scaled(input_path: str, output_path: str,
                      scale: int = 100, target_dpi: int = 254) -> str:
    """Convert image to PDF at a specific scale (default 1:100).

    At 1:100, 1 meter real = 10mm on paper.
    The image DPI determines how many pixels map to these physical dimensions.
    """
    img = Image.open(input_path)
    width_px, height_px = img.size

    original_dpi = img.info.get("dpi", (target_dpi, target_dpi))
    if isinstance(original_dpi, tuple):
        dpi_x, dpi_y = original_dpi
    else:
        dpi_x = dpi_y = original_dpi

    if dpi_x == 0:
        dpi_x = target_dpi
    if dpi_y == 0:
        dpi_y = target_dpi

    width_inches = width_px / dpi_x
    height_inches = height_px / dpi_y

    width_mm = width_inches * 25.4
    height_mm = height_inches * 25.4

    page_w = width_mm * mm
    page_h = height_mm * mm

    c = rl_canvas.Canvas(output_path, pagesize=(page_w, page_h))
    c.drawImage(input_path, 0, 0, width=page_w, height=page_h,
                preserveAspectRatio=True, anchor="sw")
    c.save()
    return output_path


def png_to_pdf_with_reference(input_path: str, output_path: str,
                              ref_pixels: int, ref_meters: float,
                              scale: int = 100) -> str:
    """Convert image to PDF using a known reference dimension for calibration.

    Args:
        ref_pixels: Number of pixels spanning the reference dimension
        ref_meters: Real-world size of the reference dimension in meters
        scale: Drawing scale (100 = 1:100)
    """
    img = Image.open(input_path)
    width_px, height_px = img.size

    paper_mm_per_meter = 1000.0 / scale  # e.g. 10mm on paper = 1m real
    ref_on_paper_mm = ref_meters * paper_mm_per_meter
    effective_dpi = ref_pixels / (ref_on_paper_mm / 25.4)

    width_mm = (width_px / effective_dpi) * 25.4
    height_mm = (height_px / effective_dpi) * 25.4

    page_w = width_mm * mm
    page_h = height_mm * mm

    c = rl_canvas.Canvas(output_path, pagesize=(page_w, page_h))
    c.drawImage(input_path, 0, 0, width=page_w, height=page_h,
                preserveAspectRatio=True, anchor="sw")

    c.setFont("Helvetica", 6)
    c.drawString(5 * mm, 3 * mm,
                 f"Scara 1:{scale} | Ref: {ref_meters}m = {ref_pixels}px | "
                 f"DPI efectiv: {effective_dpi:.0f}")
    c.save()
    return output_path


def batch_convert(input_dir: str, output_dir: str, **kwargs) -> list:
    """Convert all images in a directory to PDF."""
    os.makedirs(output_dir, exist_ok=True)
    converted = []
    for fname in os.listdir(input_dir):
        if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            input_path = os.path.join(input_dir, fname)
            pdf_name = os.path.splitext(fname)[0] + ".pdf"
            output_path = os.path.join(output_dir, pdf_name)
            png_to_pdf_scaled(input_path, output_path, **kwargs)
            converted.append(output_path)
    return converted
