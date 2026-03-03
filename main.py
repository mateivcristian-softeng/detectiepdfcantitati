"""
DrawQuantPDF - Main Entry Point
Professional application for detecting and quantifying facade elements
from architectural drawings.
"""

import sys
import os
import io

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    from gui.app import DrawQuantPDFApp
    app = DrawQuantPDFApp()
    app.mainloop()


if __name__ == "__main__":
    main()
