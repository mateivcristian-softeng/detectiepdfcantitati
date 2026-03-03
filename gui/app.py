"""
DrawQuantPDF - Professional GUI Application
Modern interface built with CustomTkinter for facade quantity analysis.
"""

import os
import sys
import threading
import traceback
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np

import customtkinter as ctk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from core.color_detector import ColorDetector
from core.ocr_engine import OCREngine
from core.area_calculator import AreaCalculator, FacadeReport, ProjectReport
from core.pdf_converter import png_to_pdf_scaled, png_to_pdf_with_reference
from core.excel_exporter import ExcelExporter
from core.pipeline import AnalysisPipeline

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class DrawQuantPDFApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title(f"{config.APP_NAME} v{config.APP_VERSION}")
        self.geometry("1500x900")
        self.minsize(1200, 700)

        self.original_image = None
        self.display_image = None
        self.current_file = None
        self.detection_results = None
        self.project_report = None
        self.photo_image = None

        self.detector = ColorDetector()
        self.ocr_engine = OCREngine()
        self.calculator = AreaCalculator()
        self.exporter = ExcelExporter()
        self.pipeline = AnalysisPipeline()

        self._build_ui()

    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_center()
        self._build_right_panel()
        self._build_statusbar()

    # ── Sidebar (Left Panel) ──────────────────────────────────────────

    def _build_sidebar(self):
        sidebar = ctk.CTkFrame(self, width=280, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nswe")
        sidebar.grid_rowconfigure(10, weight=1)

        logo_label = ctk.CTkLabel(
            sidebar, text="DrawQuantPDF",
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        logo_label.grid(row=0, column=0, padx=20, pady=(20, 5))

        version_label = ctk.CTkLabel(
            sidebar, text=f"v{config.APP_VERSION} - Analiza Cantitati",
            font=ctk.CTkFont(size=11), text_color="gray",
        )
        version_label.grid(row=1, column=0, padx=20, pady=(0, 20))

        # ── File section
        file_frame = ctk.CTkFrame(sidebar)
        file_frame.grid(row=2, column=0, padx=15, pady=5, sticky="ew")

        ctk.CTkLabel(file_frame, text="FISIER IMAGINE",
                     font=ctk.CTkFont(size=10, weight="bold"),
                     text_color="#4A9EFF").grid(
            row=0, column=0, padx=10, pady=(8, 2), sticky="w"
        )

        self.btn_load = ctk.CTkButton(
            file_frame, text="Incarca Imagine",
            command=self._load_image, height=36,
        )
        self.btn_load.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

        self.lbl_filename = ctk.CTkLabel(
            file_frame, text="Niciun fisier incarcat",
            font=ctk.CTkFont(size=10), text_color="gray",
            wraplength=230,
        )
        self.lbl_filename.grid(row=2, column=0, padx=10, pady=(0, 8))

        # ── Analysis section
        analysis_frame = ctk.CTkFrame(sidebar)
        analysis_frame.grid(row=3, column=0, padx=15, pady=5, sticky="ew")

        ctk.CTkLabel(analysis_frame, text="ANALIZA",
                     font=ctk.CTkFont(size=10, weight="bold"),
                     text_color="#4A9EFF").grid(
            row=0, column=0, padx=10, pady=(8, 2), sticky="w"
        )

        self.btn_detect = ctk.CTkButton(
            analysis_frame, text="Detecteaza Regiuni",
            command=self._run_detection, height=36,
            fg_color="#2B7539", hover_color="#1E5428",
        )
        self.btn_detect.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

        self.btn_ocr = ctk.CTkButton(
            analysis_frame, text="Extrage Text (OCR)",
            command=self._run_ocr, height=36,
            fg_color="#7B5EA7", hover_color="#5E4580",
        )
        self.btn_ocr.grid(row=2, column=0, padx=10, pady=5, sticky="ew")

        self.btn_calculate = ctk.CTkButton(
            analysis_frame, text="Calculeaza Suprafete",
            command=self._run_calculation, height=36,
            fg_color="#B8860B", hover_color="#8B6508",
        )
        self.btn_calculate.grid(row=3, column=0, padx=10, pady=(5, 8), sticky="ew")

        self.btn_full_pipeline = ctk.CTkButton(
            analysis_frame, text="ANALIZA COMPLETA",
            command=self._run_full_pipeline, height=42,
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        self.btn_full_pipeline.grid(row=4, column=0, padx=10, pady=(5, 10),
                                    sticky="ew")

        # ── Export section
        export_frame = ctk.CTkFrame(sidebar)
        export_frame.grid(row=4, column=0, padx=15, pady=5, sticky="ew")

        ctk.CTkLabel(export_frame, text="EXPORT",
                     font=ctk.CTkFont(size=10, weight="bold"),
                     text_color="#4A9EFF").grid(
            row=0, column=0, padx=10, pady=(8, 2), sticky="w"
        )

        self.btn_excel = ctk.CTkButton(
            export_frame, text="Export Excel",
            command=self._export_excel, height=36,
            fg_color="#1B6B3A", hover_color="#0E4D28",
        )
        self.btn_excel.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

        self.btn_pdf = ctk.CTkButton(
            export_frame, text="Converteste in PDF",
            command=self._convert_pdf, height=36,
            fg_color="#CC3333", hover_color="#991F1F",
        )
        self.btn_pdf.grid(row=2, column=0, padx=10, pady=(5, 10), sticky="ew")

        # ── Scale settings
        scale_frame = ctk.CTkFrame(sidebar)
        scale_frame.grid(row=5, column=0, padx=15, pady=5, sticky="ew")

        ctk.CTkLabel(scale_frame, text="SCARA DESEN",
                     font=ctk.CTkFont(size=10, weight="bold"),
                     text_color="#4A9EFF").grid(
            row=0, column=0, columnspan=2, padx=10, pady=(8, 2), sticky="w"
        )

        ctk.CTkLabel(scale_frame, text="Scara 1:",
                     font=ctk.CTkFont(size=11)).grid(
            row=1, column=0, padx=(10, 2), pady=5, sticky="e"
        )
        self.entry_scale = ctk.CTkEntry(scale_frame, width=80)
        self.entry_scale.insert(0, "100")
        self.entry_scale.grid(row=1, column=1, padx=(2, 10), pady=5, sticky="w")

        ctk.CTkLabel(scale_frame, text="Ref. (m):",
                     font=ctk.CTkFont(size=11)).grid(
            row=2, column=0, padx=(10, 2), pady=5, sticky="e"
        )
        self.entry_ref_m = ctk.CTkEntry(scale_frame, width=80,
                                        placeholder_text="11.30")
        self.entry_ref_m.grid(row=2, column=1, padx=(2, 10), pady=(5, 10),
                              sticky="w")

        # ── Appearance
        ctk.CTkLabel(sidebar, text="Aspect:",
                     font=ctk.CTkFont(size=10)).grid(
            row=11, column=0, padx=20, pady=(5, 0), sticky="w"
        )
        self.appearance_menu = ctk.CTkOptionMenu(
            sidebar, values=["Dark", "Light", "System"],
            command=self._change_appearance,
        )
        self.appearance_menu.grid(row=12, column=0, padx=20, pady=(5, 20),
                                  sticky="ew")

    # ── Center Panel (Image Viewer) ───────────────────────────────────

    def _build_center(self):
        center = ctk.CTkFrame(self, corner_radius=10)
        center.grid(row=0, column=1, padx=10, pady=(10, 0), sticky="nswe")
        center.grid_columnconfigure(0, weight=1)
        center.grid_rowconfigure(1, weight=1)

        toolbar = ctk.CTkFrame(center, height=40)
        toolbar.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        self.btn_zoom_in = ctk.CTkButton(
            toolbar, text="+", width=35, height=30,
            command=lambda: self._zoom(1.2),
        )
        self.btn_zoom_in.grid(row=0, column=0, padx=2)

        self.btn_zoom_out = ctk.CTkButton(
            toolbar, text="-", width=35, height=30,
            command=lambda: self._zoom(0.8),
        )
        self.btn_zoom_out.grid(row=0, column=1, padx=2)

        self.btn_fit = ctk.CTkButton(
            toolbar, text="Fit", width=50, height=30,
            command=self._fit_image,
        )
        self.btn_fit.grid(row=0, column=2, padx=2)

        self.btn_toggle_detections = ctk.CTkButton(
            toolbar, text="Arata Detectii", width=120, height=30,
            command=self._toggle_detections,
        )
        self.btn_toggle_detections.grid(row=0, column=3, padx=10)

        self.lbl_zoom = ctk.CTkLabel(toolbar, text="100%",
                                     font=ctk.CTkFont(size=10))
        self.lbl_zoom.grid(row=0, column=4, padx=5)

        self.canvas_frame = ctk.CTkFrame(center)
        self.canvas_frame.grid(row=1, column=0, sticky="nswe", padx=5, pady=5)
        self.canvas_frame.grid_columnconfigure(0, weight=1)
        self.canvas_frame.grid_rowconfigure(0, weight=1)

        import tkinter as tk
        self.canvas = tk.Canvas(self.canvas_frame, bg="#1a1a2e",
                                highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nswe")

        self.scroll_x = tk.Scrollbar(self.canvas_frame, orient="horizontal",
                                     command=self.canvas.xview)
        self.scroll_x.grid(row=1, column=0, sticky="ew")
        self.scroll_y = tk.Scrollbar(self.canvas_frame, orient="vertical",
                                     command=self.canvas.yview)
        self.scroll_y.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(xscrollcommand=self.scroll_x.set,
                              yscrollcommand=self.scroll_y.set)

        self.zoom_level = 1.0
        self.show_detections = False

        self.canvas.bind("<Configure>", lambda e: self._fit_image())

    # ── Right Panel (Results) ─────────────────────────────────────────

    def _build_right_panel(self):
        right = ctk.CTkFrame(self, width=380, corner_radius=10)
        right.grid(row=0, column=2, padx=(0, 10), pady=(10, 0), sticky="nswe")
        right.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(right, text="REZULTATE ANALIZA",
                     font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=0, column=0, padx=15, pady=10
        )

        self.results_text = ctk.CTkTextbox(right, width=360,
                                           font=ctk.CTkFont(family="Consolas",
                                                            size=11))
        self.results_text.grid(row=1, column=0, padx=10, pady=(0, 5),
                               sticky="nswe")
        self.results_text.insert("end",
                                 "Incarcati o imagine si rulati analiza\n"
                                 "pentru a vedea rezultatele aici.\n\n"
                                 "Pasi:\n"
                                 "1. Incarcati imaginea fatadei\n"
                                 "2. Apasati 'ANALIZA COMPLETA'\n"
                                 "3. Verificati rezultatele\n"
                                 "4. Exportati in Excel / PDF")

        # ── Manual edit section
        edit_frame = ctk.CTkFrame(right)
        edit_frame.grid(row=2, column=0, padx=10, pady=5, sticky="ew")

        ctk.CTkLabel(edit_frame, text="EDITARE MANUALA",
                     font=ctk.CTkFont(size=10, weight="bold"),
                     text_color="#4A9EFF").grid(
            row=0, column=0, columnspan=3, padx=10, pady=(8, 2), sticky="w"
        )

        self.btn_add_facade = ctk.CTkButton(
            edit_frame, text="+ Fatada", width=90, height=28,
            command=self._add_facade_manual,
            fg_color="#2B5278",
        )
        self.btn_add_facade.grid(row=1, column=0, padx=5, pady=5)

        self.btn_add_window = ctk.CTkButton(
            edit_frame, text="+ Fereastra", width=90, height=28,
            command=self._add_window_manual,
            fg_color="#785A2B",
        )
        self.btn_add_window.grid(row=1, column=1, padx=5, pady=5)

        self.btn_add_door = ctk.CTkButton(
            edit_frame, text="+ Usa", width=90, height=28,
            command=self._add_door_manual,
            fg_color="#78432B",
        )
        self.btn_add_door.grid(row=1, column=2, padx=5, pady=(5, 10))

    # ── Status Bar ────────────────────────────────────────────────────

    def _build_statusbar(self):
        status_frame = ctk.CTkFrame(self, height=30, corner_radius=0)
        status_frame.grid(row=1, column=0, columnspan=3, sticky="ew")

        self.lbl_status = ctk.CTkLabel(
            status_frame, text="Gata.",
            font=ctk.CTkFont(size=10),
        )
        self.lbl_status.grid(row=0, column=0, padx=15, pady=3, sticky="w")

        self.progress = ctk.CTkProgressBar(status_frame, width=200)
        self.progress.grid(row=0, column=1, padx=15, pady=3, sticky="e")
        self.progress.set(0)

    # ── Actions ───────────────────────────────────────────────────────

    def _set_status(self, text: str, progress: float = None):
        self.lbl_status.configure(text=text)
        if progress is not None:
            self.progress.set(progress)
        self.update_idletasks()

    def _load_image(self):
        filetypes = [
            ("Imagini", "*.png *.jpg *.jpeg *.bmp *.tiff"),
            ("PDF", "*.pdf"),
            ("Toate fisierele", "*.*"),
        ]
        path = filedialog.askopenfilename(filetypes=filetypes)
        if not path:
            return

        self._set_status(f"Se incarca: {os.path.basename(path)}...", 0.2)
        try:
            self.original_image = cv2.imread(path)
            if self.original_image is None:
                messagebox.showerror("Eroare", f"Nu s-a putut incarca:\n{path}")
                return

            self.current_file = path
            self.lbl_filename.configure(
                text=os.path.basename(path),
                text_color="white",
            )
            self.detection_results = None
            self.project_report = None
            self.show_detections = False
            self._display_image(self.original_image)
            self._set_status(
                f"Incarcat: {self.original_image.shape[1]}x"
                f"{self.original_image.shape[0]}px", 1.0
            )
        except Exception as e:
            messagebox.showerror("Eroare", f"Eroare la incarcare:\n{e}")
            self._set_status("Eroare la incarcare.", 0)

    def _display_image(self, cv_image: np.ndarray):
        rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w < 10:
            canvas_w = 800
        if canvas_h < 10:
            canvas_h = 600

        img_w, img_h = pil_img.size
        display_w = int(img_w * self.zoom_level)
        display_h = int(img_h * self.zoom_level)

        pil_resized = pil_img.resize((display_w, display_h), Image.LANCZOS)
        self.photo_image = ImageTk.PhotoImage(pil_resized)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo_image)
        self.canvas.configure(scrollregion=(0, 0, display_w, display_h))

        zoom_pct = int(self.zoom_level * 100)
        self.lbl_zoom.configure(text=f"{zoom_pct}%")

    def _zoom(self, factor):
        self.zoom_level *= factor
        self.zoom_level = max(0.1, min(5.0, self.zoom_level))
        if self.original_image is not None:
            img = self.original_image
            if self.show_detections and self.detection_results:
                img = self.detector.draw_detections(img, self.detection_results)
            self._display_image(img)

    def _fit_image(self):
        if self.original_image is None:
            return
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w < 10 or canvas_h < 10:
            return
        h, w = self.original_image.shape[:2]
        scale_w = canvas_w / w
        scale_h = canvas_h / h
        self.zoom_level = min(scale_w, scale_h) * 0.95
        img = self.original_image
        if self.show_detections and self.detection_results:
            img = self.detector.draw_detections(img, self.detection_results)
        self._display_image(img)

    def _toggle_detections(self):
        if self.original_image is None:
            return
        self.show_detections = not self.show_detections
        if self.show_detections and self.detection_results:
            viz = self.detector.draw_detections(
                self.original_image, self.detection_results
            )
            self._display_image(viz)
            self.btn_toggle_detections.configure(text="Ascunde Detectii")
        else:
            self._display_image(self.original_image)
            self.btn_toggle_detections.configure(text="Arata Detectii")

    def _run_detection(self):
        if self.original_image is None:
            messagebox.showwarning("Atentie", "Incarcati mai intai o imagine.")
            return

        self._set_status("Se detecteaza regiunile colorate...", 0.2)

        def task():
            try:
                results = self.detector.detect_all(self.original_image)
                self.detection_results = results
                n_f = len(results["facades"])
                n_w = len(results["windows"])
                n_d = len(results["doors"])

                self.after(0, lambda: self._on_detection_done(n_f, n_w, n_d))
            except Exception as e:
                self.after(0, lambda: self._on_error("detectie", e))

        threading.Thread(target=task, daemon=True).start()

    def _on_detection_done(self, n_f, n_w, n_d):
        self.show_detections = True
        viz = self.detector.draw_detections(
            self.original_image, self.detection_results
        )
        self._display_image(viz)
        self.btn_toggle_detections.configure(text="Ascunde Detectii")

        self.results_text.delete("1.0", "end")
        self.results_text.insert("end",
                                 f"DETECTIE COMPLETA\n"
                                 f"{'─' * 35}\n"
                                 f"Fatade detectate:   {n_f}\n"
                                 f"Ferestre detectate: {n_w}\n"
                                 f"Usi detectate:      {n_d}\n\n"
                                 f"Rulati OCR pentru a extrage valorile,\n"
                                 f"sau 'ANALIZA COMPLETA' pentru tot.")
        self._set_status(
            f"Detectie: {n_f} fatade, {n_w} ferestre, {n_d} usi", 1.0
        )

    def _run_ocr(self):
        if self.detection_results is None:
            messagebox.showwarning("Atentie",
                                   "Rulati mai intai detectia regiunilor.")
            return

        self._set_status("Se extrage textul (OCR)... Poate dura...", 0.3)

        def task():
            try:
                all_regions = (
                    self.detection_results["facades"]
                    + self.detection_results["windows"]
                    + self.detection_results["doors"]
                )
                self.ocr_engine.enrich_regions(self.original_image, all_regions)
                self.after(0, self._on_ocr_done)
            except Exception as e:
                self.after(0, lambda: self._on_error("OCR", e))

        threading.Thread(target=task, daemon=True).start()

    def _on_ocr_done(self):
        self.results_text.delete("1.0", "end")
        self.results_text.insert("end", "OCR COMPLET\n" + "─" * 35 + "\n\n")

        for rtype in ["facades", "windows", "doors"]:
            for r in self.detection_results[rtype]:
                self.results_text.insert(
                    "end",
                    f"[{r.region_type.upper()}] {r.label}\n"
                    f"  Suprafata: {r.area_m2 or '?'} m²\n"
                    f"  Text OCR: {r.ocr_text[:80]}\n\n"
                )

        self._set_status("OCR complet.", 1.0)

    def _run_calculation(self):
        if self.detection_results is None:
            messagebox.showwarning("Atentie", "Rulati detectia si OCR mai intai.")
            return

        self.project_report = self.calculator.compute_from_detections(
            self.detection_results
        )
        summary = self.calculator.summary_text()

        self.results_text.delete("1.0", "end")
        self.results_text.insert("end", summary)
        self._set_status("Calcul suprafete complet.", 1.0)

    def _run_full_pipeline(self):
        if self.original_image is None:
            messagebox.showwarning("Atentie", "Incarcati mai intai o imagine.")
            return

        self._set_status("ANALIZA COMPLETA - Pornire pipeline...", 0.1)
        self.btn_full_pipeline.configure(state="disabled")

        def progress_cb(msg, val):
            self.after(0, lambda: self._set_status(msg, val))

        def task():
            try:
                self.pipeline = AnalysisPipeline()
                self.project_report = self.pipeline.run(
                    self.original_image, progress_callback=progress_cb
                )
                self.detection_results = self.pipeline.detection_results
                self.after(0, self._on_full_pipeline_done)
            except Exception as e:
                self.after(0, lambda: self._on_error("analiza completa", e))
                self.after(0, lambda: self.btn_full_pipeline.configure(
                    state="normal"))

        threading.Thread(target=task, daemon=True).start()

    def _on_full_pipeline_done(self):
        self.show_detections = True
        if self.detection_results:
            viz = self.detector.draw_detections(
                self.original_image, self.detection_results
            )
            self._display_image(viz)
        self.btn_toggle_detections.configure(text="Ascunde Detectii")

        summary = self.pipeline.get_summary()
        debug = self.pipeline.get_ocr_debug()

        self.results_text.delete("1.0", "end")
        self.results_text.insert("end", summary)
        self.results_text.insert("end", "\n\n" + "=" * 50 + "\n")
        self.results_text.insert("end", debug)

        self.btn_full_pipeline.configure(state="normal")
        self._set_status("ANALIZA COMPLETA finalizata.", 1.0)

    def _export_excel(self):
        if self.project_report is None:
            messagebox.showwarning("Atentie",
                                   "Rulati mai intai analiza completa.")
            return

        default_name = "cantitati_fatade.xlsx"
        if self.current_file:
            base = os.path.splitext(os.path.basename(self.current_file))[0]
            default_name = f"{base}_cantitati.xlsx"

        path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel", "*.xlsx")],
            initialfile=default_name,
        )
        if not path:
            return

        try:
            project_name = os.path.splitext(
                os.path.basename(self.current_file or "Proiect")
            )[0]
            self.exporter.export(self.project_report, path, project_name)
            self._set_status(f"Excel exportat: {os.path.basename(path)}", 1.0)
            messagebox.showinfo("Succes",
                                f"Fisierul Excel a fost salvat:\n{path}")
        except Exception as e:
            messagebox.showerror("Eroare", f"Eroare la export Excel:\n{e}")

    def _convert_pdf(self):
        if not self.current_file:
            messagebox.showwarning("Atentie", "Incarcati mai intai o imagine.")
            return

        default_name = os.path.splitext(
            os.path.basename(self.current_file)
        )[0] + ".pdf"

        path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf")],
            initialfile=default_name,
        )
        if not path:
            return

        try:
            ref_m = self.entry_ref_m.get().strip()
            scale_val = int(self.entry_scale.get() or 100)

            if ref_m:
                ref_meters = float(ref_m.replace(",", "."))
                h, w = self.original_image.shape[:2]
                png_to_pdf_with_reference(
                    self.current_file, path,
                    ref_pixels=w, ref_meters=ref_meters,
                    scale=scale_val,
                )
            else:
                png_to_pdf_scaled(self.current_file, path, scale=scale_val)

            self._set_status(f"PDF creat: {os.path.basename(path)}", 1.0)
            messagebox.showinfo("Succes", f"PDF salvat:\n{path}")
        except Exception as e:
            messagebox.showerror("Eroare", f"Eroare la conversie PDF:\n{e}")

    # ── Manual entry dialogs ──────────────────────────────────────────

    def _add_facade_manual(self):
        dialog = ManualEntryDialog(
            self, "Adauga Fatada",
            fields=["Nume fatada", "Suprafata totala (m²)"],
        )
        self.wait_window(dialog)
        if dialog.result:
            name, area_str = dialog.result
            area = float(area_str.replace(",", "."))
            if self.project_report is None:
                self.project_report = ProjectReport()
            self.project_report.facades.append(
                FacadeReport(name=name, total_area_m2=area)
            )
            self._refresh_results()

    def _add_window_manual(self):
        if not self.project_report or not self.project_report.facades:
            messagebox.showwarning("Atentie", "Adaugati mai intai o fatada.")
            return

        facade_names = [f.name for f in self.project_report.facades]
        dialog = ManualEntryDialog(
            self, "Adauga Fereastra",
            fields=["Label (ex: F2)", "Suprafata (m²)"],
            combo_field=("Fatada parinte", facade_names),
        )
        self.wait_window(dialog)
        if dialog.result:
            label, area_str = dialog.result[:2]
            facade_name = dialog.combo_value
            area = float(area_str.replace(",", "."))
            for f in self.project_report.facades:
                if f.name == facade_name:
                    f.windows.append((label, area))
                    break
            self._refresh_results()

    def _add_door_manual(self):
        if not self.project_report or not self.project_report.facades:
            messagebox.showwarning("Atentie", "Adaugati mai intai o fatada.")
            return

        facade_names = [f.name for f in self.project_report.facades]
        dialog = ManualEntryDialog(
            self, "Adauga Usa",
            fields=["Label (ex: U1)", "Suprafata (m²)"],
            combo_field=("Fatada parinte", facade_names),
        )
        self.wait_window(dialog)
        if dialog.result:
            label, area_str = dialog.result[:2]
            facade_name = dialog.combo_value
            area = float(area_str.replace(",", "."))
            for f in self.project_report.facades:
                if f.name == facade_name:
                    f.doors.append((label, area))
                    break
            self._refresh_results()

    def _refresh_results(self):
        if self.project_report:
            self.calculator.report = self.project_report
            summary = self.calculator.summary_text()
            self.results_text.delete("1.0", "end")
            self.results_text.insert("end", summary)

    def _on_error(self, step: str, error: Exception):
        self._set_status(f"Eroare la {step}.", 0)
        messagebox.showerror("Eroare",
                             f"Eroare la {step}:\n{error}\n\n"
                             f"{traceback.format_exc()}")

    def _change_appearance(self, mode: str):
        ctk.set_appearance_mode(mode)


class ManualEntryDialog(ctk.CTkToplevel):
    """Dialog for manual data entry."""

    def __init__(self, parent, title: str, fields: list,
                 combo_field: tuple = None):
        super().__init__(parent)
        self.title(title)
        self.geometry("400x300")
        self.transient(parent)
        self.grab_set()

        self.result = None
        self.combo_value = None
        self.entries = []

        ctk.CTkLabel(self, text=title,
                     font=ctk.CTkFont(size=16, weight="bold")).pack(
            padx=20, pady=(15, 10)
        )

        for field_name in fields:
            frame = ctk.CTkFrame(self)
            frame.pack(padx=20, pady=5, fill="x")
            ctk.CTkLabel(frame, text=field_name, width=170).pack(
                side="left", padx=5
            )
            entry = ctk.CTkEntry(frame, width=180)
            entry.pack(side="right", padx=5)
            self.entries.append(entry)

        if combo_field:
            frame = ctk.CTkFrame(self)
            frame.pack(padx=20, pady=5, fill="x")
            ctk.CTkLabel(frame, text=combo_field[0], width=170).pack(
                side="left", padx=5
            )
            self.combo = ctk.CTkComboBox(frame, values=combo_field[1],
                                         width=180)
            self.combo.pack(side="right", padx=5)

        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(padx=20, pady=15)

        ctk.CTkButton(btn_frame, text="Adauga", width=120,
                      command=self._submit).pack(side="left", padx=5)
        ctk.CTkButton(btn_frame, text="Anuleaza", width=120,
                      fg_color="gray", command=self.destroy).pack(
            side="left", padx=5
        )

    def _submit(self):
        values = [e.get().strip() for e in self.entries]
        if all(values):
            self.result = values
            if hasattr(self, "combo"):
                self.combo_value = self.combo.get()
            self.destroy()
