# Foundation Model Plan – Detector ML pentru fațade

## Status curent (2026-03-17)

### Model antrenat și validat
- **Arhitectură**: YOLOv8n (nano, 6.3MB, 3M parametri)
- **Locație**: `models/yolov8n_facade_v2_best.pt`
- **Date antrenament**: 69 imagini adnotate manual cu LabelImg (59 train / 10 val)
- **Labels**: 290 total (113 window, 26 door, 77 facade, 74 socle)
- **Clase**: facade(0), window(1), door(2), socle(3)
- **Antrenament**: 123 epoci (early stop la 93, best=84), 8h pe CPU i9-12900H
- **Metrici finale (val)**: mAP50 = **0.939**, mAP50-95 = **0.759**

### Test pe ARION (caz foto/fără markup)
ML reduce x-offset-ul față de GT cu **50-60%** comparativ cu detectorul pe reguli:

| Element | GT | Reguli (dx) | ML (dx) | Îmbunătățire |
|---------|-----|-------------|---------|-------------|
| W1 fereastră | (172,544) | +167px | +64px | 2.6x mai precis |
| W4 fereastră | (833,541) | -57px | +55px | similar |
| W5 fereastră | (1475,403) | -51px | +42px | 1.2x |
| W6 fereastră | (1962,392) | -50px | +34px | 1.5x |
| D1 ușă | (521,552) | +105px | +55px | 1.9x |

**Lipsuri**: 2 sidelights mici (<82px), ușa D2 — insuficiente date de antrenament

---

## Plan de acțiune

### Pas 1: Colectare date suplimentare (ACUM)
- Target: **100-120 imagini** total (mai trebuie ~30-50)
- Focus pe: **uși** (doar 26 labels, target 60+) și **ferestre mici/sidelights** (<100px)
- Workflow: LabelImg, labels = `window`, `door`, `facade`, `socle`
- Import: `python tools/eval/import_labelimg_annotations.py --source "CALE"`

### Pas 2: Re-antrenare pe date complete
- Pe GPU (Colab T4 gratuit): ~30 min vs 8h pe CPU
- Script: vezi secțiunea de mai jos

### Pas 3: Integrare hybrid în pipeline
- Markup de culori → detectorul pe reguli (funcționează)
- Foto/fără markup → modelul ML
- Aplicația decide automat pe baza prezenței culorilor convenționale

---

## Variante de model

| Model | Dimensiune | Use case |
|-------|------------|----------|
| **YOLOv8n** (actual) | 6.3 MB | Rapid, resurse minime, funcționează pe CPU |
| YOLOv8s | ~22 MB | Mai precis, necesită mai mult RAM |
| YOLOv8m | ~50 MB | Precizie ridicată, GPU recomandat |

### Recomandare
**YOLOv8n** rămâne cel mai bun compromis: 6.3MB, inference ~350ms pe CPU, suficient de precis cu date adecvate.

---

## Cerințe hardware

### Minim (inference pe CPU)
- RAM: 8 GB
- Disk: ~50 MB (model + dependențe deja instalate)
- Python 3.10+, ultralytics, torch

### Recomandat (antrenament)
- GPU: 4+ GB VRAM (NVIDIA CUDA) sau Google Colab T4 (gratuit)
- RAM: 16 GB

---

## Mod de activare (opt-in)

```bash
# Variabilă de mediu
set DETECTIEPDF_USE_FOUNDATION=1

# Sau în cod
from core.foundation_segmentation import is_available
```

Output-ul ML este compatibil cu `DetectedRegion` din pipeline:
- `region_type`: `"facade"`, `"window"`, `"door"`, `"socle"`
- `bbox`: `(x, y, w, h)` în pixeli
- `color_detected`: `"ml-yolov8"` pentru traceability

---

## Workflow-uri

### Colectare date (LabelImg)
```bash
# Import adnotări din folder
python tools/eval/import_labelimg_annotations.py --source "C:\cale\folder"
```

### Export detecții corecte ca GT (imagini cu markup)
```bash
python tools/eval/export_detection_as_gt.py --image "detector cantitati/raw/lot/NUME.png"
```

### Re-antrenare
```bash
# CPU (~8h pe 69 imagini)
python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data='data/yolo_facade/dataset.yaml', epochs=150, imgsz=1024, batch=4, patience=30, device='cpu')
"

# GPU Colab (~30 min)
# Uploadează data/yolo_facade/ pe Colab, apoi:
# model.train(..., device=0)
```

### Test model pe imagine
```bash
python -c "
import os; os.environ['CUDA_VISIBLE_DEVICES']=''
from ultralytics import YOLO
model = YOLO('models/yolov8n_facade_v2_best.pt')
results = model.predict('CALE_IMAGINE.png', conf=0.25, imgsz=1024, device='cpu')
for r in results:
    for box in r.boxes:
        cls = int(box.cls[0])
        x1,y1,x2,y2 = box.xyxy[0].tolist()
        print(f'{[\"facade\",\"window\",\"door\",\"socle\"][cls]}: ({x1:.0f},{y1:.0f},{x2-x1:.0f},{y2-y1:.0f}) conf={float(box.conf[0]):.3f}')
"
```

---

## Fișiere relevante

| Fișier | Scop |
|--------|------|
| `models/yolov8n_facade_v2_best.pt` | Model antrenat (6.3MB) |
| `data/yolo_facade/dataset.yaml` | Config dataset YOLO |
| `data/yolo_facade/classes.txt` | Clase: facade, window, door, socle |
| `tools/eval/export_detection_as_gt.py` | Export detecție ca GT |
| `tools/eval/convert_gt_to_yolo.py` | Conversie GT → YOLO |
| `tools/eval/import_labelimg_annotations.py` | Import adnotări LabelImg |
| `tools/eval/populate_gt_from_eval.py` | Populare GT din eval report |
| `core/foundation_segmentation.py` | Schelet integrare ML (stub) |
| `core/foundation_adapter.py` | Adaptor ML → DetectedRegion (stub) |
| `docs/DATA_COLLECTION_GUIDE.md` | Ghid colectare date |
