# Ghid practic: Colectare și etichetare date DrawQuantPDF

Ghid pentru colectarea imaginilor și etichetarea anotațiilor folosite la îmbunătățirea detectării.

---

## 1. Surse de imagini

### 1.1 PDF-uri convertite
- Extrage pagini din PDF-uri de desene arhitecturale (fațade, planuri)
- Conversie la PNG/JPEG (ex: 254 DPI pentru 1:100 pe A3)
- Păstrează rezoluția suficientă pentru OCR și detectie culori

### 1.2 Fotografii fațade
- Imagini foto reale ale clădirilor
- Evită unghiuri extreme, distorsiuni puternice
- Preferă iluminare uniformă

### 1.3 Desene colorate (markup)
- Desene cu fatade cyan/albastru, ferestre galben/verde, usi portocaliu/magenta
- Formatul de culori este descris în `config.COLOR_RANGES`

---

## 2. Structura de etichetare

### 2.1 Organizare fișiere
```
data/
├── train/
│   ├── faada_01.png
│   ├── faada_01.json
│   ├── faada_02.png
│   └── faada_02.json
├── val/
│   ├── faada_val_01.png
│   └── faada_val_01.json
└── test/
    ├── faada_test_01.png
    └── faada_test_01.json
```

**Regula**: Fișierul JSON are același nume de bază ca imaginea (fără extensie).

### 2.2 Raport train/val/test
- **Train**: 70–80% din date
- **Val**: 10–15%
- **Test**: 10–15%

Evită overlap între seturi (același desen nu apare în două seturi).

---

## 3. Pași de etichetare

### 3.1 Pregătire
1. Deschide imaginea într-un editor (LabelImg, CVAT, Label Studio sau script custom)
2. Verifică `image_width` și `image_height` (dimensiuni efective în pixeli)

### 3.2 Ordinea recomandată
1. **Scale refs** – marchează întâi referința de scară (linia 1 m = X px sau textul cu dimensiuni)
2. **Fatade** – regiunile principale ale fațadei (exclude acoperișul, include soclul dacă e relevant)
3. **Ferestre** – fiecare fereastră ca bounding box dreptunghiular
4. **Uși** – fiecare ușă ca bounding box

### 3.3 Format bounding box
- **Format**: `[x, y, w, h]`
  - `(x, y)` = colțul stânga-sus
  - `w` = lățimea în pixeli
  - `h` = înălțimea în pixeli
- Originea (0,0) este stânga-sus a imaginii.

### 3.4 Etichete
| Tip        | Exemple acceptate                    |
|------------|--------------------------------------|
| Fatade     | FATADA PRINCIPALA, FATADA POSTERIOARA, FATADA LATERALA |
| Ferestre   | F1, F2, F2-1, F 2                    |
| Uși        | U1, U2, U2-1                         |

---

## 4. Reguli QA (Quality Assurance)

### 4.1 Obligatorii
- [ ] Toate bbox-urile sunt în interiorul imaginii
- [ ] `bbox` în format `[x, y, w, h]` cu `w > 0`, `h > 0`
- [ ] Fiecare fereastră/ușă are `parent_facade` care există în lista de fatade
- [ ] Ferestrele și ușile sunt incluse în bbox-ul fatadei părinte (cu toleranță ±20 px)

### 4.2 Plauzibilitate dimensiuni (m²)
| Element | Suprafață (m²) | Lățime (m) | Înălțime (m) |
|---------|----------------|------------|--------------|
| Fereastră | 0.15 – 15    | 0.2 – 8    | 0.2 – 5      |
| Ușă      | 0.5 – 6.5     | 0.5 – 5    | 1.4 – 5      |
| Fatadă   | 1 – 5000      | -          | -            |

### 4.3 Scale reference
- Cel puțin o referință liniară (`linear` sau `linear_from_ocr`) când există anotații
- `ref_m` și `ref_px` > 0 pentru tipul `linear`
- Pentru desene la scară 1:100 pe A3: 1 m ≈ 254 px (254 DPI)

### 4.4 Consistență
- Nu lipsesc ferestre/uși vizibile
- Nu se dublează elemente (același dreptunghi etichetat de două ori)
- `parent_facade` corespunde cu fatada în care se află geometric elementul

---

## 5. Validare automată

După etichetare, rulează scriptul de validare:

```powershell
# Din rădăcina proiectului
python -m tools.data_prep.validate_annotations
```

Opțiuni:
- `--dir PATH` – validează doar un director
- `--strict` – cere scale ref când există anotații
- `--verbose` – afișează și fișierele care trec

Exemplu:
```powershell
python -m tools.data_prep.validate_annotations --dir data/train --strict
```

---

## 6. Workflow recomandat

1. **Colectare** – adaugă imagini în `data/train`, `data/val`, `data/test`
2. **Etichetare** – creează fișiere JSON conform `docs/DATASET_SCHEMA.md`
3. **Validare** – rulează `validate_annotations.py`
4. **Corectare** – remediază erorile raportate
5. **Repetare** – până când toate fișierele trec validarea

---

## 7. Colectare rapidă pentru ML (Workflow practic)

### 7.1 Metoda cea mai rapidă: imagini cu markup procesate de aplicație

Dacă ai desene cu fațade colorizate (cyan=fațadă, galben=ferestre, portocaliu=uși):

1. Plasează PNG-ul în `detector cantitati/raw/<lot>/NUME.png`
2. Plasează worldfile-ul `.pgw` lângă el (dacă există)
3. Deschide imaginea în DrawQuantPDF și rulează detecția
4. Verifică vizual că rezultatul e corect (ferestre, uși, fațadă, soclu la pozițiile corecte)
5. Exportă GT-ul automat:
```bash
python tools/eval/export_detection_as_gt.py --image "detector cantitati/raw/<lot>/NUME.png"
```

### 7.2 Imagini foto/fără markup — adnotare manuală

Pentru imagini fără culori convenționale (nori de puncte colorizate, fotografii):

1. Plasează PNG-ul în `detector cantitati/raw/<lot>/NUME.png`
2. Creează manual fișierul JSON în `detector cantitati/gt/<lot>/NUME.json` cu formatul:
```json
{
  "sample_id": "lot_nume",
  "image_size": {"width": 2414, "height": 908},
  "source": "manual_annotation",
  "facades": [{"bbox": [x, y, w, h], "type": "facade"}],
  "windows": [{"bbox": [x, y, w, h], "type": "window"}],
  "doors": [{"bbox": [x, y, w, h], "type": "door"}],
  "socles": [{"bbox": [x, y, w, h], "type": "socle"}]
}
```
3. Folosește un tool de adnotare (LabelImg, CVAT) sau GUI-ul aplicației

### 7.3 Regenerare dataset YOLO după colectare

```bash
# Regenerează din toate GT-urile populare
python tools/eval/convert_gt_to_yolo.py

# Apoi re-antrenează
python tools/eval/train_yolo_facade.py
```

### 7.4 Câte imagini sunt suficiente?

| Nivel | Total | Din care foto | Efect |
|-------|-------|---------------|-------|
| Minim | 50 | 30+ | Funcționează pe cazuri simple |
| Bun | 100-150 | 60+ | Generalizare decentă |
| Robust | 300+ | 150+ | Fațade complexe/atipice |

Cele 21 imagini actuale sunt **insuficiente** — antrenamentul YOLOv8 pe ele produce 0 detecții.

---

## 8. Referințe

- Schema: `docs/DATASET_SCHEMA.md`
- Pipeline și contracte: `core/pipeline.py`, `core/color_detector.py`, `core/area_calculator.py`
- Config culori: `config.py`
- Conversie YOLO: `tools/eval/convert_gt_to_yolo.py`
- Populare GT: `tools/eval/populate_gt_from_eval.py`
