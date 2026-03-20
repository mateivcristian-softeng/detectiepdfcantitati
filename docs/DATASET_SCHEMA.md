# Schema de Anotare – DrawQuantPDF

Schema pentru pachetul de date folosit la antrenarea/îmbunătățirea detectării fatadelor, ferestrelor, ușilor și referințelor de scară.

---

## 1. Format General

Fiecare anotare este stocată într-un fișier JSON, unul per imagine. Fișierul trebuie să poarte același nume cu imaginea, cu extensia `.json` (ex: `faada_01.png` → `faada_01.json`).

### 1.1 Rădăcina documentului

```json
{
  "version": "1.0",
  "image_file": "faada_01.png",
  "image_width": 2480,
  "image_height": 3508,
  "scale_refs": [],
  "facades": [],
  "windows": [],
  "doors": []
}
```

| Câmp         | Tip     | Obligatoriu | Descriere |
|-------------|---------|-------------|-----------|
| `version`   | string  | da          | Versiunea schemei (ex: "1.0") |
| `image_file`| string  | da          | Numele fișierului imagine asociat |
| `image_width`   | int     | da      | Lățimea imaginii în pixeli |
| `image_height`  | int     | da      | Înălțimea imaginii în pixeli |
| `scale_refs`| array   | da          | Referințe de scară (liniare sau de suprafață) |
| `facades`   | array   | da          | Măști/regiuni fatadă |
| `windows`   | array   | da          | Ferestre |
| `doors`     | array   | da          | Uși |

---

## 2. Referințe de scară (`scale_refs`)

Permit calibrarea pixel–metru pentru conversia dimensiunilor din pixeli în unități metrice.

### 2.1 Referință liniară (Ref. m + Ref. px)

```json
{
  "type": "linear",
  "ref_m": 1.0,
  "ref_px": 254,
  "bbox": [100, 3200, 354, 3240]
}
```

| Câmp     | Tip   | Descriere |
|----------|-------|-----------|
| `type`   | "linear" | Tip referință |
| `ref_m`  | float | Lungimea cunoscută în metri (ex: 1.0 pentru scară grafică) |
| `ref_px` | float | Lungimea în pixeli corespunzătoare |
| `bbox`   | [x,y,w,h] | Bounding box al liniei de scară (opțional) |

**Formulă**: `linear_scale_px_per_m = ref_px / ref_m`

### 2.2 Referință din text OCR (dimensiuni liniare)

```json
{
  "type": "linear_from_ocr",
  "value_m": 6.08,
  "bbox": [120, 150, 280, 185]
}
```

| Câmp       | Tip   | Descriere |
|------------|-------|-----------|
| `type`     | "linear_from_ocr" | Valoare extrasă din text (ex: "6.080 m") |
| `value_m`  | float | Valoarea în metri |
| `bbox`     | [x,y,w,h] | Poziția textului |

---

## 3. Fatade (`facades`)

Regiuni care reprezintă fațada clădirii (mască principală, fără acoperiș/soclu dacă e exclus).

### 3.1 Mască fatadă

```json
{
  "label": "FATADA PRINCIPALA",
  "bbox": [120, 450, 1850, 2450],
  "area_px": 4625000,
  "total_area_m2": 58.86,
  "net_area_m2": 48.20,
  "socle_excluded_area_m2": 2.50,
  "source": "manual"
}
```

| Câmp                   | Tip   | Obligatoriu | Descriere |
|------------------------|-------|-------------|-----------|
| `label`                | string| da          | Nume fatadă (FATADA PRINCIPALA, FATADA POSTERIOARA, etc.) |
| `bbox`                 | [x,y,w,h] | da     | Bounding box în pixeli: (x,y) colț stânga-sus, w=lățime, h=înălțime |
| `area_px`              | float | recomandat  | Suprafață în pixeli |
| `total_area_m2`        | float | recomandat  | Suprafață totală fatadă (m²) |
| `net_area_m2`          | float | opțional    | Fatadă netă după excluderea ferestrelor/ușilor |
| `socle_excluded_area_m2` | float | opțional | Suprafață soclu exclusă din total |
| `source`               | string| opțional    | "manual", "color", "ocr", "photo" |

**Convenții bbox**: Format `(x, y, w, h)` – coordonate în pixeli relativ la imagine.

---

## 4. Ferestre (`windows`)

Elemente de tip fereastră, cu bounding box și opțional dimensiuni metrice.

```json
{
  "label": "F2",
  "bbox": [450, 1200, 320, 180],
  "parent_facade": "FATADA PRINCIPALA",
  "area_m2": 5.76,
  "width_m": 2.40,
  "height_m": 2.40,
  "source": "manual"
}
```

| Câmp            | Tip   | Obligatoriu | Descriere |
|-----------------|-------|-------------|-----------|
| `label`         | string| da          | Etichetă (F1, F2, F2-1, etc.) |
| `bbox`          | [x,y,w,h] | da     | Bounding box în pixeli |
| `parent_facade` | string| recomandat  | Fatada părinte (trebuie să existe în `facades`) |
| `area_m2`       | float | recomandat  | Suprafață în m² (0.15–15 m²) |
| `width_m`       | float | opțional    | Lățime în m (0.2–8 m) |
| `height_m`      | float | opțional    | Înălțime în m (0.2–5 m) |
| `source`        | string| opțional    | "manual", "color", "ocr" |

**Limite plauzibile**: suprafață 0.15–15 m², lățime 0.2–8 m, înălțime 0.2–5 m.

---

## 5. Uși (`doors`)

Elemente de tip ușă.

```json
{
  "label": "U2",
  "bbox": [800, 1100, 180, 400],
  "parent_facade": "FATADA PRINCIPALA",
  "area_m2": 2.88,
  "width_m": 1.20,
  "height_m": 2.40,
  "source": "manual"
}
```

| Câmp            | Tip   | Obligatoriu | Descriere |
|-----------------|-------|-------------|-----------|
| `label`         | string| da          | Etichetă (U1, U2, U2-1, etc.) |
| `bbox`          | [x,y,w,h] | da     | Bounding box în pixeli |
| `parent_facade` | string| recomandat  | Fatada părinte |
| `area_m2`       | float | recomandat  | Suprafață în m² (0.5–6.5 m²) |
| `width_m`       | float | opțional    | Lățime în m (0.5–5 m) |
| `height_m`      | float | opțional    | Înălțime în m (1.4–5 m) |
| `source`        | string| opțional    | "manual", "color", "ocr" |

**Limite plauzibile**: suprafață 0.5–6.5 m², lățime 0.5–5 m, înălțime 1.4–5 m.

---

## 6. Reguli de consistență

1. **Bbox**:
   - Format: `[x, y, w, h]` cu `w>0`, `h>0`.
   - `x`, `y` în interiorul imaginii; `x+w ≤ image_width`, `y+h ≤ image_height`.

2. **Relații spațiale**:
   - Ferestre și uși trebuie să fie în interiorul sau foarte aproape de `bbox`-ul fatadei părinte.

3. **Etichete**:
   - Fatade: "FATADA PRINCIPALA", "FATADA POSTERIOARA", "FATADA LATERALA", etc.
   - Ferestre: F1, F2, F2-1, F 2 (variante acceptate).
   - Uși: U1, U2, U2-1, etc.

4. **Scale refs**:
   - Cel puțin o referință liniară (`linear` sau `linear_from_ocr`) recomandată pentru calibrare.

5. **Unicitate**:
   - `label`-urile ferestrelor/ușilor pot fi duplicate între fatade diferite, dar nu în interiorul aceleiași fatade.

---

## 7. Mapare cu DetectedRegion / ParsedFacade

| Câmp JSON          | DetectedRegion / ParsedFacade |
|--------------------|-------------------------------|
| `bbox`             | `bbox` (x,y,w,h)              |
| `label`            | `label`                       |
| `parent_facade`    | `parent_facade`               |
| `area_m2`          | `area_m2`                     |
| `width_m`, `height_m` | `width_m`, `height_m`      |
| `total_area_m2`    | `total_area` / `area_m2` (fatadă) |
| `region_bbox` (fatadă) | `region_bbox` (ParsedFacade) |
