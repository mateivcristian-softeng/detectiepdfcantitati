# Plan Executie Multi-Agent (AutoContour Hardening)

## Scop sprint
Corectarea detectiei pe foto pentru cazurile dificile:
- ferestre false-pozitive pe acoperis;
- fatada detectata prea mare;
- crestere robustete fara regresii pe fluxul actual.

## Alocare agenti (fara suprapunere)

### Agent 1 - Quick Wins (`feat/ai1-quickwins-cv`)
- **In scope:** `core/color_detector.py`, `config.py`
- **Out of scope:** `core/pipeline.py`, `tests/`, `data/`, `.github/workflows/`
- **Task-uri:**
  1. roof-cut pentru `detect_photo_facade_region`;
  2. roof-zone exclusion in `detect_photo_windows`;
  3. filtru textura/gradient pentru validare fereastra;
  4. parametri configurabili in `config.py`;
  5. zero schimbari de arhitectura.
- **DoD local:** `python test_pipeline.py` trece, fara erori runtime.

### Agent 2 - Medium (`feat/ai2-medium-classic-ml`)
- **In scope:** `core/pipeline.py`, `core/photo_refiner.py`, `core/window_validator.py`, `requirements.txt`
- **Out of scope:** `core/color_detector.py` (doar consum API), `data/`, `tests/` (except smoke intern local)
- **Task-uri:**
  1. adaugare validator clasic (feature-based) pentru ferestre;
  2. optional GrabCut/refinare ROI in modul foto;
  3. integrare strict prin hook in pipeline cu feature flag;
  4. fallback safe la logica existenta daca validatorul nu are semnal.
- **DoD local:** compile + smoke pe minim 3 imagini foto.

### Agent 3 - Major (`feat/ai3-major-foundation`)
- **In scope:** `core/foundation_adapter.py`, `core/foundation_segmentation.py`, `docs/FOUNDATION_MODEL_PLAN.md`, `requirements-optional.txt`
- **Out of scope:** runtime principal in `pipeline.py` (fara activare default), `gui/`
- **Task-uri:**
  1. design adaptor model segmentare (SAM/SegFormer/YOLO-seg);
  2. interfata unica de inferenta cu output compatibil `DetectedRegion`;
  3. implementare **opt-in only** (nu afecteaza default path);
  4. documentatie cost/latenta/hardware.
- **DoD local:** cod importabil, fara impact pe fluxul default.

### Agent 4 - Date necesare (`feat/ai4-data-annotations`)
- **In scope:** `data/`, `tools/data_prep/`, `docs/DATASET_SCHEMA.md`, `docs/DATA_COLLECTION_GUIDE.md`
- **Out of scope:** `core/`, `gui/`, `tests/`
- **Task-uri:**
  1. schema de anotare (facade mask, windows, doors, scale refs);
  2. structura dataset train/val/test;
  3. ghid de etichetare cu exemple si reguli anti-zgomot;
  4. script de validare consistenta dataset.
- **DoD local:** validare schema pe minim 10 mostre.

### Agent 5 - Metrici tinta (`feat/ai5-metrics-eval`)
- **In scope:** `tests/`, `tools/eval/`, `test_pipeline.py`, `.github/workflows/`, `docs/METRICS_TARGETS.md`
- **Out of scope:** `core/color_detector.py`, `core/pipeline.py`
- **Task-uri:**
  1. definire metrici: IoU fatada, precision/recall ferestre, eroare arii;
  2. baseline evaluator + raport JSON/CSV;
  3. test de regresie pentru cazul "roof false window";
  4. quality gate CI pe praguri minime.
- **DoD local:** evaluator ruleaza local si raporteaza metrici pe set fixture.

## Checklist prioritar si ordinea de integrare

1. **AI4 + AI5 in paralel (fara dependinte de cod core):**
   - AI4 livreaza schema/date;
   - AI5 livreaza evaluator + praguri.
2. **AI1 (Quick Wins) pe runtime existent:**
   - reduce imediat false-pozitive si oversized facade.
3. **AI2 (Medium) dupa merge AI1 in `integration`:**
   - refineaza doar unde quick wins nu ajung la prag.
4. **AI3 (Major) in paralel ca R&D controlat:**
   - fara impact default, doar adaptor opt-in.
5. **Merge train final:**
   - AI4 -> AI5 -> AI1 -> AI2 -> AI3.

## Protocol anti-halucinatie / anti-regresie

- Nu se inventeaza API-uri: orice functie noua se leaga de structurile deja existente.
- Fiecare agent livreaza schimbari mici, verificabile, cu diff orientat pe scop.
- Fiecare PR include:
  - ce problema rezolva;
  - ce nu acopera;
  - dovezi (comenzi, metrici, sample output).
- Orice warning nou in pipeline trebuie explicat si testat.
- Daca un agent atinge fisier in afara ownership, PR se opreste pana la handoff explicit.

## Comenzi obligatorii pe fiecare branch

```powershell
powershell -ExecutionPolicy Bypass -File scripts/worktree-status.ps1
powershell -ExecutionPolicy Bypass -File scripts/sync-worktrees.ps1
powershell -ExecutionPolicy Bypass -File scripts/check-ownership.ps1
python -m compileall .
python test_pipeline.py
```

