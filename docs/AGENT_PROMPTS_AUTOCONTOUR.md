# Prompturi operative pentru cei 5 agenti

Foloseste promptul corespunzator branch-ului tau. Nu modifica fisiere in afara zonei de ownership.

## Agent 1 - Quick Wins CV
```
Lucrezi pe branch `feat/ai1-quickwins-cv`.
Scop: redu false-positive windows pe roof si oversized facade in photo mode.
Fisiere permise: `core/color_detector.py`, `config.py`.
Nu modifica `core/pipeline.py`, `tests/`, `data/`.

Implementare:
1) adauga roof-cut robust in `detect_photo_facade_region`;
2) adauga roof-zone exclusion in `detect_photo_windows`;
3) adauga un validator simplu de textura/gradient pentru candidati windows;
4) muta pragurile in `config.py`.

Reguli:
- pastreaza fallback existent;
- nu elimina logica veche, doar adauga filtre controlate de praguri;
- fara dependinte noi.

Verificare:
- `python -m compileall .`
- `python test_pipeline.py`
- rezuma ce s-a imbunatatit si ce cazuri raman riscante.
```

## Agent 2 - Medium
```
Lucrezi pe branch `feat/ai2-medium-classic-ml`.
Scop: adauga rafinare clasica si validator mediu, fara a strica fluxul existent.
Fisiere permise: `core/pipeline.py`, `core/photo_refiner.py`, `core/window_validator.py`, `requirements.txt`.
Nu modifica `core/color_detector.py`.

Implementare:
1) creeaza `photo_refiner.py` pentru rafinare ROI (ex: GrabCut optional);
2) creeaza `window_validator.py` pentru scor feature-based;
3) integreaza in `pipeline.py` prin feature flag;
4) daca validatorul nu are semnal, fallback pe logica curenta.

Reguli:
- impact minim in pipeline;
- orice cale noua trebuie sa fie opt-in/configurabila.

Verificare:
- compileall + smoke pe 3 imagini foto;
- documenteaza latenta adaugata.
```

## Agent 3 - Major
```
Lucrezi pe branch `feat/ai3-major-foundation`.
Scop: pregateste integrarea de model de segmentare fara sa afecteze runtime default.
Fisiere permise: `core/foundation_adapter.py`, `core/foundation_segmentation.py`, `docs/FOUNDATION_MODEL_PLAN.md`, `requirements-optional.txt`.
Nu modifica `pipeline.py` pentru activare default.

Implementare:
1) defineste interfata adapter -> output compatibil `DetectedRegion`;
2) adauga schelet de inferenta (opt-in);
3) documenteaza cerinte hardware/cost/latenta;
4) propune strategie fallback pe OpenCV daca modelul lipseste.

Reguli:
- strict opt-in;
- codul default trebuie sa functioneze identic.
```

## Agent 4 - Date necesare
```
Lucrezi pe branch `feat/ai4-data-annotations`.
Scop: livreaza schema de date si ghid de etichetare pentru antrenare/evaluare.
Fisiere permise: `data/`, `tools/data_prep/`, `docs/DATASET_SCHEMA.md`, `docs/DATA_COLLECTION_GUIDE.md`.
Nu modifica `core/`, `gui/`, `tests/`.

Implementare:
1) schema anotari: facade mask, windows, doors, scale refs;
2) layout train/val/test;
3) script validare consistenta;
4) ghid practic de etichetare si QA.

Verificare:
- ruleaza validator pe minim 10 mostre.
```

## Agent 5 - Metrici tinta
```
Lucrezi pe branch `feat/ai5-metrics-eval`.
Scop: sistem de evaluare obiectiv + quality gate.
Fisiere permise: `tests/`, `tools/eval/`, `test_pipeline.py`, `.github/workflows/`, `docs/METRICS_TARGETS.md`.
Nu modifica `core/color_detector.py` sau `core/pipeline.py`.

Implementare:
1) metrici: IoU fatada, precision/recall ferestre, eroare arii;
2) evaluator cu output JSON/CSV;
3) test de regresie explicit pentru "window pe roof";
4) gate CI cu praguri minime.

Verificare:
- evaluator local pe fixture dataset;
- raport baseline stocat ca artefact.
```

