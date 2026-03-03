# Workflow sincronizat DrawQuantPDF

## Obiectiv
Acest proces permite 5 agenti AI sa lucreze simultan pe acelasi proiect fara conflicte majore, cu integrare continua pe un branch comun.

## Topologie Git
- `main`: branch stabil, gata pentru release.
- `integration`: branch de integrare zilnica (merge train).
- `feat/ai1-ingest-scale`
- `feat/ai2-detectie-cv`
- `feat/ai3-calcule-excel`
- `feat/ai4-gui-annotare`
- `feat/ai5-testare-ci`

## Zone de ownership (default)
- `AI1` -> `core/pdf_converter.py`, `core/ocr_engine.py`, ingest/scara.
- `AI2` -> `core/color_detector.py`, detectii de fatada/ferestre/usi.
- `AI3` -> `core/area_calculator.py`, `core/excel_exporter.py`.
- `AI4` -> `gui/app.py`, flux UI de validare/correctie.
- `AI5` -> `test_pipeline.py`, `README.md`, CI si quality gates.

Regula: daca un task atinge o zona care nu este a ta, creezi un micro-PR separat sau ceri handoff explicit.

## Cadenta de lucru
1. Start zi:
   - Ruleaza `scripts/worktree-status.ps1`.
   - Ruleaza `scripts/sync-worktrees.ps1`.
2. In timpul implementarii:
   - Commit-uri mici, tematice, cu mesaj clar.
   - Rebase periodic pe `origin/integration`.
3. Fereastra de integrare:
   - Fiecare AI face PR -> `integration`.
   - Dupa review si verificari, se face merge in `integration`.
4. Stabilizare:
   - `scripts/integration-merge.ps1` ruleaza merge train + test command.
   - Daca trece, `integration` poate fi promovat in `main`.

## Comenzi standard
Bootstrapping (one-time):
```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup-worktrees.ps1
```

Status rapid:
```powershell
powershell -ExecutionPolicy Bypass -File scripts/worktree-status.ps1
```

Sincronizare in masa:
```powershell
powershell -ExecutionPolicy Bypass -File scripts/sync-worktrees.ps1
```

Integrare automata:
```powershell
powershell -ExecutionPolicy Bypass -File scripts/integration-merge.ps1
```

## Definition of Done (obligatoriu)
- Branch-ul este sincronizat cu `origin/integration`.
- Testele locale relevante trec.
- PR-ul are context, risc, test plan si impact functional.
- Nu exista schimbari nelegate de task.
- Exportul Excel, calculele si outputul vizual sunt valide pe minim un exemplu real.

## Escaladare conflict
- Conflicte simple de text: rezolvare directa in branch-ul feature.
- Conflicte de logica: pairing intre AI owner + AI integrator.
- Blocaj > 30 min: freeze pe fisierul afectat, issue dedicat, decizie de arhitectura.
