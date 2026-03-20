# DrawQuantPDF v1.0.0

Aplicatie profesionala pentru detectia si cuantificarea suprafetelor de fatade, ferestre si usi din desene de arhitectura.

## Functionalitati

- **Detectie automata** a fatadelor, ferestrelor si usilor din imagini annotate
- **OCR inteligent** cu upscaling automat si corectie automata a punctelor zecimale
- **Calibrare automata px/m²** din elemente cunoscute pentru completarea ariilor lipsa
- **Calibrare liniara px/m** din cote OCR pentru stabilizarea estimarilor
- **Filtrare robusta anti-fals-pozitive** pentru usi (text FATADA exclus)
- **Detecție markup**: contur rosu pentru fatada + contur galben pentru ferestre
- **Excludere automata soclu** (pentru fatade auto-detectate din foto/segmentare)
- **Motor cantitati liniare**: glaf, picurator (sus ferestre+usi), coltare verticale
- **Quality gate automat** care semnaleaza input insuficient pentru analiza full-auto
- **Conversie PNG → PDF** pastrand scara 1:100
- **Export Excel** cu raport formatat profesional (suprafete, subtotaluri, totaluri)
- **Calcul automat**:
  - Suprafata totala fatade
  - Suprafata tamplarie (ferestre + usi)
  - Suprafata termosistem = Fatade - Tamplarie
  - Lungime glaf = suma latimilor ferestrelor
  - Lungime picurator = suma latimilor ferestrelor + usilor
  - Lungime coltare = aproximare 2 x inaltime fatada
- **GUI moderna** cu CustomTkinter (dark/light mode)
- **Editare manuala** pentru corectii si completari

## Instalare

```bash
pip install -r requirements.txt
```

## Utilizare

```bash
python main.py
```

Sau dublu-click pe `run.bat`

### Pasi:
1. **Incarcati imaginea** fatadei (PNG/JPG)
2. Apasati **ANALIZA COMPLETA** - pipeline-ul ruleaza automat:
   - OCR extrage textul si masuratorile
   - Detectie culoare identifica ferestrele (dreptunghiuri galbene)
   - Calcul suprafete nete
   - Daca exista markup rosu/galben pe imagine, acesta are prioritate la segmentare
3. **Verificati** rezultatele in panoul din dreapta
4. **Editati manual** daca este necesar (butoanele + Fatada / + Fereastra / + Usa)
5. **Exportati** in Excel sau convertiti in PDF
   - Pentru calibrare PDF pe referinta: completati **Ref. (m)** si **Ref. (px)**
     impreuna; altfel lasati ambele goale pentru modul standard 1:100

### Mod Schita In Aplicatie
Pentru imagini foto unde detectia full-auto nu este suficienta:
1. Apasati **Deseneaza Fatada** si marcati conturul cu click pe puncte.
2. Apasati **Finalizeaza Contur**.
3. Apasati **Deseneaza Fereastra** si trasati dreptunghiurile ferestrelor.
4. Completati **Ref. (m)** si **Ref. (px)**.
5. Apasati **ANALIZA DIN SCHITA** pentru calcul direct.

## Structura Proiect

```
├── main.py              # Punct de intrare
├── config.py            # Configuratie (culori HSV, praguri)
├── run.bat              # Lansator Windows
├── requirements.txt     # Dependente Python
├── core/
│   ├── pipeline.py      # Pipeline analiza (OCR + culoare + calcul)
│   ├── color_detector.py # Detectie regiuni colorate
│   ├── ocr_engine.py    # Motor OCR cu EasyOCR
│   ├── area_calculator.py # Calcul suprafete
│   ├── pdf_converter.py # Conversie imagine → PDF
│   └── excel_exporter.py # Export Excel formatat
├── gui/
│   └── app.py           # Interfata grafica
└── assets/              # Imagini de test
```

## Note Tehnice

- Imaginile trebuie sa fie la **scara 1:100**
- Annotarile colorate asteptate:
  - **Cyan/albastru** = contur fatade
  - **Galben** = dreptunghiuri ferestre (F2, etc.)
  - **Rosu** = text "FATADA" cu suprafete
  - **Portocaliu/magenta** = annotari usi
- OCR-ul face upscaling 2x automat pentru text mic
- Formulele de calcul:
  - `Termosistem = Suprafata Fatada - Ferestre - Usi`
  - `Glaf = Σ latimi ferestre`
  - `Picurator = Σ latimi ferestre + Σ latimi usi`
  - `Coltare = 2 x inaltime fatada` (estimare automata)

## Workflow sincronizat (5 AI)

Procesul de colaborare multi-agent este implementat in proiect.

Documentatie:
- `docs/WORKFLOW_SYNC.md`

Scripturi:
- `scripts/setup-worktrees.ps1` - creeaza worktree-uri si branch-uri pentru cei 5 AI
- `scripts/worktree-status.ps1` - status centralizat (dirty/ahead/behind/last commit)
- `scripts/sync-worktrees.ps1` - pull/rebase sincronizat pe toate worktree-urile
- `scripts/check-ownership.ps1` - verifica daca branch-ul modifica doar zonele permise
- `scripts/integration-merge.ps1` - merge train pe `integration` + quality command
- `.github/workflows/sync-gate.yml` - quality gate automat pe `integration` si `main`

Comenzi rapide:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup-worktrees.ps1
powershell -ExecutionPolicy Bypass -File scripts/worktree-status.ps1
powershell -ExecutionPolicy Bypass -File scripts/sync-worktrees.ps1
powershell -ExecutionPolicy Bypass -File scripts/check-ownership.ps1
powershell -ExecutionPolicy Bypass -File scripts/integration-merge.ps1
```
