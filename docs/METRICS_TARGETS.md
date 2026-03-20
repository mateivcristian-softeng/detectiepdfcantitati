# DrawQuantPDF – Metrici țintă (AI5)

Praguri obiective pentru evaluarea pipeline-ului de detecție.

## Metrici

| Metrică | Descriere | Unitate |
|---------|-----------|---------|
| `facade_iou_mean` | IoU mediu fatade (pred vs GT) | 0–1 |
| `window_precision` | P / (P + FP) ferestre | 0–1 |
| `window_recall` | P / (P + FN) ferestre | 0–1 |
| `window_f1` | F1 ferestre | 0–1 |
| `area_mae_m2` | Eroare absolută medie arii | m² |
| `area_max_error_m2` | Eroare maximă arie | m² |

## Praguri țintă (actuale)

| Metrică | Țintă minimă | Notă |
|---------|--------------|------|
| `facade_iou_mean` | ≥ 0.30 | Detectie fatade imperfectă pe desene complexe |
| `window_precision` | ≥ 0.60 | Acceptabil FP pe ferestre mici/decorative |
| `window_recall` | ≥ 0.70 | Prioritizare recall pentru cantitati |
| `window_f1` | ≥ 0.55 | Echilibru P/R |
| `area_mae_m2` | ≤ 0.50 | Toleranta ±0.5 m² pe element |
| `area_max_error_m2` | ≤ 2.00 | Outlier max 2 m² |

## Scenarii de regresie

- **Window pe roof**: Fereastra în zona acoperiș (top 12% fatadă) trebuie exclusă sau nu contează ca TP; nu trebuie să apară ca FP dacă GT nu o include.
- **Soclu**: Excluderea soclului nu penalizează metricile de arie fatadă.

## Limitări actuale

- Fixture-urile sunt sintetice; comportamentul pe imagini reale poate diferi.
- IoU fatadă depinde de calitatea `region_bbox` / `region_bbox` estimat.
- Match ferestre: `match_distance_px=80`; ferestre foarte apropiate pot fi confuse.

## Verificare

```bash
python -m tools.eval.run_eval tests/fixtures/images/fixture_facade.png tests/fixtures/gt/fixture_facade.json -o json
```
