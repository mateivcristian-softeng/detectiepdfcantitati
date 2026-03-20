# Fixtures for evaluation

- `images/` – Imagini de test (PNG)
- `gt/` – Ground truth JSON (facades, windows cu bbox și area_m2)

Format GT:
```json
{
  "facades": [{"name": "...", "bbox": [x,y,w,h], "area_m2": 10.5}],
  "windows": [{"label": "F2", "bbox": [x,y,w,h], "area_m2": 1.2}]
}
```
