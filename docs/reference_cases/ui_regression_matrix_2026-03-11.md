# UI Regression Matrix - 2026-03-11

Scop: snapshot vizual pentru 11 teste rulate manual in DrawQuantPDF.

## Concluzie

Motorul actual este mult mai bun pe `flat_long_facade` si `composite_stepped_facade` decat pe `gable_facade` sau `sparse_openings_flat`.

Clase observate:
- `flat_long_facade`: ROMCEA-like, STREMTAN-like. Cea mai buna clasa actuala.
- `composite_stepped_facade`: ARION-like. Acceptabil, dar inca supradimensioneaza unele box-uri.
- `sparse_openings_flat`: BACI / POPA / unele imagini cu putine goluri. Fragil la soclu si la false positives/false negatives.
- `gable_facade`: MARGINEAN / MUNTEAN / STETCU-like. Cea mai slaba clasa actuala; logica de banda plana taie gresit corpul.

## Teste

1. `Screenshot 2026-03-11 165053.png`
- Imagine incarcata: `POPA MARINELA_colorized_fatada_spate.png`
- Clasa: `sparse_openings_flat`
- Observatie: fațada este rezonabila, dar deschiderile sunt confuze; apar clasificari imprecise pentru golurile centrale/laterale.
- Verdict: `medium`

2. `Screenshot 2026-03-11 165834.png`
- Imagine incarcata: `BACI CATALINA_colorized_fatada_spate.png`
- Clasa: `sparse_openings_flat`
- Observatie: corpul mare este prins, dar soclul este doar partial; cazul este aproape fara ancore din goluri.
- Verdict: `medium`

3. `Screenshot 2026-03-11 165711.png`
- Imagine incarcata: `MARGINEAN ANA_colorized_fatada_dreapta.png`
- Clasa: `gable_facade`
- Observatie: conturul fatadei este gresit pentru fronton; soclul si usa/fereastra sunt suprapuse pe o geometrie slaba.
- Verdict: `poor`

4. `Screenshot 2026-03-11 165641.png`
- Imagine incarcata: `MARGINEAN ANA_colorized_fatada_stanga.png`
- Clasa: `gable_facade`
- Observatie: golurile sunt partial rezonabile, dar corpul superior al fatadei este subestimat.
- Verdict: `medium`

5. `Screenshot 2026-03-11 165612.png`
- Imagine incarcata: `MARGINEAN ANA_colorized_fatada_fata.png`
- Clasa: `sparse_openings_flat`
- Observatie: corpul lung este prins, dar foarte putine goluri/soclu valid; nevoie de ramura dedicata pentru fatade aproape goale.
- Verdict: `medium`

6. `Screenshot 2026-03-11 165547.png`
- Imagine incarcata: `MUNTEAN LUCRETIA_colorized_fatada_stanga.png`
- Clasa: `gable_facade`
- Observatie: usi/ferestre sunt tolerabile, dar fatada e tratata ca banda plana si pierde frontonul.
- Verdict: `poor`

7. `Screenshot 2026-03-11 165354.png`
- Imagine incarcata: `STREMTAN MARIA_colorized_fatada_fata.png`
- Clasa: `flat_long_facade`
- Observatie: unul dintre cele mai bune rezultate globale; corp, goluri si soclu sunt relativ coerente.
- Verdict: `good`

8. `Screenshot 2026-03-11 165324.png`
- Imagine incarcata: `ARION SIMION_colorized_fatada_fata.png`
- Clasa: `composite_stepped_facade`
- Observatie: volum compus prins rezonabil; box-urile pe unele ferestre inca sunt prea mari.
- Verdict: `good-usable`

9. `Screenshot 2026-03-11 165240.png`
- Imagine incarcata: `POPA MARINELA 2_COLORIZED_fatada_fata.png`
- Clasa: `sparse_openings_flat`
- Observatie: motorul prinde doar un fragment de corp; cazul are semnal geometric foarte slab.
- Verdict: `poor`

10. `Screenshot 2026-03-11 165202.png`
- Imagine incarcata: `POPA MARINELA_colorized_fatada_fata.png`
- Clasa: `sparse_openings_flat`
- Observatie: corpul mare este prins, dar deschiderile sunt subdetectate; soclul e inca euristic.
- Verdict: `medium`

11. `Screenshot 2026-03-11 165128.png`
- Imagine incarcata: `STETCUCALINA_colorized_fatada_dreapta.png`
- Clasa: `gable_facade`
- Observatie: deschiderile sunt partial utile, dar conturul fatadei taie gresit volumetria superioara.
- Verdict: `poor-medium`

## Evaluare de ansamblu

Stare estimata pe clase:
- `flat_long_facade`: bun
- `composite_stepped_facade`: utilizabil
- `sparse_openings_flat`: mediu
- `gable_facade`: slab

Estimare globala pe cele 11 teste:
- bune sau utilizabile: 3-4
- medii: 3-4
- slabe: 3-4

## Directie corecta

Urmatorul pas tehnic recomandat nu este un nou patch local pe un caz singular.
Este introducerea unui clasificator determinist de scena si rutare pe ramuri:

1. `flat_long_facade`
2. `composite_stepped_facade`
3. `sparse_openings_flat`
4. `gable_facade`

Abia dupa aceasta separare are sens sa rafinam geometria pe fiecare clasa fara sa stricam cazurile bune.
