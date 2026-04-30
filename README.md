# Greenness Index – Hoflabor

Berechnung des **Greenness Index (GI = G / R+G+B)** aus RGB-Drohnenfotos mit interaktivem Polygon-Editor.

**Autor:** Eve Bücheler · [Hoflabor](https://hoflabor.ch)

---

## Workflow

1. RGB GeoTIFF auswählen (Datum aus Dateiname: `_JJMMTT`)
2. Feldname aus Dropdown wählen (lädt passendes Polygon-GeoJSON)
3. Vorschau: Bild mit Polygon-Overlay
4. Optional: Polygone im Editor anpassen (verschieben, rotieren, skalieren)
5. Greenness Index berechnen
6. Ergebnisse speichern (GeoJSON + CSV)

## Installation

```bash
pip install -r requirements.txt
python green_ml.py
```

## Ordnerstruktur

```
Greenness Index/
├── green_ml.py           # Hauptprogramm (aktuell)
├── requirements.txt      # Python-Abhängigkeiten
├── polygone/             # Polygon-Definitionen (GeoJSON) pro Feld
└── Resultate GI/         # Berechnete Ergebnisse
```

## Drohnenbilder

Grosse GeoTIFF-Dateien (`*.tif`) sind **nicht im Repository** gespeichert
(zu grosse Dateigrösse). Die Bilder liegen lokal auf dem Netzlaufwerk.

## GitHub Sync

- **`SETUP_GITHUB.ps1`** – einmalige Einrichtung (Rechtsklick → Mit PowerShell ausführen)
- **`SYNC_GITHUB.ps1`** – manueller oder automatischer Sync (Windows Task Scheduler)
- Automatischer Sync läuft alle **2 Stunden** wenn Änderungen vorhanden sind
