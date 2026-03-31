#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Greenness Index Berechnung  –  ML-Erweiterung
==============================================
Projekt : Greenness Index – Hoflabor
Autor   : Eve
Datum   : 2026-03-26
Version : green_ml.py  (Lernalgorithmus fuer Polygon-Vorhersage)

Workflow:
  1. RGB GeoTIFF auswaehlen (Datum wird aus Dateiname extrahiert: _JJMMTT)
  2. Feldname aus Dropdown waehlen (laedt passendes Polygon-GeoJSON)
  3. Vorschau: Bild mit Polygon-Overlay
  4. Optional: Polygone im Editor anpassen
  5. Indizes berechnen (GI = G/(R+G+B)  |  MGRVI = (G²-R²)/(G²+R²))
  6. Ergebnisse speichern (GeoJSON + CSV)

Voraussetzungen:
  pip install -r requirements.txt
"""

import os
import re
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.enums import Resampling as rio_Resampling
from shapely.geometry import Polygon, Point, box as shp_box, mapping
from shapely.affinity import (rotate    as shp_rotate,
                               scale     as shp_scale,
                               translate as shp_translate)
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog


# ==============================================================================
# Konfiguration
# ==============================================================================

BASE_DIR           = os.path.dirname(os.path.abspath(__file__))
POLYGONE_DIR       = os.path.join(BASE_DIR, "polygone")
RESULTATE_DIR      = os.path.join(BASE_DIR, "Resultate GI")
SAVED_LAYOUTS_DIR  = os.path.join(POLYGONE_DIR, "saved_layouts")

DEFAULT_IMG_DIR = (
    r"G:\.shortcut-targets-by-id"
    r"\1OsI23NlCQNPWlzDcBsbJ-CjfaSTpsItZ"
    r"\TEAM\30 - Aktuelle Projekte"
    r"\11 - Drohne\Python Drohne"
)

# Datum im Dateinamen: _JJMMTT  (z.B. _260326 = 26.03.2026)
DATE_RE   = re.compile(r'_(\d{6})(?=[._]|$)')
NAME_PROP = 'name'   # GeoJSON-Property fuer Streifennamen

# Throttle fuer Drag-Redraws: max. 30 fps (~33 ms)
_DRAG_MIN_INTERVAL = 0.033


# ==============================================================================
# Design-System
# ==============================================================================

# Farben – Slate/Green Palette
CLR_BG       = "#f8fafc"
CLR_CARD     = "#ffffff"
CLR_BORDER   = "#e2e8f0"
CLR_BORDER_D = "#cbd5e1"
CLR_TEXT_H   = "#0f172a"
CLR_TEXT_B   = "#334155"
CLR_TEXT_S   = "#64748b"
CLR_TEXT_D   = "#94a3b8"
CLR_GREEN    = "#15803d"
CLR_GREEN_H  = "#166534"
CLR_GREEN_L  = "#f0fdf4"
CLR_GREEN_T  = "#dcfce7"
CLR_BLUE     = "#1d4ed8"
CLR_BLUE_H   = "#1e40af"
CLR_HINT_BG  = "#eff6ff"
CLR_HINT_BD  = "#bfdbfe"
CLR_RED      = "#dc2626"
CLR_RED_H    = "#b91c1c"
CLR_AMBER    = "#d97706"

# Fonts
FONT_TITLE   = ("Helvetica", 22, "bold")
FONT_SUB     = ("Helvetica", 10)
FONT_STEP    = ("Helvetica", 11, "bold")
FONT_LABEL   = ("Helvetica", 10)
FONT_SMALL   = ("Helvetica", 9)
FONT_BTN_PRI = ("Helvetica", 12, "bold")
FONT_BTN     = ("Helvetica", 10)


# ==============================================================================
# Layout-Hilfsfunktionen
# ==============================================================================

def screen_geometry(win, w_frac: float, h_frac: float,
                    min_w: int = 600, min_h: int = 480) -> str:
    """
    Berechnet eine zentrierte Fenstergeometrie relativ zur Bildschirmgroesse.
    Gibt einen Tkinter-Geometry-String zurueck: 'WxH+X+Y'.
    """
    win.update_idletasks()          # sicherstellen dass Screen-Infos verfuegbar
    sw = win.winfo_screenwidth()
    sh = win.winfo_screenheight()
    w  = max(int(sw * w_frac), min_w)
    h  = max(int(sh * h_frac), min_h)
    x  = (sw - w) // 2
    y  = max((sh - h) // 2, 30)    # nie hinter Taskleiste
    return f"{w}x{h}+{x}+{y}"


def canvas_figsize(win, w_frac: float = 0.85, h_frac: float = 0.60,
                   dpi: int = 100):
    """Berechnet eine zur Fenstergrösse passende matplotlib figsize."""
    sw = win.winfo_screenwidth()
    sh = win.winfo_screenheight()
    fw = max(sw * w_frac / dpi, 7.0)
    fh = max(sh * h_frac / dpi, 5.0)
    return fw, fh


# ==============================================================================
# UI-Hilfsfunktionen
# ==============================================================================

def hover_btn(btn: tk.Button, normal: str, hover: str):
    btn.bind('<Enter>', lambda _: btn.config(bg=hover))
    btn.bind('<Leave>', lambda _: btn.config(bg=normal))


def make_card(parent, **kwargs) -> tk.Frame:
    defaults = dict(bg=CLR_CARD,
                    highlightbackground=CLR_BORDER,
                    highlightthickness=1,
                    bd=0, relief='flat',
                    padx=18, pady=14)
    defaults.update(kwargs)
    return tk.Frame(parent, **defaults)


def make_primary_btn(parent, text, command, width=None, **kw):
    opts = dict(bg=CLR_GREEN, fg="white",
                activebackground=CLR_GREEN_H, activeforeground="white",
                font=FONT_BTN_PRI, relief='flat', bd=0,
                cursor="hand2", padx=18, pady=8)
    if width:
        opts['width'] = width
    opts.update(kw)
    btn = tk.Button(parent, text=text, command=command, **opts)
    hover_btn(btn, CLR_GREEN, CLR_GREEN_H)
    return btn


def make_secondary_btn(parent, text, command, **kw):
    opts = dict(bg=CLR_BLUE, fg="white",
                activebackground=CLR_BLUE_H, activeforeground="white",
                font=FONT_BTN, relief='flat', bd=0,
                cursor="hand2", padx=14, pady=7)
    opts.update(kw)
    btn = tk.Button(parent, text=text, command=command, **opts)
    hover_btn(btn, CLR_BLUE, CLR_BLUE_H)
    return btn


def make_danger_btn(parent, text, command, **kw):
    opts = dict(bg=CLR_RED, fg="white",
                activebackground=CLR_RED_H, activeforeground="white",
                font=FONT_BTN, relief='flat', bd=0,
                cursor="hand2", padx=14, pady=7)
    opts.update(kw)
    btn = tk.Button(parent, text=text, command=command, **opts)
    hover_btn(btn, CLR_RED, CLR_RED_H)
    return btn


def make_ghost_btn(parent, text, command, **kw):
    opts = dict(bg=CLR_CARD, fg=CLR_TEXT_B,
                activebackground="#f1f5f9", activeforeground=CLR_TEXT_H,
                font=FONT_BTN, relief='flat', bd=0,
                highlightbackground=CLR_BORDER, highlightthickness=1,
                cursor="hand2", padx=14, pady=6)
    opts.update(kw)
    btn = tk.Button(parent, text=text, command=command, **opts)
    hover_btn(btn, CLR_CARD, "#f1f5f9")
    return btn


# ==============================================================================
# Hilfsfunktionen (Logik)
# ==============================================================================

def extract_date(filepath: str):
    m = DATE_RE.search(os.path.basename(filepath))
    return m.group(1) if m else None


def gi_col(date_str: str) -> str:
    return f"GI_{date_str}"


def mgrvi_col(date_str: str) -> str:
    return f"MGRVI_{date_str}"


def available_fields() -> list:
    if not os.path.isdir(POLYGONE_DIR):
        return []
    fields = []
    for fname in sorted(os.listdir(POLYGONE_DIR)):
        if fname.lower().endswith(('.geojson', '.json')):
            stem = os.path.splitext(fname)[0]
            stem = stem[8:] if stem.lower().startswith('stripes_') else stem
            fields.append(stem)
    return fields


def geojson_for_field(field: str):
    if not os.path.isdir(POLYGONE_DIR):
        return None
    for fname in os.listdir(POLYGONE_DIR):
        if fname.lower().endswith(('.geojson', '.json')):
            stem  = os.path.splitext(fname)[0]
            clean = stem[8:] if stem.lower().startswith('stripes_') else stem
            if clean == field:
                return os.path.join(POLYGONE_DIR, fname)
    return None


def _layout_dir(field: str) -> str:
    """Gibt den Ordner fuer gespeicherte Layouts eines Feldes zurueck."""
    safe = re.sub(r'[\\/:*?"<>|]', '_', field)
    return os.path.join(SAVED_LAYOUTS_DIR, safe)


def list_saved_layouts(field: str) -> list:
    """Gibt sortierte Liste aller gespeicherten Layout-Namen fuer ein Feld zurueck."""
    d = _layout_dir(field)
    if not os.path.isdir(d):
        return []
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(d)
        if f.lower().endswith('.geojson'))


def save_layout_file(gdf: gpd.GeoDataFrame, field: str, name: str):
    """Speichert gdf als benanntes Layout-GeoJSON fuer das angegebene Feld."""
    d = _layout_dir(field)
    os.makedirs(d, exist_ok=True)
    safe_name = re.sub(r'[\\/:*?"<>|]', '_', name)
    path = os.path.join(d, f"{safe_name}.geojson")
    gdf.to_file(path, driver='GeoJSON')
    return path


def load_layout_file(field: str, name: str,
                     target_crs) -> gpd.GeoDataFrame:
    """Laedt ein gespeichertes Layout und projiziert es in target_crs."""
    safe_name = re.sub(r'[\\/:*?"<>|]', '_', name)
    path = os.path.join(_layout_dir(field), f"{safe_name}.geojson")
    gdf  = gpd.read_file(path)
    if gdf.crs and gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)
    return gdf.reset_index(drop=True)


# ==============================================================================
# Lernalgorithmus – Polygon-Vorhersage aus gespeicherten Layouts
# ==============================================================================
# Ansatz: Procrustes-Analyse (SVD)
#   Fuer jedes gespeicherte Layout wird die optimale Rigid-Transformation
#   (Rotation + Translation) vom Original-GeoJSON zum gespeicherten Layout
#   berechnet. Die gewichtete mittlere Transformation (neuere Saves = mehr
#   Gewicht) wird auf das Original angewendet -> KI-Vorhersage.
#
#   Feldbreite 1.27 m + Pfad 0.5 m sind als physikalische Constraints bekannt,
#   werden aber nicht erzwungen – der Algorithmus lernt direkt aus den Edits.
# ==============================================================================

def _centroids_array(gdf: gpd.GeoDataFrame):
    """
    Gibt (pts, names) zurueck:
      pts   – (N, 2) float64-Array der Polygon-Schwerpunkte
      names – Liste der zugehoerigen NAME_PROP-Werte
    """
    pts, names = [], []
    for _, row in gdf.iterrows():
        g = row.geometry
        if g is None or g.is_empty:
            continue
        n = str(row[NAME_PROP]) if NAME_PROP in row.index else str(len(names))
        names.append(n)
        pts.append([g.centroid.x, g.centroid.y])
    return np.array(pts, dtype=np.float64), names


def _fit_rigid(src: np.ndarray, dst: np.ndarray):
    """
    Berechnet optimale Rigid-Transformation (Rotation R + Translation t)
    von src nach dst per Procrustes/SVD.
    Gibt (R: 2x2, t: 2,) zurueck  mit:  dst ≈ (R @ src.T).T + t
    """
    sc = src.mean(0)
    dc = dst.mean(0)
    H  = (src - sc).T @ (dst - dc)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:          # Spiegelung verhindern
        Vt[-1] *= -1
        R = Vt.T @ U.T
    return R, dc - R @ sc


def _transform_gdf(gdf: gpd.GeoDataFrame,
                   R: np.ndarray, t: np.ndarray) -> gpd.GeoDataFrame:
    """Wendet Rigid-Transformation (R, t) auf alle Geometrien in gdf an."""
    from shapely.geometry import MultiPolygon as _MP
    gc  = gdf.geometry.name
    out = gdf.copy()
    for idx, row in out.iterrows():
        g = row[gc]
        if g is None or g.is_empty:
            continue
        if g.geom_type == 'Polygon':
            c = np.array(g.exterior.coords)
            out.at[idx, gc] = Polygon((R @ c.T).T + t)
        else:
            new_polys = []
            for poly in g.geoms:
                c = np.array(poly.exterior.coords)
                new_polys.append(Polygon((R @ c.T).T + t))
            out.at[idx, gc] = _MP(new_polys)
    return out


def _weighted_mean_gdf(gdfs_weights: list) -> gpd.GeoDataFrame:
    """
    Gewichtetes geometrisches Mittel der Polygon-Geometrien ueber mehrere GDFs.
    Polygone werden per NAME_PROP abgeglichen.
    Voraussetzung: identische Eckpunkt-Anzahl je Polygon (gilt fuer Rechtecke).
    Fallback fuer nicht passende Shapes: Polygon aus dem hoechstgewichteten GDF.
    """
    if not gdfs_weights:
        return None
    first_gdf, _ = gdfs_weights[0]
    gc  = first_gdf.geometry.name
    out = first_gdf.copy()

    for idx, row in first_gdf.iterrows():
        name = str(row[NAME_PROP]) if NAME_PROP in row.index else None
        g0   = row[gc]
        if g0 is None or g0.is_empty:
            continue
        ref_shape = np.array(g0.exterior.coords).shape

        coords_sum = np.zeros(ref_shape, dtype=np.float64)
        w_sum      = 0.0

        for gdf, w in gdfs_weights:
            match = None
            if name:
                for _, r2 in gdf.iterrows():
                    rn = str(r2[NAME_PROP]) if NAME_PROP in r2.index else ''
                    if rn == name:
                        match = r2[gc]
                        break
            if match is None or match.is_empty:
                continue
            c = np.array(match.exterior.coords)
            if c.shape == ref_shape:
                coords_sum += c * w
                w_sum      += w

        if w_sum > 0:
            try:
                out.at[idx, gc] = Polygon(coords_sum / w_sum)
            except Exception:
                pass   # behalte Originalgeometrie als Fallback
    return out


def learn_and_predict(field: str, target_crs) -> gpd.GeoDataFrame | None:
    """
    Lernalgorithmus – erzeugt eine KI-Vorhersage fuer das Polygon-Layout.

    Ablauf:
      1. Lade Original-GeoJSON des Feldes als Transformations-Referenz.
      2. Fuer jedes gespeicherte Layout: berechne Rigid-Transformation
         Original → gespeichert (Procrustes-Analyse der Schwerpunkte).
      3. Gewichteter Mittelpunkt aller Transformationen
         (exponentiell: neuere Saves > aeltere).
      4. Wende gemittelte Transformation auf Original an → Vorhersage.

    Fallback (kein Original vorhanden):
      Gewichtetes geometrisches Mittel der Vertex-Koordinaten aller Saves.

    Gibt None zurueck wenn zu wenig Daten vorhanden.
    """
    layouts = list_saved_layouts(field)
    if not layouts:
        return None

    # Exponentielles Recency-Gewicht: neuester Save hat Gewicht 2^(n-1)
    n       = len(layouts)
    raw_w   = [2.0 ** i for i in range(n)]
    total_w = sum(raw_w)
    weights = [w / total_w for w in raw_w]

    # Original-GeoJSON laden (Transformations-Referenz)
    orig_gdf = None
    orig_path = geojson_for_field(field)
    if orig_path:
        try:
            og = gpd.read_file(orig_path)
            if og.crs and og.crs != target_crs:
                og = og.to_crs(target_crs)
            orig_gdf = og.reset_index(drop=True)
        except Exception:
            pass

    # Alle gespeicherten Layouts einlesen
    saved_gdfs = []
    for name, w in zip(layouts, weights):
        try:
            gdf = load_layout_file(field, name, target_crs)
            saved_gdfs.append((gdf, w))
        except Exception:
            continue

    if not saved_gdfs:
        return None

    # ── Weg 1: Rigid-Transform-Lernen (bevorzugt, braucht Original) ──────────
    if orig_gdf is not None:
        orig_pts, orig_names = _centroids_array(orig_gdf)
        if len(orig_pts) >= 2:
            Rs, ts, ws = [], [], []
            for gdf, w in saved_gdfs:
                saved_pts, saved_names = _centroids_array(gdf)
                common = sorted(set(orig_names) & set(saved_names))
                if len(common) < 2:
                    continue
                oi = [orig_names.index(c)  for c in common]
                si = [saved_names.index(c) for c in common]
                try:
                    R, t = _fit_rigid(orig_pts[oi], saved_pts[si])
                    Rs.append(R)
                    ts.append(t)
                    ws.append(w)
                except Exception:
                    continue

            if Rs:
                wt = sum(ws)
                # Gewichtete mittlere Translation
                t_mean = sum(t * w for t, w in zip(ts, ws)) / wt
                # Kreismittelwert der Rotationswinkel
                angles   = [np.arctan2(R[1, 0], R[0, 0]) for R in Rs]
                angle_m  = np.arctan2(
                    sum(np.sin(a) * w for a, w in zip(angles, ws)) / wt,
                    sum(np.cos(a) * w for a, w in zip(angles, ws)) / wt)
                R_mean = np.array([[np.cos(angle_m), -np.sin(angle_m)],
                                   [np.sin(angle_m),  np.cos(angle_m)]])
                return _transform_gdf(orig_gdf, R_mean, t_mean)

    # ── Weg 2: Vertex-Mittelwert (Fallback ohne Original) ────────────────────
    return _weighted_mean_gdf(saved_gdfs)


def normalize_rgb(arr: np.ndarray) -> np.ndarray:
    """
    Wandelt (3, H, W) -> (H, W, 3) float32 [0..1] um.
    2-98 Perzentil-Streckung pro Band.
    """
    out = np.empty((arr.shape[1], arr.shape[2], 3), dtype=np.float32)
    for c in range(3):
        band  = arr[c].astype(np.float32)
        valid = band[band > 0]
        if valid.size > 0:
            sample = valid[::10]  # every 10th pixel is enough for percentile
            vmin = float(np.percentile(sample, 2))
            vmax = float(np.percentile(sample, 98))
        else:
            vmin, vmax = 0.0, 1.0
        rng = max(vmax - vmin, 1e-6)
        np.clip((band - vmin) / rng, 0.0, 1.0, out=out[:, :, c])
    return out


# ==============================================================================
# Zeichenfunktionen (getrennt fuer Performance)
# ==============================================================================

def _setup_axes(ax):
    """Gemeinsame Achsen-Formatierung."""
    ax.set_xlabel("X (m)", fontsize=8, color=CLR_TEXT_S)
    ax.set_ylabel("Y (m)", fontsize=8, color=CLR_TEXT_S)
    ax.tick_params(labelsize=7, colors=CLR_TEXT_S)
    for spine in ax.spines.values():
        spine.set_edgecolor(CLR_BORDER_D)


def draw_scene(ax, rgb_norm: np.ndarray, bounds, gdf: gpd.GeoDataFrame,
               selected: set = None, show_handles: bool = False):
    """
    Vollstaendiger Neuaufbau: Hintergrundbild + Polygone.
    Fuer initiales Laden und Zustandsaenderungen (Auswahl, Undo).
    """
    ax.clear()
    ax.set_facecolor("#0f172a")

    ext = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    ax.imshow(rgb_norm, extent=ext, aspect='equal',
              origin='upper', interpolation='bilinear')

    _draw_polygons(ax, gdf, selected, show_handles)
    _setup_axes(ax)


def _draw_polygons(ax, gdf: gpd.GeoDataFrame,
                   selected: set = None, show_handles: bool = False):
    """Nur Polygon-Layer (ohne Hintergrundbild)."""
    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        is_sel     = (selected is not None and idx in selected)
        edge_color = '#22c55e' if is_sel else '#f87171'
        lw         = 2.5       if is_sel else 1.5
        polys      = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)

        for poly in polys:
            xs, ys = poly.exterior.xy
            ax.plot(xs, ys, color=edge_color, linewidth=lw, zorder=3)

            if is_sel and show_handles:
                coords = list(poly.exterior.coords[:-1])
                ax.plot([c[0] for c in coords],
                        [c[1] for c in coords],
                        's', color='#fde047', markersize=9,
                        markeredgecolor='#1e293b', markeredgewidth=0.8,
                        zorder=5)

            name_val = row[NAME_PROP] if NAME_PROP in row.index else str(idx)
            ctr = poly.centroid
            ax.text(ctr.x, ctr.y, str(name_val),
                    fontsize=6.5, color='white',
                    ha='center', va='center', zorder=4,
                    bbox=dict(boxstyle='round,pad=0.25', fc=edge_color,
                              alpha=0.82, ec='none'))


# ==============================================================================
# Hauptfenster
# ==============================================================================

class MainApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Greenness Index · Hoflabor  [ML]")
        self.root.resizable(True, True)
        self.root.configure(bg=CLR_BG)
        self.root.minsize(520, 340)

        self.image_path    = None
        self.image_date    = None
        self.field_name    = None
        self.rgb_array     = None
        self.rgb_norm      = None
        self.raster_bounds = None
        self.raster_crs    = None
        self.raster_shape  = None
        self.gdf           = None

        self._build_ui()

        # Fenster zentrieren
        self.root.update_idletasks()
        geo = screen_geometry(self.root, 0.35, 0.42, min_w=500, min_h=320)
        self.root.geometry(geo)

    # --------------------------------------------------------------------------
    # UI-Aufbau
    # --------------------------------------------------------------------------

    def _build_ui(self):
        outer = tk.Frame(self.root, bg=CLR_BG)
        outer.pack(fill='both', expand=True, padx=28, pady=24)

        # ── Header ────────────────────────────────────────────────────────────
        header = tk.Frame(outer, bg=CLR_BG)
        header.pack(fill='x', pady=(0, 20))

        tk.Frame(header, bg=CLR_GREEN, width=4).pack(
            side='left', fill='y', padx=(0, 16))

        tg = tk.Frame(header, bg=CLR_BG)
        tg.pack(side='left', pady=4)
        tk.Label(tg, text="Greenness Index",
                 font=FONT_TITLE, fg=CLR_TEXT_H, bg=CLR_BG).pack(anchor='w')
        tk.Label(tg, text="RGB Drohnenfoto Auswertung  ·  Hoflabor",
                 font=FONT_SUB, fg=CLR_TEXT_S, bg=CLR_BG).pack(
            anchor='w', pady=(2, 0))

        # ── Schritt 1 ─────────────────────────────────────────────────────────
        c1 = make_card(outer)
        c1.pack(fill='x', pady=(0, 10))

        h1 = tk.Frame(c1, bg=CLR_CARD)
        h1.pack(fill='x', pady=(0, 12))
        tk.Label(h1, text=" 01 ", font=("Helvetica", 9, "bold"),
                 fg=CLR_GREEN, bg=CLR_GREEN_T, padx=6, pady=2).pack(
            side='left', padx=(0, 10))
        tk.Label(h1, text="RGB Bild auswaehlen",
                 font=FONT_STEP, fg=CLR_TEXT_H, bg=CLR_CARD).pack(side='left')

        r1 = tk.Frame(c1, bg=CLR_CARD)
        r1.pack(fill='x')
        make_secondary_btn(r1, "Bild auswaehlen ...",
                           self._select_image).pack(side='left', padx=(0, 18))

        ic = tk.Frame(r1, bg=CLR_CARD)
        ic.pack(side='left', fill='x', expand=True)
        self.lbl_file = tk.Label(ic, text="Kein Bild ausgewaehlt",
                                 font=FONT_LABEL, fg=CLR_TEXT_D,
                                 bg=CLR_CARD, anchor='w')
        self.lbl_file.pack(fill='x')
        self.lbl_date = tk.Label(ic, text="", font=FONT_SMALL,
                                 fg=CLR_TEXT_S, bg=CLR_CARD, anchor='w')
        self.lbl_date.pack(fill='x', pady=(3, 0))

        # ── Schritt 2 ─────────────────────────────────────────────────────────
        c2 = make_card(outer)
        c2.pack(fill='x', pady=(0, 20))

        h2 = tk.Frame(c2, bg=CLR_CARD)
        h2.pack(fill='x', pady=(0, 12))
        tk.Label(h2, text=" 02 ", font=("Helvetica", 9, "bold"),
                 fg=CLR_GREEN, bg=CLR_GREEN_T, padx=6, pady=2).pack(
            side='left', padx=(0, 10))
        tk.Label(h2, text="Feldname waehlen",
                 font=FONT_STEP, fg=CLR_TEXT_H, bg=CLR_CARD).pack(side='left')

        r2 = tk.Frame(c2, bg=CLR_CARD)
        r2.pack(fill='x')
        tk.Label(r2, text="Feld:", font=FONT_LABEL,
                 fg=CLR_TEXT_B, bg=CLR_CARD).pack(side='left', padx=(0, 8))
        self.field_var = tk.StringVar()
        self.field_cb  = ttk.Combobox(r2, textvariable=self.field_var,
                                      state='readonly', width=32,
                                      font=FONT_LABEL)
        self.field_cb.pack(side='left', padx=(0, 12))
        self.field_cb.bind('<<ComboboxSelected>>', lambda _: self._check_ready())
        make_ghost_btn(r2, "Aktualisieren",
                       self._refresh_fields).pack(side='left')
        self._refresh_fields()

        # ── CTA ───────────────────────────────────────────────────────────────
        self.btn_start = make_primary_btn(
            outer, "Vorschau anzeigen  ->",
            self._open_preview, width=30)
        self.btn_start.config(state='disabled', bg=CLR_TEXT_D,
                              activebackground=CLR_TEXT_D)
        self.btn_start.unbind('<Enter>')
        self.btn_start.unbind('<Leave>')
        self.btn_start.pack(fill='x', ipady=4)

    # --------------------------------------------------------------------------

    def _refresh_fields(self):
        fields = available_fields()
        self.field_cb['values'] = fields
        if fields:
            self.field_cb.set(fields[0])
        self._check_ready()

    def _select_image(self):
        init = (DEFAULT_IMG_DIR if os.path.isdir(DEFAULT_IMG_DIR)
                else os.path.expanduser("~"))
        path = filedialog.askopenfilename(
            title="RGB Drohnenbild auswaehlen",
            initialdir=init,
            filetypes=[("GeoTIFF", "*.tif *.tiff"), ("Alle Dateien", "*.*")])
        if not path:
            return
        self.image_path = path
        self.image_date = extract_date(path)
        self.lbl_file.config(text=os.path.basename(path),
                             fg=CLR_TEXT_H,
                             font=("Helvetica", 10, "bold"))
        if self.image_date:
            d = self.image_date
            human = f"{d[4:]}.{d[2:4]}.20{d[:2]}"
            self.lbl_date.config(
                text=f"Datum: {human}   ·   Spalten: {gi_col(d)}, {mgrvi_col(d)}",
                fg=CLR_BLUE)
        else:
            self.lbl_date.config(
                text="Warnung: Kein Datum (_JJMMTT) im Dateinamen erkannt.",
                fg=CLR_AMBER)
        self._check_ready()

    def _check_ready(self):
        ok = bool(self.image_path and self.field_var.get())
        if hasattr(self, 'btn_start'):
            if ok:
                self.btn_start.config(state='normal', bg=CLR_GREEN,
                                      activebackground=CLR_GREEN_H)
                hover_btn(self.btn_start, CLR_GREEN, CLR_GREEN_H)
            else:
                self.btn_start.config(state='disabled', bg=CLR_TEXT_D,
                                      activebackground=CLR_TEXT_D)
                self.btn_start.unbind('<Enter>')
                self.btn_start.unbind('<Leave>')

    def load_data(self) -> bool:
        try:
            MAX_DISPLAY_PX = 3000
            with rasterio.open(self.image_path) as src:
                self.raster_crs    = src.crs
                self.raster_bounds = src.bounds
                self.raster_shape  = (src.height, src.width)
                factor = max(1, src.width  // MAX_DISPLAY_PX,
                                src.height // MAX_DISPLAY_PX)
                out_h = src.height // factor
                out_w = src.width  // factor
                self.rgb_array = src.read(
                    [1, 2, 3],
                    out_shape=(3, out_h, out_w),
                    resampling=rio_Resampling.average)
            self.rgb_norm = normalize_rgb(self.rgb_array)
        except Exception as e:
            messagebox.showerror("Fehler - Raster", str(e))
            return False

        fpath = geojson_for_field(self.field_var.get())
        if not fpath:
            messagebox.showerror(
                "Fehler - Polygone",
                f"Keine GeoJSON-Datei fuer Feld '{self.field_var.get()}' "
                f"gefunden.\nErwartet in: {POLYGONE_DIR}")
            return False
        try:
            gdf = gpd.read_file(fpath)
            if gdf.crs and gdf.crs != self.raster_crs:
                gdf = gdf.to_crs(self.raster_crs)
            self.gdf        = gdf.reset_index(drop=True)
            self.field_name = self.field_var.get()
        except Exception as e:
            messagebox.showerror("Fehler - GeoJSON", str(e))
            return False
        return True

    def _open_preview(self):
        if not self.load_data():
            return
        self.root.withdraw()
        PreviewWindow(self)

    def run_gi_calculation(self, gdf: gpd.GeoDataFrame):
        os.makedirs(RESULTATE_DIR, exist_ok=True)
        poly_dir = os.path.join(RESULTATE_DIR, "polygone_gi")
        os.makedirs(poly_dir, exist_ok=True)

        date_str   = self.image_date or "unbekannt"
        col_gi     = gi_col(date_str)
        col_mgrvi  = mgrvi_col(date_str)
        gi_map     = {}
        mgrvi_map  = {}

        with rasterio.open(self.image_path) as src:
            for _, row in gdf.iterrows():
                name = (str(row[NAME_PROP]) if NAME_PROP in row.index
                        else str(row.name))
                try:
                    out_img, _ = rio_mask(
                        src, [mapping(row.geometry)],
                        crop=True, nodata=0, filled=True)
                    R     = out_img[0].astype(np.float64)
                    G     = out_img[1].astype(np.float64)
                    B     = out_img[2].astype(np.float64)
                    total = R + G + B
                    valid = total > 0

                    # GI = G / (R + G + B)
                    gi = np.where(valid, G / total, np.nan)
                    gi_map[name] = (float(np.nanmean(gi))
                                    if valid.any() else np.nan)

                    # MGRVI = (G² - R²) / (G² + R²)
                    g2      = G ** 2
                    r2      = R ** 2
                    denom   = g2 + r2
                    valid_m = denom > 0
                    mgrvi   = np.where(valid_m, (g2 - r2) / denom, np.nan)
                    mgrvi_map[name] = (float(np.nanmean(mgrvi))
                                       if valid_m.any() else np.nan)

                except Exception as exc:
                    gi_map[name]    = np.nan
                    mgrvi_map[name] = np.nan
                    print(f"  Warnung Polygon '{name}': {exc}")

        gdf_out = gdf.copy()
        if NAME_PROP in gdf_out.columns:
            gdf_out['GI']    = gdf_out[NAME_PROP].astype(str).map(gi_map)
            gdf_out['MGRVI'] = gdf_out[NAME_PROP].astype(str).map(mgrvi_map)
        else:
            gdf_out['GI']    = np.nan
            gdf_out['MGRVI'] = np.nan

        safe_name = re.sub(r'[\\/:*?"<>|]', '_', self.field_name)
        poly_path = os.path.join(poly_dir, f"{safe_name}_{date_str}.geojson")
        gdf_out.to_file(poly_path, driver='GeoJSON')

        csv_path = os.path.join(RESULTATE_DIR, "greenness_index.csv")
        new_df   = pd.DataFrame({
            col_gi:    pd.Series(gi_map),
            col_mgrvi: pd.Series(mgrvi_map),
        })
        new_df.index.name = 'Streifen'

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, index_col=0)
            df.index.name = 'Streifen'
            for col, series in new_df.items():
                if col in df.columns:
                    for strip, val in series.items():
                        df.loc[strip, col] = val
                else:
                    df = df.join(series, how='outer')
        else:
            df = new_df
        df.to_csv(csv_path)

        n_ok  = sum(1 for v in gi_map.values() if not np.isnan(v))
        n_all = len(gi_map)
        messagebox.showinfo(
            "Fertig - Indizes berechnet",
            f"Berechnung abgeschlossen!\n\n"
            f"Polygone berechnet: {n_ok} / {n_all}\n\n"
            f"Gespeichert in:\n  {RESULTATE_DIR}\n\n"
            f"  Polygone: polygone_gi/{os.path.basename(poly_path)}\n"
            f"  CSV:      greenness_index.csv\n"
            f"  Spalten:  {col_gi}, {col_mgrvi}")


# ==============================================================================
# Vorschaufenster
# ==============================================================================

class PreviewWindow:
    def __init__(self, app: MainApp):
        self.app = app
        self.win = tk.Toplevel(app.root)
        d = app.image_date or "?"
        self.win.title(f"Vorschau  ·  {app.field_name}  ·  {d}")
        self.win.configure(bg=CLR_BG)
        self.win.minsize(640, 480)
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

        # Bildschirmangepasste Groesse
        self.win.geometry(
            screen_geometry(self.win, 0.65, 0.82, min_w=800, min_h=600))

        self._panning      = False
        self._pan_last_pix = None
        self._build()

    def _on_close(self):
        self.app.root.deiconify()
        self.win.destroy()

    def _build(self):
        plt.rcParams.update({
            'figure.facecolor': CLR_BG,
            'axes.facecolor':   '#0f172a',
            'axes.edgecolor':   CLR_BORDER_D,
            'xtick.color':      CLR_TEXT_S,
            'ytick.color':      CLR_TEXT_S,
            'text.color':       CLR_TEXT_B,
        })

        # ── Reihenfolge: erst feste Elemente (bottom), dann Canvas (expand) ──
        # Buttons ganz unten – werden zuerst gepackt, damit sie immer sichtbar
        bf = tk.Frame(self.win, bg=CLR_CARD,
                      highlightbackground=CLR_BORDER, highlightthickness=1)
        bf.pack(side='bottom', fill='x', padx=8, pady=(0, 8))

        btn_row = tk.Frame(bf, bg=CLR_CARD)
        btn_row.pack(fill='x', padx=14, pady=10)
        make_primary_btn(btn_row, "Greenness Index berechnen",
                         self._calc).pack(side='left', padx=(0, 8))
        make_secondary_btn(btn_row, "Polygone bearbeiten",
                           self._edit).pack(side='left')
        make_ghost_btn(btn_row, "Schliessen",
                       self._on_close).pack(side='right')

        # Toolbar direkt ueber den Buttons
        toolbar_frame = tk.Frame(self.win, bg=CLR_CARD,
                                 highlightbackground=CLR_BORDER,
                                 highlightthickness=1)
        toolbar_frame.pack(side='bottom', fill='x', padx=8, pady=(0, 2))
        self.toolbar = NavigationToolbar2Tk.__new__(NavigationToolbar2Tk)

        # Canvas fuellt den verbleibenden Platz (nach den fixen Elementen)
        canvas_frame = tk.Frame(self.win, bg=CLR_BG)
        canvas_frame.pack(side='top', fill='both', expand=True,
                          padx=8, pady=(8, 0))

        fw, fh = canvas_figsize(self.win, w_frac=0.62, h_frac=0.60)
        self.fig, self.ax = plt.subplots(figsize=(fw, fh), dpi=72)
        self.fig.patch.set_facecolor(CLR_BG)
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        # Toolbar nachtraeglich initialisieren (nach canvas)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

        # Maus-Events
        self.canvas.mpl_connect('scroll_event',         self._on_scroll)
        self.canvas.mpl_connect('button_press_event',   self._on_pan_press)
        self.canvas.mpl_connect('motion_notify_event',  self._on_pan_motion)
        self.canvas.mpl_connect('button_release_event', self._on_pan_release)

        self._draw()

    def _draw(self):
        draw_scene(self.ax, self.app.rgb_norm,
                   self.app.raster_bounds, self.app.gdf)
        self.ax.set_title(
            f"{self.app.field_name}  ·  {self.app.image_date or 'kein Datum'}",
            fontsize=11, color=CLR_TEXT_H, pad=10)
        self.fig.tight_layout(pad=1.2)
        self.canvas.draw()

    def _calc(self):
        self.app.run_gi_calculation(self.app.gdf)

    def _edit(self):
        self.win.withdraw()
        EditorWindow(self.app, self)

    def _on_scroll(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        now = time.monotonic()
        if now - getattr(self, '_last_scroll_t', 0.0) < 0.05:
            return
        self._last_scroll_t = now
        factor = 0.8 if event.button == 'up' else 1.25
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        xc, yc = event.xdata, event.ydata
        self.ax.set_xlim([xc + (x - xc) * factor for x in xlim])
        self.ax.set_ylim([yc + (y - yc) * factor for y in ylim])
        self.canvas.draw_idle()

    def _on_pan_press(self, event):
        if event.button == 3 and event.inaxes == self.ax:
            self._panning      = True
            self._pan_last_pix = (event.x, event.y)

    def _on_pan_motion(self, event):
        if not self._panning or self._pan_last_pix is None:
            return
        if event.x is None or event.y is None:
            return
        dx_pix = event.x - self._pan_last_pix[0]
        dy_pix = event.y - self._pan_last_pix[1]
        self._pan_last_pix = (event.x, event.y)
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        bbox = self.ax.get_window_extent()
        if bbox.width == 0 or bbox.height == 0:
            return
        xs = (xlim[1] - xlim[0]) / bbox.width
        ys = (ylim[1] - ylim[0]) / bbox.height
        self.ax.set_xlim(xlim[0] - dx_pix * xs, xlim[1] - dx_pix * xs)
        self.ax.set_ylim(ylim[0] + dy_pix * ys, ylim[1] + dy_pix * ys)
        self.canvas.draw_idle()

    def _on_pan_release(self, event):
        if event.button == 3:
            self._panning      = False
            self._pan_last_pix = None


# ==============================================================================
# Editor-Fenster
# ==============================================================================

class EditorWindow:
    """
    Interaktives Polygon-Bearbeitungsfenster.

    Maussteuerung:
      Linksklick auf Polygon      -> Auswahlen
      Ctrl + Linksklick           -> Mehrfachauswahl
      Linksklick auf Leerraum     -> Auswahl aufheben
      Polygon ziehen              -> Verschieben
      Eckpunkt (gelbes Quadrat)
        ziehen                   -> Eckpunkt verschieben

    Tastatur:
      Ctrl + Z                    -> Letzten Schritt rueckgaengig
    """

    def __init__(self, app: MainApp, preview: PreviewWindow):
        self.app     = app
        self.preview = preview
        self.win     = tk.Toplevel(app.root)
        self.win.title(f"Polygon Editor  ·  {app.field_name}")
        self.win.configure(bg=CLR_BG)
        self.win.minsize(800, 560)
        self.win.protocol("WM_DELETE_WINDOW", self._cancel)

        self.win.geometry(
            screen_geometry(self.win, 0.78, 0.88, min_w=900, min_h=680))

        self.gdf_work   = app.gdf.copy()
        self.undo_stack = []

        self.selected    = set()
        self._drag_mode  = None
        self._drag_start = None
        self._act_corner = None
        self._mouse_down = False
        self._ctrl_held  = False
        self._panning      = False
        self._pan_last_pix = None

        # Performance: Zeitpunkt des letzten Drag-Redraws
        self._last_drag_t   = 0.0
        self._last_scroll_t = 0.0
        # Performance: gecachtes Hintergrundbild (blit)
        self._bg_cache = None
        # Performance: rauemlicher Index fuer schnelle Polygon-Treffersuche
        self._sindex = None

        self._build()
        self._rebuild_sindex()
        self.win.bind('<Control-z>', lambda _: self._undo())
        self.win.bind('<Control-Z>', lambda _: self._undo())

    # --------------------------------------------------------------------------
    # UI-Aufbau
    # --------------------------------------------------------------------------

    def _build(self):
        # ── Hint-Leiste (ganz oben) ───────────────────────────────────────────
        bar = tk.Frame(self.win, bg=CLR_HINT_BG,
                       highlightbackground=CLR_HINT_BD, highlightthickness=1)
        bar.pack(side='top', fill='x')
        tk.Label(bar,
                 text=("Klick = Auswaehlen    Ctrl+Klick = Mehrfach    "
                       "Polygon ziehen = Verschieben    "
                       "Ecke (Quadrat) ziehen = Verformen    "
                       "Ctrl+Z = Rueckgaengig"),
                 bg=CLR_HINT_BG, fg=CLR_BLUE, font=FONT_SMALL
                 ).pack(side='left', padx=14, pady=6)
        make_danger_btn(bar, "Rueckgaengig  Ctrl+Z",
                        self._undo).pack(side='right', padx=10, pady=5)

        # ── Aktionsbuttons (ganz unten) – VOR dem Canvas packen! ─────────────
        bf = tk.Frame(self.win, bg=CLR_CARD,
                      highlightbackground=CLR_BORDER, highlightthickness=1)
        bf.pack(side='bottom', fill='x', padx=8, pady=(0, 8))
        btn_row = tk.Frame(bf, bg=CLR_CARD)
        btn_row.pack(fill='x', padx=14, pady=10)
        make_primary_btn(btn_row, "Greenness Index berechnen",
                         self._calculate).pack(side='left', padx=(0, 8))
        make_secondary_btn(btn_row, "Layout speichern",
                           self._save_layout).pack(side='left', padx=(0, 8))
        # "Layout laden"-Button nur anzeigen wenn gespeicherte Layouts vorhanden
        self.btn_load_layout = make_ghost_btn(
            btn_row, "Layout laden", self._load_layout_dialog)
        self._refresh_load_btn()
        make_danger_btn(btn_row, "Abbrechen",
                        self._cancel).pack(side='right')

        # ── Transformationspanel (ueber Buttons) – ebenfalls VOR Canvas ──────
        tf = make_card(self.win, pady=10)
        tf.pack(side='bottom', fill='x', padx=8, pady=(0, 4))

        tf_hdr = tk.Frame(tf, bg=CLR_CARD)
        tf_hdr.pack(fill='x', pady=(0, 10))
        tk.Label(tf_hdr, text="Ausgewaehlte Polygone transformieren",
                 font=FONT_STEP, fg=CLR_TEXT_H, bg=CLR_CARD).pack(side='left')

        grid = tk.Frame(tf, bg=CLR_CARD)
        grid.pack(fill='x')

        def _lbl(text, row, col):
            tk.Label(grid, text=text, font=FONT_LABEL,
                     fg=CLR_TEXT_B, bg=CLR_CARD).grid(
                row=row, column=col, sticky='e', padx=(0, 6), pady=3)

        def _entry(var, row, col):
            e = tk.Entry(grid, textvariable=var, width=8,
                         font=FONT_LABEL, bg="#f8fafc", fg=CLR_TEXT_H,
                         relief='flat',
                         highlightbackground=CLR_BORDER, highlightthickness=1,
                         insertbackground=CLR_TEXT_H)
            e.grid(row=row, column=col, padx=(0, 10), pady=3, ipady=3)
            return e

        _lbl("Rotieren (Grad):",   0, 0)
        self.var_rot = tk.StringVar(value="0")
        _entry(self.var_rot, 0, 1)
        make_secondary_btn(grid, "Anwenden", self._apply_rotate).grid(
            row=0, column=2, padx=(0, 28), pady=3)

        _lbl("Skalieren (Faktor):", 0, 3)
        self.var_scl = tk.StringVar(value="1.0")
        _entry(self.var_scl, 0, 4)
        make_secondary_btn(grid, "Anwenden", self._apply_scale).grid(
            row=0, column=5, pady=3)

        _lbl("Verschieben  X (m):", 1, 0)
        self.var_dx = tk.StringVar(value="0")
        _entry(self.var_dx, 1, 1)
        _lbl("Y (m):", 1, 2)
        self.var_dy = tk.StringVar(value="0")
        _entry(self.var_dy, 1, 3)
        make_secondary_btn(grid, "Verschieben", self._apply_translate).grid(
            row=1, column=4, columnspan=2, pady=3, sticky='w')

        # ── Toolbar (ueber Transform-Panel) ───────────────────────────────────
        toolbar_frame = tk.Frame(self.win, bg=CLR_CARD,
                                 highlightbackground=CLR_BORDER,
                                 highlightthickness=1)
        toolbar_frame.pack(side='bottom', fill='x', padx=8, pady=(0, 2))

        # ── Canvas (fuellt den restlichen Platz) ──────────────────────────────
        canvas_frame = tk.Frame(self.win, bg=CLR_BG)
        canvas_frame.pack(side='top', fill='both', expand=True,
                          padx=8, pady=(6, 0))

        fw, fh = canvas_figsize(self.win, w_frac=0.74, h_frac=0.55)
        self.fig, self.ax = plt.subplots(figsize=(fw, fh), dpi=72)
        self.fig.patch.set_facecolor(CLR_BG)
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

        # Maus-Events
        self.canvas.mpl_connect('scroll_event',         self._on_scroll)
        self.canvas.mpl_connect('button_press_event',   self._on_press)
        self.canvas.mpl_connect('motion_notify_event',  self._on_motion)
        self.canvas.mpl_connect('button_release_event', self._on_release)
        self.canvas.mpl_connect(
            'key_press_event',
            lambda e: setattr(self, '_ctrl_held',
                              e.key is not None
                              and 'control' in str(e.key).lower()))
        self.canvas.mpl_connect(
            'key_release_event',
            lambda e: setattr(self, '_ctrl_held', False))

        self._redraw()

    # --------------------------------------------------------------------------
    # Zeichnen (mit Hintergrund-Cache fuer schnelle Drag-Updates)
    # --------------------------------------------------------------------------

    def _redraw(self):
        """Vollstaendiger Neuaufbau (Auswahl geaendert, Undo, etc.)."""
        # Zoom-Stand sichern (ax.images ist leer beim allerersten Aufruf)
        xlim_save = self.ax.get_xlim() if self.ax.images else None
        ylim_save = self.ax.get_ylim() if self.ax.images else None

        # -- Hintergrundbild zeichnen (ohne Polygone) --------------------------
        bounds = self.app.raster_bounds
        self.ax.clear()
        self.ax.set_facecolor("#0f172a")
        ext = [bounds.left, bounds.right, bounds.bottom, bounds.top]
        self.ax.imshow(self.app.rgb_norm, extent=ext, aspect='equal',
                       origin='upper', interpolation='bilinear')
        _setup_axes(self.ax)
        # Zoom wiederherstellen (verhindert Zurueckspringen nach Auswahl/Undo)
        if xlim_save is not None:
            self.ax.set_xlim(xlim_save)
            self.ax.set_ylim(ylim_save)
        n = len(self.selected)
        self.ax.set_title(
            f"Editor  ·  {self.app.field_name}  ·  "
            f"{n} Polygon{'e' if n != 1 else ''} ausgewaehlt",
            fontsize=10, color=CLR_TEXT_H, pad=8)
        self.fig.tight_layout(pad=0.5)
        self.canvas.draw()
        # Hintergrund VOR Polygonen cachen (fuer schnellen blit beim Drag)
        self._bg_cache = self.canvas.copy_from_bbox(self.fig.bbox)
        # -- Polygone per blit auf den gecachten Hintergrund zeichnen ----------
        _draw_polygons(self.ax, self.gdf_work,
                       selected=self.selected, show_handles=True)
        for artist in self.ax.lines + self.ax.texts + self.ax.collections:
            self.ax.draw_artist(artist)
        self.canvas.blit(self.fig.bbox)

    def _fast_redraw(self):
        """
        Schneller Redraw waehrend Drag: stellt gecachten Hintergrund wieder her
        und zeichnet nur den Polygon-Layer neu (kein teures imshow).
        """
        if self._bg_cache is None:
            self._redraw()
            return
        # Alte Polygon-Artists entfernen (verhindert Akkumulation)
        for artist in (list(self.ax.lines)
                       + list(self.ax.texts)
                       + list(self.ax.collections)):
            artist.remove()
        # Gecachten Hintergrund (nur Bild, ohne Polygone) wiederherstellen
        self.canvas.restore_region(self._bg_cache)
        # Polygone in neuer Position zeichnen und per blit ausgeben
        _draw_polygons(self.ax, self.gdf_work,
                       selected=self.selected, show_handles=True)
        for artist in self.ax.lines + self.ax.texts + self.ax.collections:
            self.ax.draw_artist(artist)
        self.canvas.blit(self.fig.bbox)

    def _rebuild_sindex(self):
        """Rauemlichen Index neu aufbauen (nach Aenderungen an gdf_work)."""
        self._sindex = self.gdf_work.sindex

    # --------------------------------------------------------------------------
    # Undo
    # --------------------------------------------------------------------------

    def _push_undo(self):
        self.undo_stack.append(self.gdf_work.copy())

    def _undo(self):
        if self.undo_stack:
            self.gdf_work = self.undo_stack.pop()
            self.selected.clear()
            self._rebuild_sindex()
            self._redraw()
        else:
            messagebox.showinfo("Rueckgaengig",
                                "Keine weiteren Schritte zum Rueckgaengigmachen.")

    # --------------------------------------------------------------------------
    # Hilfsmethoden fuer Interaktion
    # --------------------------------------------------------------------------

    def _geo_col(self) -> str:
        return self.gdf_work.geometry.name

    def _tol(self) -> float:
        b = self.app.raster_bounds
        return (b.right - b.left) / self.app.raster_shape[1] * 8

    def _poly_at(self, x, y):
        pt  = Point(x, y)
        tol = self._tol()
        gc  = self._geo_col()
        if self._sindex is not None:
            candidates = [self.gdf_work.index[pos]
                          for pos in self._sindex.query(
                              shp_box(x - tol, y - tol, x + tol, y + tol))]
        else:
            candidates = list(self.gdf_work.index)
        for idx in candidates:
            g = self.gdf_work.at[idx, gc]
            if g and not g.is_empty:
                if g.contains(pt) or g.distance(pt) < tol:
                    return idx
        return None

    def _corner_at(self, x, y):
        tol = self._tol()
        for idx in self.selected:
            g = self.gdf_work.at[idx, self._geo_col()]
            if g is None or g.is_empty:
                continue
            polys = ([g] if g.geom_type == 'Polygon' else list(g.geoms))
            for poly in polys:
                for ci, (cx, cy) in enumerate(poly.exterior.coords[:-1]):
                    if abs(cx - x) < tol and abs(cy - y) < tol:
                        return (idx, ci)
        return None

    # --------------------------------------------------------------------------
    # Mausereignisse
    # --------------------------------------------------------------------------

    def _on_scroll(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        now = time.monotonic()
        if now - self._last_scroll_t < 0.05:
            return
        self._last_scroll_t = now
        self._bg_cache = None  # Ansicht hat sich geaendert, Cache ungueltig
        factor = 0.8 if event.button == 'up' else 1.25
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        xc, yc = event.xdata, event.ydata
        self.ax.set_xlim([xc + (x - xc) * factor for x in xlim])
        self.ax.set_ylim([yc + (y - yc) * factor for y in ylim])
        self.canvas.draw_idle()

    def _on_press(self, event):
        if event.button == 3 and event.inaxes == self.ax:
            self._panning      = True
            self._pan_last_pix = (event.x, event.y)
            return
        if getattr(self, 'toolbar', None) and self.toolbar.mode != '':
            return
        if (event.inaxes != self.ax
                or event.xdata is None
                or event.button != 1):
            return
        self._mouse_down = True
        x, y = event.xdata, event.ydata

        corner = self._corner_at(x, y)
        if corner:
            self._push_undo()
            self._drag_mode  = 'corner'
            self._act_corner = corner
            self._drag_start = (x, y)
            return

        idx  = self._poly_at(x, y)
        ctrl = (self._ctrl_held
                or (event.key is not None
                    and 'control' in str(event.key).lower()))

        if idx is not None:
            if ctrl:
                if idx in self.selected:
                    self.selected.discard(idx)
                else:
                    self.selected.add(idx)
            else:
                if idx not in self.selected:
                    self.selected = {idx}
                self._push_undo()
                self._drag_mode  = 'move'
                self._drag_start = (x, y)
        else:
            if not ctrl:
                self.selected.clear()

        self._redraw()

    def _on_motion(self, event):
        # Rechtsklick-Pan (kein Throttle noetig – nur Achsen)
        if self._panning and self._pan_last_pix is not None:
            if event.x is not None and event.y is not None:
                dx_pix = event.x - self._pan_last_pix[0]
                dy_pix = event.y - self._pan_last_pix[1]
                self._pan_last_pix = (event.x, event.y)
                xlim = self.ax.get_xlim()
                ylim = self.ax.get_ylim()
                bbox = self.ax.get_window_extent()
                if bbox.width > 0 and bbox.height > 0:
                    xs = (xlim[1] - xlim[0]) / bbox.width
                    ys = (ylim[1] - ylim[0]) / bbox.height
                    self.ax.set_xlim(xlim[0] - dx_pix * xs,
                                     xlim[1] - dx_pix * xs)
                    self.ax.set_ylim(ylim[0] + dy_pix * ys,
                                     ylim[1] + dy_pix * ys)
                    self.canvas.draw_idle()
            return

        if getattr(self, 'toolbar', None) and self.toolbar.mode != '':
            return
        if (not self._mouse_down
                or event.inaxes != self.ax
                or event.xdata is None
                or self._drag_start is None
                or self._drag_mode is None):
            return

        # Throttle: Drag-Redraws auf max. ~30 fps begrenzen
        now = time.monotonic()
        if now - self._last_drag_t < _DRAG_MIN_INTERVAL:
            return
        self._last_drag_t = now

        x, y = event.xdata, event.ydata
        dx   = x - self._drag_start[0]
        dy   = y - self._drag_start[1]
        gc   = self._geo_col()

        if self._drag_mode == 'move':
            for idx in self.selected:
                self.gdf_work.at[idx, gc] = shp_translate(
                    self.gdf_work.at[idx, gc], dx, dy)
            self._drag_start = (x, y)
            self._fast_redraw()

        elif self._drag_mode == 'corner' and self._act_corner:
            ridx, cidx = self._act_corner
            g = self.gdf_work.at[ridx, gc]
            if g and g.geom_type == 'Polygon':
                coords       = list(g.exterior.coords[:-1])
                coords[cidx] = (x, y)
                try:
                    ng = Polygon(coords)
                    if ng.is_valid and not ng.is_empty:
                        self.gdf_work.at[ridx, gc] = ng
                        self._drag_start = (x, y)
                        self._fast_redraw()
                except Exception:
                    pass

    def _on_release(self, event):
        if event.button == 3:
            self._panning      = False
            self._pan_last_pix = None
            return
        if self._drag_mode is not None:
            # Abschliessender sauberer Redraw nach dem Loslassen
            self._rebuild_sindex()
            self._redraw()
        self._mouse_down = False
        self._drag_mode  = None
        self._drag_start = None
        self._act_corner = None

    # --------------------------------------------------------------------------
    # Numerische Transformationen
    # --------------------------------------------------------------------------

    def _need_selection(self) -> bool:
        if not self.selected:
            messagebox.showinfo("Hinweis",
                                "Bitte zuerst mindestens ein Polygon auswaehlen.")
            return False
        return True

    def _apply_rotate(self):
        if not self._need_selection():
            return
        try:
            angle = float(self.var_rot.get())
        except ValueError:
            messagebox.showerror("Fehler",
                                 "Ungültiger Winkel - bitte eine Zahl eingeben.")
            return
        self._push_undo()
        gc = self._geo_col()
        for idx in self.selected:
            self.gdf_work.at[idx, gc] = shp_rotate(
                self.gdf_work.at[idx, gc], angle, origin='centroid')
        self._rebuild_sindex()
        self._redraw()

    def _apply_scale(self):
        if not self._need_selection():
            return
        try:
            f = float(self.var_scl.get())
        except ValueError:
            messagebox.showerror(
                "Fehler",
                "Ungültiger Skalierungsfaktor - bitte eine Zahl eingeben.")
            return
        self._push_undo()
        gc = self._geo_col()
        for idx in self.selected:
            self.gdf_work.at[idx, gc] = shp_scale(
                self.gdf_work.at[idx, gc], f, f, origin='centroid')
        self._rebuild_sindex()
        self._redraw()

    def _apply_translate(self):
        if not self._need_selection():
            return
        try:
            dx = float(self.var_dx.get())
            dy = float(self.var_dy.get())
        except ValueError:
            messagebox.showerror("Fehler",
                                 "Ungültige Werte - bitte Zahlen eingeben.")
            return
        self._push_undo()
        gc = self._geo_col()
        for idx in self.selected:
            self.gdf_work.at[idx, gc] = shp_translate(
                self.gdf_work.at[idx, gc], dx, dy)
        self._rebuild_sindex()
        self._redraw()

    # --------------------------------------------------------------------------
    # Layout speichern / laden
    # --------------------------------------------------------------------------

    def _refresh_load_btn(self):
        """Zeigt oder versteckt den 'Layout laden'-Button je nach Verfuegbarkeit."""
        has = bool(list_saved_layouts(self.app.field_name))
        if has:
            self.btn_load_layout.pack(side='left', padx=(0, 8))
        else:
            self.btn_load_layout.pack_forget()

    def _save_layout(self):
        name = simpledialog.askstring(
            "Layout speichern",
            f"Name fuer dieses Layout (Feld: {self.app.field_name}):",
            parent=self.win)
        if not name or not name.strip():
            return
        name = name.strip()
        try:
            path = save_layout_file(self.gdf_work, self.app.field_name, name)
            self._refresh_load_btn()
            messagebox.showinfo(
                "Layout gespeichert",
                f"Layout '{name}' wurde gespeichert:\n{path}",
                parent=self.win)
        except Exception as exc:
            messagebox.showerror("Fehler beim Speichern", str(exc),
                                 parent=self.win)

    def _load_layout_dialog(self):
        layouts = list_saved_layouts(self.app.field_name)
        if not layouts:
            messagebox.showinfo("Keine Layouts",
                                "Es sind noch keine Layouts gespeichert.",
                                parent=self.win)
            return

        # ── KI-Vorhersage berechnen (laeuft schnell, nur Lineralgebra) ────────
        pred_gdf = None
        try:
            pred_gdf = learn_and_predict(self.app.field_name,
                                         self.app.raster_crs)
        except Exception:
            pass

        # entries = list of (display_label, kind, data)
        # kind 'pred'  -> data ist GeoDataFrame
        # kind 'saved' -> data ist Layout-Name (str)
        entries = []
        if pred_gdf is not None:
            n   = len(layouts)
            lbl = (f"\u2728  KI-Vorhersage  "
                   f"({n} Layout{'s' if n > 1 else ''} gelernt)")
            entries.append((lbl, 'pred', pred_gdf))
        for name in layouts:
            entries.append((name, 'saved', name))

        # ── Dialog aufbauen ───────────────────────────────────────────────────
        dlg = tk.Toplevel(self.win)
        dlg.title(f"Layout laden  ·  {self.app.field_name}")
        dlg.configure(bg=CLR_BG)
        dlg.resizable(False, False)
        dlg.grab_set()

        self.win.update_idletasks()
        wx = self.win.winfo_x() + self.win.winfo_width()  // 2 - 230
        wy = self.win.winfo_y() + self.win.winfo_height() // 2 - 150
        dlg.geometry(f"460x300+{wx}+{wy}")

        tk.Label(dlg,
                 text=f"Layout laden  \u00b7  {self.app.field_name}",
                 font=FONT_STEP, fg=CLR_TEXT_H, bg=CLR_BG
                 ).pack(pady=(18, 4), padx=20, anchor='w')

        if pred_gdf is not None:
            tk.Label(dlg,
                     text=(f"Der Lernalgorithmus hat aus {len(layouts)} "
                           f"gespeicherten Layout{'s' if len(layouts)>1 else ''} "
                           f"eine Vorhersage berechnet."),
                     font=FONT_SMALL, fg=CLR_TEXT_S, bg=CLR_BG,
                     wraplength=420, justify='left'
                     ).pack(pady=(0, 8), padx=20, anchor='w')

        frame = tk.Frame(dlg, bg=CLR_CARD,
                         highlightbackground=CLR_BORDER, highlightthickness=1)
        frame.pack(fill='both', expand=True, padx=20)

        scrollbar = tk.Scrollbar(frame, orient='vertical')
        lb = tk.Listbox(frame, font=FONT_LABEL, bg=CLR_CARD, fg=CLR_TEXT_H,
                        selectbackground=CLR_GREEN, selectforeground='white',
                        activestyle='none', relief='flat', bd=0,
                        yscrollcommand=scrollbar.set, height=7)
        scrollbar.config(command=lb.yview)
        scrollbar.pack(side='right', fill='y')
        lb.pack(side='left', fill='both', expand=True, padx=6, pady=6)

        for i, (label, kind, _) in enumerate(entries):
            lb.insert('end', '  ' + label)
            if kind == 'pred':
                lb.itemconfig(i, fg=CLR_GREEN,
                              selectbackground=CLR_GREEN_H)
        lb.selection_set(0)

        # Separator-Linie nach Vorhersage-Eintrag
        if pred_gdf is not None and len(layouts) > 0:
            lb.insert(len(entries) if False else 1,
                      '  \u2500' * 22)
            lb_entries_with_sep = (
                entries[:1]
                + [('─' * 22, 'sep', None)]
                + entries[1:])
            # Trennlinie nicht selektierbar (via Farbe kennzeichnen)
            lb.itemconfig(1, fg=CLR_BORDER_D, selectbackground=CLR_CARD,
                          selectforeground=CLR_BORDER_D)
        else:
            lb_entries_with_sep = entries

        def _do_load():
            sel = lb.curselection()
            if not sel:
                return
            pos = sel[0]
            lbl, kind, data = lb_entries_with_sep[pos]
            if kind == 'sep':
                return
            dlg.destroy()
            try:
                if kind == 'pred':
                    new_gdf      = data
                    display_name = "KI-Vorhersage"
                else:
                    new_gdf      = load_layout_file(
                        self.app.field_name, data, self.app.raster_crs)
                    display_name = f"Layout '{data}'"
                self._push_undo()
                self.gdf_work = new_gdf
                self.selected.clear()
                self._rebuild_sindex()
                self._redraw()
                messagebox.showinfo(
                    "Layout geladen",
                    f"{display_name} wurde geladen.",
                    parent=self.win)
            except Exception as exc:
                messagebox.showerror("Fehler beim Laden", str(exc),
                                     parent=self.win)

        btn_row = tk.Frame(dlg, bg=CLR_BG)
        btn_row.pack(fill='x', padx=20, pady=12)
        make_primary_btn(btn_row, "Laden", _do_load,
                         width=14).pack(side='left', padx=(0, 10))
        make_ghost_btn(btn_row, "Abbrechen",
                       dlg.destroy).pack(side='left')

        lb.bind('<Double-Button-1>', lambda _: _do_load())
        dlg.bind('<Return>', lambda _: _do_load())
        dlg.bind('<Escape>', lambda _: dlg.destroy())

    # --------------------------------------------------------------------------
    # Abschluss-Aktionen
    # --------------------------------------------------------------------------

    def _calculate(self):
        self.app.gdf = self.gdf_work.copy()
        self.app.run_gi_calculation(self.app.gdf)
        self.win.destroy()
        self.preview.win.destroy()
        self.app.root.deiconify()

    def _cancel(self):
        self.win.destroy()
        self.preview.win.deiconify()


# ==============================================================================
# Einstiegspunkt
# ==============================================================================

if __name__ == '__main__':
    root = tk.Tk()
    MainApp(root)
    root.mainloop()
