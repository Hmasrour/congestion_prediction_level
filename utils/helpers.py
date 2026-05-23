"""
utils/helpers.py
================
Fonctions utilitaires partagées entre les modules.
"""

import logging
import os
from math import radians, sin, cos, sqrt, atan2
from typing import Dict, Tuple

import numpy as np
import pandas as pd


# ── Logger centralisé ──────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    """Retourne un logger formaté pour le projet."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# ── Géographie ─────────────────────────────────────────────────
def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance en mètres entre deux points GPS (formule haversine)."""
    R = 6_371_000
    p1, p2 = radians(lat1), radians(lat2)
    dp = radians(lat2 - lat1)
    dl = radians(lon2 - lon1)
    a = sin(dp / 2) ** 2 + cos(p1) * cos(p2) * sin(dl / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def nearest_road_point(
    seg_lat: float,
    seg_lon: float,
    road_points: Dict[str, Tuple[float, float]],
    radius_m: float,
) -> Tuple[str | None, float]:
    """
    Retourne (road_id, distance_m) du point le plus proche dans le rayon,
    ou (None, inf) si aucun point n'est dans le rayon.
    """
    best_id, best_dist = None, float("inf")
    for road_id, (lat, lon) in road_points.items():
        d = haversine_m(seg_lat, seg_lon, lat, lon)
        if d < best_dist:
            best_id, best_dist = road_id, d
    if best_dist <= radius_m:
        return best_id, best_dist
    return None, best_dist


# ── Série temporelle ───────────────────────────────────────────
def make_lag_features(series: pd.Series, n_lags: int) -> pd.DataFrame:
    """Crée un DataFrame de features lag pour les modèles ML."""
    df = pd.DataFrame({"y": series.values})
    for i in range(1, n_lags + 1):
        df[f"lag_{i}"] = df["y"].shift(i)
    if hasattr(series.index, "hour"):
        df["hour"]    = series.index.hour
        df["weekday"] = series.index.dayofweek
    return df.dropna()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calcule MAE, RMSE, MAPE et R² entre valeurs réelles et prédites."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    n = min(len(y_true), len(y_pred))
    yt, yp = y_true[:n], y_pred[:n]
    denom = np.where(yt == 0, 1, yt)
    return {
        "MAE (km/h)":  round(float(mean_absolute_error(yt, yp)), 3),
        "RMSE (km/h)": round(float(np.sqrt(mean_squared_error(yt, yp))), 3),
        "MAPE (%)":    round(float(np.mean(np.abs((yt - yp) / denom)) * 100), 2),
        "R²":          round(float(r2_score(yt, yp)), 4),
    }


# ── Fichiers ───────────────────────────────────────────────────
def ensure_dirs(*paths: str) -> None:
    """Crée les répertoires s'ils n'existent pas."""
    for path in paths:
        os.makedirs(path, exist_ok=True)
