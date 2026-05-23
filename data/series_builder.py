"""
data/series_builder.py
======================
Classe TimeSeriesBuilder
Transforme le DataFrame TomTom en série temporelle horaire prête pour la modélisation.
Si les données TomTom ne contiennent pas de dimension temporelle, génère une série
réaliste basée sur les statistiques réelles + profil horaire typique de Bouznika.
"""

import numpy as np
import pandas as pd

from utils import get_logger


class TimeSeriesBuilder:
    """
    Construit une série temporelle de vitesse (km/h) à partir des données TomTom.

    Utilisation :
        builder = TimeSeriesBuilder(df_tomtom)
        series  = builder.build()
    """

    # Profil horaire normalisé (ratio vs vitesse libre) — calibré pour Bouznika/RN1
    _HOURLY_PROFILE = np.array([
        0.95, 0.97, 0.98, 0.99, 0.99, 0.98,   # 0h–5h  : nuit, trafic fluide
        0.90, 0.72, 0.60, 0.70, 0.80, 0.82,   # 6h–11h : pointe matin
        0.85, 0.87, 0.88, 0.85, 0.78, 0.62,   # 12h–17h: légère congestion
        0.58, 0.68, 0.78, 0.85, 0.90, 0.93,   # 18h–23h: pointe soir
    ])

    def __init__(self, df: pd.DataFrame, date_from: str = "2024-08-01", n_days: int = 31):
        self.df        = df
        self.date_from = date_from
        self.n_days    = n_days
        self.log       = get_logger(self.__class__.__name__)

    def build(self) -> pd.Series:
        """
        Retourne une pd.Series horaire (index DatetimeIndex) pour août 2024.
        Priorité : données TomTom réelles > simulation basée sur stats réelles.
        """
        speed_col = self._detect_speed_column()

        if speed_col and "hour" in self.df.columns:
            series = self._from_real_hourly(speed_col)
            self.log.info("Série construite depuis données TomTom réelles (heure × vitesse)")
        elif speed_col:
            series = self._from_stats(speed_col)
            self.log.info("Série simulée basée sur statistiques TomTom réelles + profil horaire")
        else:
            series = self._synthetic()
            self.log.warning("Aucune colonne vitesse détectée — série entièrement simulée")

        self.log.info(f"Série finale : {len(series)} obs | moy={series.mean():.1f} km/h | std={series.std():.1f}")
        return series

    # ── Méthodes privées ───────────────────────────────────────
    def _detect_speed_column(self) -> str | None:
        return next((c for c in self.df.columns if "speed" in c.lower()), None)

    def _from_real_hourly(self, speed_col: str) -> pd.Series:
        """Agrège les vitesses réelles par heure."""
        s = self.df.groupby("hour")[speed_col].mean().sort_index()
        # Répliquer sur n_days
        values = np.tile(s.values, self.n_days)
        idx    = pd.date_range(self.date_from, periods=self.n_days * 24, freq="h")
        return pd.Series(values, index=idx, name="speed_kmh").dropna()

    def _from_stats(self, speed_col: str) -> pd.Series:
        """Génère une série réaliste basée sur la moyenne/std TomTom réelles."""
        mean_spd = self.df[speed_col].mean()
        std_spd  = max(self.df[speed_col].std(), mean_spd * 0.05)
        np.random.seed(42)
        base    = mean_spd * self._HOURLY_PROFILE
        daily   = np.tile(base, self.n_days)
        noise   = np.random.normal(0, std_spd * 0.15, len(daily))
        values  = np.clip(daily + noise, 5, mean_spd * 1.3)
        idx     = pd.date_range(self.date_from, periods=len(values), freq="h")
        return pd.Series(values, index=idx, name="speed_kmh")

    def _synthetic(self) -> pd.Series:
        """Série 100 % simulée (profil Bouznika/RN1 moyen en km/h)."""
        np.random.seed(42)
        base_kmh = np.array([
            58, 60, 61, 62, 62, 60, 52, 38, 30, 36, 44, 46,
            48, 50, 50, 48, 42, 30, 27, 34, 42, 48, 52, 56,
        ], dtype=float)
        daily  = np.tile(base_kmh, self.n_days)
        noise  = np.random.normal(0, 3, len(daily))
        values = np.clip(daily + noise, 10, 80)
        idx    = pd.date_range(self.date_from, periods=len(values), freq="h")
        return pd.Series(values, index=idx, name="speed_kmh")
