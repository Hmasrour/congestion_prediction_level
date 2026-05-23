"""
results/exporter.py
===================
Classe ResultsExporter — Besoin 3
Exporte les résultats comparatifs en CSV et génère les figures.

Fichiers produits :
    results/csv/comparaison_modeles.csv   — métriques MAE/RMSE/MAPE/R²
    results/csv/predictions_vs_reel.csv   — prédictions heure par heure
    results/csv/stats_par_point.csv       — stats vitesse par point GPS
    results/figures/resultats_comparatifs.png — 4 graphiques
"""

import os
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

import config.settings as cfg
from utils import ensure_dirs, get_logger


class ResultsExporter:
    """
    Consolide et exporte les résultats des 3 modèles prédictifs.

    Utilisation :
        exporter = ResultsExporter(model_results, train, test, df_traffic)
        exporter.run()
    """

    def __init__(
        self,
        model_results: List[Dict[str, Any]],
        train: pd.Series,
        test: pd.Series,
        df_traffic: pd.DataFrame,
    ):
        self.model_results = model_results   # Liste de dicts {name, predictions, actual, metrics}
        self.train         = train
        self.test          = test
        self.df_traffic    = df_traffic
        self.log           = get_logger(self.__class__.__name__)
        ensure_dirs(cfg.RESULTS_CSV_DIR, cfg.RESULTS_FIG_DIR)

    # ── API publique ───────────────────────────────────────────
    def run(self) -> None:
        """Lance tous les exports."""
        self.log.info("=== BESOIN 3 — RÉSULTATS & EXPORT CSV ===")
        df_metrics = self._export_metrics()
        self._export_predictions()
        self._export_stats_by_point()
        self._plot_results(df_metrics)
        self._print_summary(df_metrics)

    # ── Exports CSV ────────────────────────────────────────────
    def _export_metrics(self) -> pd.DataFrame:
        """Tableau comparatif des métriques de performance."""
        rows = [{"Modèle": r["name"], **r["metrics"]} for r in self.model_results]
        df   = pd.DataFrame(rows).set_index("Modèle")
        path = os.path.join(cfg.RESULTS_CSV_DIR, "comparaison_modeles.csv")
        df.to_csv(path)
        self.log.info(f"Métriques exportées → {path}")
        return df

    def _export_predictions(self) -> None:
        """Prédictions de chaque modèle vs valeurs réelles, heure par heure."""
        n = min(len(self.test), *(len(r["predictions"]) for r in self.model_results))
        data = {"timestamp": self.test.index[:n], "actual_speed_kmh": self.test.values[:n]}
        for r in self.model_results:
            col = "pred_" + r["name"].split()[0]
            data[col] = r["predictions"][:n]
        df   = pd.DataFrame(data)
        path = os.path.join(cfg.RESULTS_CSV_DIR, "predictions_vs_reel.csv")
        df.to_csv(path, index=False)
        self.log.info(f"Prédictions exportées → {path}")

    def _export_stats_by_point(self) -> None:
        """Statistiques de vitesse par point GPS (si disponibles)."""
        speed_col = next((c for c in self.df_traffic.columns if "speed" in c.lower()), None)
        if not speed_col or "road_point" not in self.df_traffic.columns:
            self.log.warning("Stats par point non exportées : colonnes manquantes")
            return
        df   = self.df_traffic.groupby("road_point")[speed_col].agg(
            Moyenne="mean", Minimum="min", Maximum="max", Ecart_type="std"
        ).round(2)
        path = os.path.join(cfg.RESULTS_CSV_DIR, "stats_par_point.csv")
        df.to_csv(path)
        self.log.info(f"Stats par point exportées → {path}")

    # ── Visualisations ─────────────────────────────────────────
    def _plot_results(self, df_metrics: pd.DataFrame) -> None:
        """Génère 4 graphiques dans une figure 2×2."""
        fig = plt.figure(figsize=(18, 11))
        fig.suptitle(
            "Bouznika — Analyse Prédictive du Trafic (Août 2024)\n"
            "Master Smart Cities UM6P — PFE 2025",
            fontsize=13, fontweight="bold", y=0.98
        )
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])

        self._plot_predictions(ax1)
        self._plot_rmse_bars(ax2, df_metrics)
        self._plot_hourly_profile(ax3)
        self._plot_metrics_table(ax4, df_metrics)

        path = os.path.join(cfg.RESULTS_FIG_DIR, "resultats_comparatifs.png")
        plt.savefig(path, dpi=cfg.FIGURE_DPI, bbox_inches="tight", facecolor="white")
        plt.close()
        self.log.info(f"Figure exportée → {path}")

    def _plot_predictions(self, ax: plt.Axes) -> None:
        """Graphique 1 : prédictions vs valeurs réelles."""
        n = min(len(self.test), 72)   # Afficher 72h max pour la lisibilité
        x = range(n)
        ax.plot(x, self.test.values[:n], "k-", linewidth=1.8, label="Réel", alpha=0.85, zorder=5)
        for r in self.model_results:
            color = cfg.MODEL_COLORS.get(r["name"].split()[0], "#9E9E9E")
            ax.plot(x, r["predictions"][:n], "--", color=color,
                    linewidth=1.3, label=r["name"], alpha=0.85)
        ax.set_title("Prédictions vs Valeurs Réelles (72h test)", fontweight="bold")
        ax.set_xlabel("Heures")
        ax.set_ylabel("Vitesse (km/h)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    def _plot_rmse_bars(self, ax: plt.Axes, df_metrics: pd.DataFrame) -> None:
        """Graphique 2 : RMSE par modèle."""
        models    = df_metrics.index.tolist()
        rmse_vals = df_metrics["RMSE (km/h)"].values
        colors    = [cfg.MODEL_COLORS.get(m.split()[0], "#9E9E9E") for m in models]
        bars      = ax.bar(models, rmse_vals, color=colors, width=0.5, edgecolor="white", linewidth=1.5)
        for bar, val in zip(bars, rmse_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_title("RMSE par modèle (plus bas = meilleur)", fontweight="bold")
        ax.set_ylabel("RMSE (km/h)")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m.split()[0] for m in models])
        ax.grid(True, alpha=0.25, axis="y")

    def _plot_hourly_profile(self, ax: plt.Axes) -> None:
        """Graphique 3 : profil horaire moyen sur août 2024."""
        full = pd.concat([self.train, self.test])
        if hasattr(full.index, "hour"):
            hourly = full.groupby(full.index.hour).mean()
            ax.fill_between(hourly.index, hourly.values, alpha=0.2, color="#2196F3")
            ax.plot(hourly.index, hourly.values, "o-", color="#2196F3", linewidth=2, markersize=4)
            ax.set_xticks(range(0, 24, 2))
            ax.set_xlabel("Heure de la journée")
        else:
            ax.plot(full.values[:168], color="#2196F3", linewidth=1.5)
            ax.set_xlabel("Observations")
        ax.axvspan(7, 9,   alpha=0.15, color="red",    label="Pointe matin (7h–9h)")
        ax.axvspan(17, 19, alpha=0.15, color="orange", label="Pointe soir (17h–19h)")
        ax.set_title("Profil horaire moyen de vitesse — Août 2024", fontweight="bold")
        ax.set_ylabel("Vitesse moyenne (km/h)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    def _plot_metrics_table(self, ax: plt.Axes, df_metrics: pd.DataFrame) -> None:
        """Graphique 4 : tableau récapitulatif des métriques."""
        ax.axis("off")
        best  = df_metrics["RMSE (km/h)"].idxmin()
        table = ax.table(
            cellText=df_metrics.reset_index().values,
            colLabels=["Modèle"] + df_metrics.columns.tolist(),
            cellLoc="center", loc="center",
            bbox=[0.02, 0.15, 0.96, 0.72],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9.5)
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#1565C0")
                cell.set_text_props(color="white", fontweight="bold")
            elif row % 2 == 0:
                cell.set_facecolor("#E3F2FD")
            cell.set_edgecolor("#BDBDBD")
        ax.set_title(
            f"Résumé des performances\n🏆 Meilleur modèle (RMSE) : {best.split()[0]}",
            fontweight="bold", pad=14
        )

    # ── Résumé console ─────────────────────────────────────────
    def _print_summary(self, df_metrics: pd.DataFrame) -> None:
        """Affiche le tableau des métriques et la liste des fichiers produits."""
        self.log.info("\n" + "=" * 58)
        self.log.info("  TABLEAU COMPARATIF DES MODÈLES")
        self.log.info("=" * 58)
        print(df_metrics.to_string())
        best = df_metrics["RMSE (km/h)"].idxmin()
        self.log.info(f"\n🏆 Meilleur modèle : {best}")
        self.log.info("\n  FICHIERS PRODUITS")
        self.log.info("=" * 58)
        files = [
            (cfg.RESULTS_CSV_DIR + "/comparaison_modeles.csv",   "Métriques MAE/RMSE/MAPE/R²"),
            (cfg.RESULTS_CSV_DIR + "/predictions_vs_reel.csv",   "Prédictions heure par heure"),
            (cfg.RESULTS_CSV_DIR + "/stats_par_point.csv",       "Vitesse par point GPS"),
            (cfg.RESULTS_FIG_DIR + "/resultats_comparatifs.png", "4 graphiques comparatifs"),
            (cfg.DATA_PROCESSED_DIR + "/tomtom_filtered.csv",    "Données TomTom filtrées"),
        ]
        for path, desc in files:
            self.log.info(f"  📄 {path:<48} {desc}")
