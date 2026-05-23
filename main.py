"""
main.py
=======
Point d'entrée — Bouznika Traffic Intelligence
Orchestre les 3 besoins du PFE :
    1. Collecte données TomTom Move (Août 2024)
    2. Modélisation prédictive (ARIMA · Random Forest · LSTM)
    3. Export résultats comparatifs (CSV + figures)

Lancement :
    python main.py                  # pipeline complet
    python main.py --collect-only   # CSV TomTom uniquement
"""

import argparse

from data import DataCollector, TimeSeriesBuilder
from models import ARIMAModel, RandomForestModel, LSTMModel, BaseModel
from results import ResultsExporter
from utils import get_logger

log = get_logger("main")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bouznika Traffic Intelligence")
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Collecte TomTom → CSV (data/raw + data/processed) sans modélisation",
    )
    args = parser.parse_args()

    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║  BOUZNIKA TRAFFIC INTELLIGENCE — PFE Smart Cities   ║")
    log.info("║  Master Exécutif UM6P — Août 2024                   ║")
    log.info("╚══════════════════════════════════════════════════════╝")

    # ── BESOIN 1 : Collecte ────────────────────────────────────
    collector  = DataCollector()
    df_traffic = collector.run()

    if args.collect_only:
        log.info(f"✅ Collecte terminée — {len(df_traffic)} lignes exportées en CSV")
        return

    # ── BESOIN 2 : Modélisation ────────────────────────────────
    log.info("=== BESOIN 2 — MODÉLISATION PRÉDICTIVE ===")

    series = TimeSeriesBuilder(df_traffic).build()
    train, test = BaseModel.train_test_split(series)
    log.info(f"Train : {len(train)} obs | Test : {len(test)} obs")

    models = [ARIMAModel(), RandomForestModel(), LSTMModel()]
    results = []
    for model in models:
        model.fit(train)
        results.append(model.evaluate(test))

    # ── BESOIN 3 : Export ──────────────────────────────────────
    exporter = ResultsExporter(results, train, test, df_traffic)
    exporter.run()

    log.info("✅ PIPELINE COMPLET TERMINÉ")


if __name__ == "__main__":
    main()
