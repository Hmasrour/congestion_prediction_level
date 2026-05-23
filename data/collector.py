"""
data/collector.py
=================
Classe DataCollector — Besoin 1
Collecte les données historiques de trafic via TomTom Traffic Stats (Move).
Pipeline : soumettre job → attendre → télécharger → filtrer → sauvegarder.
"""

import gzip
import json
import os
import time
from typing import Dict, Tuple

import pandas as pd
import requests
from shapely.geometry import Point, mapping
from shapely.ops import unary_union

import config.settings as cfg
from utils import ensure_dirs, get_logger, nearest_road_point


class DataCollector:
    """
    Collecte les données historiques TomTom Move pour les points GPS définis.

    Utilisation :
        collector = DataCollector()
        df = collector.run()
    """

    def __init__(
        self,
        api_key: str = cfg.TOMTOM_API_KEY,
        road_points: Dict[str, Tuple[float, float]] = cfg.ROAD_POINTS,
        radius_deg: float = cfg.RADIUS_DEG,
        filter_radius_m: float = cfg.FILTER_RADIUS_M,
        poll_interval: int = cfg.POLL_INTERVAL_S,
    ):
        self.api_key         = api_key
        self.road_points     = road_points
        self.radius_deg      = radius_deg
        self.filter_radius_m = filter_radius_m
        self.poll_interval   = poll_interval
        self.log             = get_logger(self.__class__.__name__)

        ensure_dirs(cfg.DATA_RAW_DIR, cfg.DATA_PROCESSED_DIR)

        self._raw_path      = os.path.join(cfg.DATA_RAW_DIR, "tomtom_raw.csv")
        self._filtered_path = os.path.join(cfg.DATA_PROCESSED_DIR, "tomtom_filtered.csv")

    # ── API publique ───────────────────────────────────────────
    def run(self) -> pd.DataFrame:
        """
        Lance le pipeline complet.
        Si les données filtrées existent déjà, les charge directement.
        """
        if os.path.exists(self._filtered_path):
            self.log.info(f"Données déjà présentes — chargement depuis {self._filtered_path}")
            return pd.read_csv(self._filtered_path)

        self.log.info("=== BESOIN 1 — COLLECTE DONNÉES TOMTOM MOVE (Août 2024) ===")
        geometry = self._build_geometry()
        job_id   = self._submit_job(geometry)
        job_data = self._wait_for_job(job_id)
        df_raw   = self._download_and_parse(job_data)
        df_raw.to_csv(self._raw_path, index=False)
        self.log.info(f"{len(df_raw)} segments bruts sauvegardés → {self._raw_path}")

        df = self._filter_near_points(df_raw)
        df.to_csv(self._filtered_path, index=False)
        self.log.info(f"{len(df)} segments filtrés sauvegardés → {self._filtered_path}")
        return df

    # ── Méthodes privées ───────────────────────────────────────
    def _build_geometry(self) -> dict:
        """
        Construit un GeoJSON valide autour des points GPS.
        Utilise unary_union pour fusionner les cercles qui se chevauchent
        (les 4 points sont ~100m les uns des autres → overlap inévitable).
        """
        buffers = [
            Point(lon, lat).buffer(self.radius_deg)
            for _, (lat, lon) in self.road_points.items()
        ]
        merged = unary_union(buffers).buffer(0)
        geo = mapping(merged)
        if geo["type"] == "Polygon":
            geo = {"type": "MultiPolygon", "coordinates": [geo["coordinates"]]}
        self.log.info(
            f"Géométrie construite : {len(self.road_points)} points fusionnés, "
            f"rayon ~{int(self.radius_deg * 111_000)}m, type={geo['type']}"
        )
        return geo

    def _submit_job(self, geometry: dict) -> str:
        """
        Soumet le job Area Analysis à TomTom Traffic Stats.
        Si un job identique existe déjà (erreur 400 + jobId dans la réponse),
        réutilise le jobId existant au lieu d'échouer.
        """
        url = f"{cfg.TOMTOM_BASE_URL}/areaanalysis/1?key={self.api_key}"
        payload = {
            "jobName": "Bouznika_PFE_Aout2024",
            "distanceUnit": "KILOMETERS",
            "network": {
                "name": "Bouznika_Roads",
                "geometry": geometry,
                "timeZoneId": cfg.TOMTOM_TIMEZONE,
                "frcs": cfg.TOMTOM_FRC_CLASSES,
                "probeSource": "ALL",
            },
            "dateRange": {
                "name": "Aout_2024",
                "from": cfg.TOMTOM_DATE_FROM,
                "to": cfg.TOMTOM_DATE_TO,
            },
            "timeSets": [
                {
                    "name": "Toute_la_journee",
                    "timeGroups": [{
                        "days": ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"],
                        "times": ["0:00-24:00"],
                    }],
                }
            ],
            "acceptMode": "AUTO",
        }
        r    = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
        body = r.json()

        if r.status_code in (200, 201, 202):
            job_id = body.get("jobId")
            self.log.info(f"Job soumis — jobId: {job_id}")
            return job_id

        # TomTom renvoie 400 + jobId si le même job existe déjà → réutilisation
        existing_job_id = body.get("jobId")
        if r.status_code == 400 and existing_job_id:
            self.log.info(f"Job déjà existant — réutilisation du jobId: {existing_job_id}")
            return existing_job_id

        raise RuntimeError(f"Erreur soumission job: {r.status_code} — {r.text}")

    def _wait_for_job(self, job_id: str) -> dict:
        """
        Interroge TomTom toutes les N secondes jusqu'à la fin du job.
        Endpoint officiel : GET /traffic/trafficstats/status/1/{job_id}
        Champ d'état : jobState (NEW → … → DONE).
        """
        url = f"{cfg.TOMTOM_BASE_URL}/status/1/{job_id}?key={self.api_key}"
        accept_url = f"{cfg.TOMTOM_BASE_URL}/status/1/{job_id}/accept?key={self.api_key}"
        self.log.info("En attente des résultats TomTom...")

        DONE_STATUSES   = {"DONE"}
        FAILED_STATUSES = {"FAILED", "CANCELLED", "ERROR", "REJECTED", "EXPIRED"}

        while True:
            r = requests.get(url)
            data = r.json()

            if data.get("responseStatus") == "ERROR":
                msgs = data.get("messages", [])
                raise RuntimeError(
                    f"Erreur API TomTom (statut job) : {msgs or data}"
                )

            status = (
                data.get("jobState")
                or data.get("state")
                or data.get("status")
                or data.get("jobStatus")
                or "UNKNOWN"
            ).upper()

            self.log.info(f"Statut job : {status}")

            if status in DONE_STATUSES:
                return data
            if status in FAILED_STATUSES:
                raise RuntimeError(f"Job TomTom échoué : {status} | {data}")
            if status == "NEED_CONFIRMATION":
                requests.post(accept_url)
                self.log.info("Job accepté automatiquement (mode NEED_CONFIRMATION)")

            time.sleep(self.poll_interval)

    def _download_and_parse(self, job_data: dict) -> pd.DataFrame:
        """Télécharge et parse le fichier JSON/GeoJSON compressé."""
        dl_url = None
        urls = job_data.get("urls", [])
        for candidate in urls:
            path = candidate.split("?")[0].lower()
            if path.endswith(".json"):
                dl_url = candidate
                break
        if not dl_url:
            for candidate in urls:
                path = candidate.split("?")[0].lower()
                if path.endswith(".geojson"):
                    dl_url = candidate
                    break
        if not dl_url:
            legacy = job_data.get("downloadUrls", {})
            dl_url = legacy.get("json") or legacy.get("geojson")
        if not dl_url:
            for r in job_data.get("results", []):
                if "url" in r:
                    dl_url = r["url"]
                    break
        if not dl_url:
            raise RuntimeError("Aucune URL de téléchargement dans la réponse TomTom.")

        self.log.info("Téléchargement des données...")
        r = requests.get(dl_url)
        raw = r.content
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        data = json.loads(raw)

        if "features" in data:
            return pd.DataFrame([f.get("properties", {}) for f in data["features"]])
        if "network" in data or "segmentResults" in data:
            return self._parse_area_analysis_json(data)
        return pd.json_normalize(data)

    @staticmethod
    def _parse_area_analysis_json(data: dict) -> pd.DataFrame:
        """Parse le JSON Area Analysis (network.segmentResults + shape)."""
        network = data.get("network") or {}
        segments = network.get("segmentResults") or data.get("segmentResults") or []
        rows = []
        for seg in segments:
            shape = seg.get("shape") or []
            lat, lon = None, None
            if shape:
                lats, lons = [], []
                for pt in shape:
                    if isinstance(pt, dict):
                        lats.append(pt.get("latitude", pt.get("lat")))
                        lons.append(pt.get("longitude", pt.get("lon", pt.get("lng"))))
                    elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
                        lons.append(pt[0])
                        lats.append(pt[1])
                if lats and lons:
                    lat = sum(lats) / len(lats)
                    lon = sum(lons) / len(lons)

            for tr in seg.get("segmentTimeResults") or [{}]:
                rows.append({
                    "segmentId": seg.get("segmentId"),
                    "newSegmentId": seg.get("newSegmentId"),
                    "streetName": seg.get("streetName"),
                    "frc": seg.get("frc"),
                    "distance_m": seg.get("distance"),
                    "speedLimit": seg.get("speedLimit"),
                    "latitude": lat,
                    "longitude": lon,
                    "harmonicAverageSpeed": tr.get("harmonicAverageSpeed"),
                    "medianSpeed": tr.get("medianSpeed"),
                    "averageSpeed": tr.get("averageSpeed"),
                    "sampleSize": tr.get("sampleSize"),
                    "dateRangeId": tr.get("dateRange"),
                    "timeSetId": tr.get("timeSet"),
                })
        return pd.DataFrame(rows)

    def _filter_near_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filtre les segments dans le rayon autour des points d'intérêt."""
        lat_col = next(
            (c for c in df.columns if c.lower() in ("latitude", "lat")),
            next((c for c in df.columns if "lat" in c.lower()), None),
        )
        lon_col = next(
            (c for c in df.columns if c.lower() in ("longitude", "lon", "lng")),
            next((c for c in df.columns if "lon" in c.lower() or "lng" in c.lower()), None),
        )
        if not lat_col or not lon_col:
            self.log.warning("Colonnes lat/lon non détectées — retour du DataFrame complet")
            return df

        rows = []
        for _, row in df.iterrows():
            slat, slon = row.get(lat_col), row.get(lon_col)
            if pd.isna(slat) or pd.isna(slon):
                continue
            road_id, dist = nearest_road_point(slat, slon, self.road_points, self.filter_radius_m)
            if road_id:
                r = row.to_dict()
                r["road_point"] = road_id
                r["distance_m"] = round(dist, 1)
                rows.append(r)

        result = pd.DataFrame(rows)
        self.log.info(f"{len(result)} segments retenus (rayon {self.filter_radius_m}m)")
        return result