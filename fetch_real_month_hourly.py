import os
import time
import requests
import pandas as pd
from datetime import datetime
import config.settings as cfg
from data.collector import DataCollector

class FullMonthCollector(DataCollector):
    def run_full_month(self):
        geometry = self._build_geometry()
        dfs = []
        
        days = list(range(1, 32))
        batch_size = 5
        
        for i in range(0, len(days), batch_size):
            batch_days = days[i:i+batch_size]
            job_ids = {}
            
            self.log.info(f"--- Lancement du batch de jours : {batch_days} ---")
            
            for day in batch_days:
                date_str = f"2024-08-{day:02d}"
                while True:
                    try:
                        self.log.info(f"Soumission pour {date_str}...")
                        job_id = self._submit_daily_job(geometry, date_str)
                        job_ids[date_str] = job_id
                        break
                    except RuntimeError as e:
                        if "limit[20] exceed" in str(e) or "limit[10] exceed" in str(e):
                            self.log.warning("Limite de jobs atteinte, attente 30s avant de ressayer...")
                            time.sleep(30)
                        else:
                            raise e
            
            for date_str, job_id in job_ids.items():
                self.log.info(f"Attente du job {job_id} pour {date_str}...")
                job_data = self._wait_for_job(job_id)
                df_day = self._download_and_parse(job_data)
                df_day['date'] = date_str
                dfs.append(df_day)
                
        df_raw = pd.concat(dfs, ignore_index=True)
        df_raw.to_csv("data/raw/tomtom_raw_744h.csv", index=False)
        
        df_filtered = self._filter_near_points(df_raw)
        
        df_grouped = df_filtered.groupby(['date', 'road_point', 'timeSetId'])['averageSpeed'].mean().reset_index()
        unique_timesets = sorted(df_grouped['timeSetId'].unique())
        time_map = {ts: f"{i:02d}:00:00" for i, ts in enumerate(unique_timesets)}
        df_grouped['hour_str'] = df_grouped['timeSetId'].map(time_map)
        df_grouped['timestamp'] = pd.to_datetime(df_grouped['date'] + " " + df_grouped['hour_str'])
        
        df_pivot = df_grouped.pivot(index='timestamp', columns='road_point', values='averageSpeed').round(2)
        df_pivot.to_csv("data/processed/donnees_reelles_744h_aout2024.csv")
        
        self.log.info("Srie complte 744h relle exporte avec succs dans data/processed/donnees_reelles_744h_aout2024.csv")
        return df_pivot

    def _submit_daily_job(self, geometry, date_str):
        url = f"{cfg.TOMTOM_BASE_URL}/areaanalysis/1?key={self.api_key}"
        payload = {
            "jobName": f"Bouznika_PFE_{date_str}",
            "distanceUnit": "KILOMETERS",
            "network": {
                "name": "Bouznika_Roads",
                "geometry": geometry,
                "timeZoneId": cfg.TOMTOM_TIMEZONE,
                "frcs": cfg.TOMTOM_FRC_CLASSES,
                "probeSource": "ALL",
            },
            "dateRange": {
                "name": date_str,
                "from": date_str,
                "to": date_str,
            },
            "timeSets": [
                {
                    "name": f"{h:02d}h-{h+1:02d}h",
                    "timeGroups": [{
                        "days": ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"],
                        "times": [f"{h}:00-{h+1 if h < 23 else 24}:00"],
                    }],
                } for h in range(24)
            ],
            "acceptMode": "AUTO",
        }
        r = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
        body = r.json()
        if r.status_code in (200, 201, 202):
            return body.get("jobId")
        existing_job_id = body.get("jobId")
        if r.status_code == 400 and existing_job_id:
            return existing_job_id
        raise RuntimeError(f"Erreur soumission job {date_str}: {r.status_code}  {r.text}")

if __name__ == "__main__":
    c = FullMonthCollector()
    df = c.run_full_month()
    print("Termin ! 31 jours traits.")
