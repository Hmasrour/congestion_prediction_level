import pandas as pd

def main():
    df = pd.read_csv('data/processed/tomtom_filtered.csv')
    df_grouped = df.groupby(['road_point', 'timeSetId'])['averageSpeed'].mean().reset_index()

    unique_timesets = sorted(df_grouped['timeSetId'].unique())
    time_map = {ts: f"{i:02d}:00" for i, ts in enumerate(unique_timesets)}
    df_grouped['hour'] = df_grouped['timeSetId'].map(time_map)

    df_pivot = df_grouped.pivot(index='hour', columns='road_point', values='averageSpeed').round(2)
    df_pivot.to_csv('data/processed/profil_reel_24h_aout2024.csv')
    print("Profil réel 24h exporté !")

if __name__ == '__main__':
    main()
