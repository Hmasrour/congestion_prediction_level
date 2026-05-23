import pandas as pd
from data import TimeSeriesBuilder

def main():
    df_traffic = pd.read_csv('data/processed/tomtom_filtered.csv')
    routes = df_traffic['road_point'].dropna().unique()
    series_dict = {}

    for route in routes:
        df_route = df_traffic[df_traffic['road_point'] == route].copy()
        builder = TimeSeriesBuilder(df_route, date_from='2024-08-01', n_days=31)
        series_dict[route] = builder.build()

    df_routes = pd.DataFrame(series_dict)
    df_routes.index.name = 'timestamp'
    df_routes.to_csv('data/processed/donnees_horaires_par_route_aout2024.csv')
    print("Données par route exportées avec succès.")

if __name__ == '__main__':
    main()
