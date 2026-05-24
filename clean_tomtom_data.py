import pandas as pd
import os
import re
import sys

# Ensure stdout uses utf-8
sys.stdout.reconfigure(encoding='utf-8')

os.makedirs('data_cleaned', exist_ok=True)

def process_file(filepath):
    print(f"Processing {filepath}...")
    df = pd.read_excel(filepath, sheet_name='Route1-Speeds(harmonic avg)', header=None)
    
    dates = df.iloc[3].values
    hours = df.iloc[4].values
    
    records = []
    
    for col_idx in range(len(dates)):
        date_val = str(dates[col_idx])
        hour_val = str(hours[col_idx])
        
        if 'Aug_' in date_val and 'Hour_' in hour_val:
            # Parse Date
            day = date_val.split('_')[1]
            date_str = f"2024-08-{day}"
            
            # Parse Hour
            match = re.search(r'Hour_(\d{2})', hour_val)
            if match:
                hour_str = f"{match.group(1)}:00"
            else:
                continue
                
            # Parse data from row 5 onwards for this column
            col_data = pd.to_numeric(df.iloc[5:, col_idx], errors='coerce')
            
            # TomTom often uses 0 for missing data/no probes, we'll replace 0 with NaN 
            # to avoid skewing the mean down to 0 artificially
            col_data = col_data.replace(0, pd.NA)
            
            avg_speed = col_data.mean(skipna=True)
            
            records.append({
                'date': date_str,
                'hour': hour_str,
                'average_speed_kph': round(avg_speed, 2) if pd.notna(avg_speed) else None
            })
            
    return records

if __name__ == "__main__":
    records_part1 = process_file('tomtom_data_daily/RP3323_part1.xlsx')
    records_part2 = process_file('tomtom_data_daily/RP3323_part2.xlsx')

    all_records = records_part1 + records_part2

    final_df = pd.DataFrame(all_records)
    final_df = final_df.sort_values(by=['date', 'hour']).reset_index(drop=True)

    final_path = 'data_cleaned/RP3323_August2024_hourly.csv'
    final_df.to_csv(final_path, index=False)
    
    print(f"\nData successfully cleaned and saved to {final_path}")
    print(f"Total rows: {len(final_df)}")
    print("\nSample (first 5 rows):")
    print(final_df.head())
