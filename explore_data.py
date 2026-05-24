import pandas as pd

file_path = "tomtom_data_daily/RP3323_part1.xlsx"

try:
    xl = pd.ExcelFile(file_path)
    sheet_name = "Route1-Speeds(harmonic avg)"
    df = xl.parse(sheet_name, header=None)
    with open("info.txt", "w", encoding="utf-8") as f:
        # Print first 6 rows, first 10 columns
        f.write(df.iloc[:6, :10].to_string())
except Exception as e:
    with open("info.txt", "w", encoding="utf-8") as f:
        f.write(f"Error: {e}")
