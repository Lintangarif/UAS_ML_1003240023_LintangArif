import os
import pandas as pd

def load_data():
    data_path = os.path.join("data", "used_cars.csv")
    
    if not os.path.exists(data_path):
        print(f"[ERROR] File '{data_path}' tidak ditemukan!")
        return None

    df = pd.read_csv(data_path)
    
    print("=" * 45)
    print("       PEMUATAN DATASET BERHASIL      ")
    print("=" * 45)
    print(f"Jumlah Baris : {df.shape[0]}")
    print(f"Jumlah Kolom : {df.shape[1]}")
    print("\n--- Tipe Data Kolom ---")
    print(df.dtypes)
    print("\n--- Jumlah Nilai Hilang (Missing Values) ---")
    print(df.isna().sum())
    print("=" * 45)
    
    return df

if __name__ == "__main__":
    load_data()
