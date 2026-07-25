import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style grafik
sns.set_theme(style="whitegrid")

def generate_eda():
    data_path = os.path.join("data", "used_cars.csv")
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    
    # 1. Hitung Umur Mobil (Tahun saat ini: 2026)
    df['age'] = 2026 - df['year']

    # --- Grafik 1: Distribusi Target (Price) ---
    plt.figure(figsize=(8, 5))
    sns.histplot(df['price'], kde=True, color='skyblue')
    plt.title('Grafik 1: Distribusi Harga Mobil Bekas (Target)')
    plt.xlabel('Harga (Price)')
    plt.ylabel('Frekuensi')
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, 'target_distribution.png'))
    plt.close()

    # --- Grafik 2: Jumlah Missing Values per Kolom ---
    plt.figure(figsize=(8, 5))
    missing = df.isna().sum()
    sns.barplot(x=missing.index, y=missing.values, palette='viridis')
    plt.title('Grafik 2: Jumlah Nilai Hilang per Kolom')
    plt.xlabel('Kolom')
    plt.ylabel('Jumlah Missing Values')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, 'missing_values.png'))
    plt.close()

    # --- Grafik 3: Non-Linear Hubungan Umur vs Harga ---
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=df['age'], y=df['price'], alpha=0.5, color='coral')
    plt.title('Grafik 3: Hubungan Non-Linear Umur Mobil vs Harga')
    plt.xlabel('Umur Mobil (Tahun)')
    plt.ylabel('Harga (Price)')
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, 'age_vs_price.png'))
    plt.close()

    # --- Grafik 4: Boxplot Harga berdasarkan Transmisi ---
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='transmission', y='price', data=df, palette='Set2')
    plt.title('Grafik 4: Variasi Harga Berdasarkan Tipe Transmisi')
    plt.xlabel('Tipe Transmisi')
    plt.ylabel('Harga (Price)')
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, 'transmission_price_boxplot.png'))
    plt.close()

    print("=" * 45)
    print("      EDA & PEMBUATAN GRAFIK BERHASIL      ")
    print("=" * 45)
    print("4 Grafik PNG telah disimpan di folder 'reports/':")
    print(" 1. target_distribution.png")
    print(" 2. missing_values.png")
    print(" 3. age_vs_price.png")
    print(" 4. transmission_price_boxplot.png")
    print("=" * 45)

if __name__ == "__main__":
    generate_eda()
