import os
import json
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model():
    print("Mulai Evaluasi Model Final...")
    X_test = pd.read_csv("data/X_test.csv")
    y_test = pd.read_csv("data/y_test.csv")

    # 1. TEST SET DISENTUH SEKALI & LOAD PIPELINE UTUH
    pipeline = joblib.load("models/model.joblib")
    y_pred = pipeline.predict(X_test)

    # 2. HITUNG METRIK
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    print(f"\nHasil Evaluasi Test Set:")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R2   : {r2:.4f}")

    # 3. METADATA & JUSTIFIKASI METRIK
    metadata = {
        "metrics": {
            "MAE": mae,
            "RMSE": rmse,
            "R2_score": r2
        },
        "model_type": "RandomForestRegressor (dalam Pipeline)",
        "justifikasi_metrik": "Metrik MAE (Mean Absolute Error) dipilih sebagai metrik utama karena paling mudah dipahami secara bisnis. MAE langsung merepresentasikan rata-rata model meleset sekian Poundsterling dalam menebak harga wajar mobil bekas."
    }
    
    with open("models/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    print("\nMetadata berhasil disimpan ke models/metadata.json")

if __name__ == "__main__":
    evaluate_model()
