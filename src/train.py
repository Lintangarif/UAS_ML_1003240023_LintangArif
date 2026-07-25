import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

def train_model():
    print("Mulai Tahap 3: Training Model...")
    df = pd.read_csv("data/used_cars.csv")
    
    # Rekayasa Fitur Dasar & Pembersihan Outlier
    df['age'] = 2026 - df['year']
    df = df[df['price'] < 50000] # Buang outlier ekstrem agar model akurat
    
    # Pisahkan Fitur (X) dan Target (y)
    # Kolom 'model' dibuang karena terlalu banyak variasi unik, 'year' diganti 'age'
    X = df.drop(columns=['price', 'model', 'year'])
    y = df['price']

    # 1. SPLIT SEBELUM PREPROCESSING (Syarat Wajib UAS)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Simpan test set untuk digunakan di evaluate.py nanti
    X_test.to_csv("data/X_test.csv", index=False)
    y_test.to_csv("data/y_test.csv", index=False)

    # 2. BUNGKUS PREPROCESSING DI DALAM COLUMN TRANSFORMER
    numeric_features = ['mileage', 'tax', 'mpg', 'engineSize', 'age']
    categorical_features = ['transmission', 'fuelType']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # 3. BANDINGKAN 3 ALGORITMA DENGAN 5-FOLD CV
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42)
    }

    print("\n--- Perbandingan 3 Model (5-Fold CV - Metrik MAE) ---")
    best_model_name = None
    best_score = float('inf')
    
    for name, algo in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', algo)])
        # CV pakai neg_mean_absolute_error karena scikit-learn memaksimalkan skor (semakin mendekati 0 semakin baik)
        scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
        mae_score = -scores.mean()
        print(f"{name}: MAE = {mae_score:.2f} GBP")
        
        if mae_score < best_score:
            best_score = mae_score
            best_model_name = name

    print(f"\nModel Terbaik: {best_model_name}")

    # 4. SIMPAN PIPELINE UTUH (Syarat Wajib UAS)
    print("Melatih model final dan menyimpan ke models/model.joblib...")
    final_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', models[best_model_name])])
    final_pipeline.fit(X_train, y_train)

    joblib.dump(final_pipeline, "models/model.joblib")
    print("Training Selesai!")

if __name__ == "__main__":
    train_model()
