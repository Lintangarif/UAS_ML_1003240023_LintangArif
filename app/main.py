import os
import joblib
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from enum import Enum

model_pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_pipeline
    model_path = os.path.join("models", "model.joblib")
    if os.path.exists(model_path):
        model_pipeline = joblib.load(model_path)
        print("✅ Model berhasil dimuat ke memori API.")
    else:
        print("❌ Model tidak ditemukan!")
    yield
    model_pipeline = None

app = FastAPI(
    title="API Estimasi Harga Kendaraan Bekas",
    description="REST API Machine Learning End-to-End",
    version="1.0.0",
    lifespan=lifespan
)

class TransmissionEnum(str, Enum):
    Automatic = "Automatic"
    Manual = "Manual"
    Semi_Auto = "Semi-Auto"

class FuelTypeEnum(str, Enum):
    Petrol = "Petrol"
    Diesel = "Diesel"
    Hybrid = "Hybrid"
    Other = "Other"

class CarInput(BaseModel):
    year: int = Field(..., ge=1990, le=2026, description="Tahun pembuatan mobil")
    transmission: TransmissionEnum
    mileage: int = Field(..., ge=0, description="Jarak tempuh mil")
    fuelType: FuelTypeEnum
    tax: int = Field(..., ge=0, description="Pajak")
    mpg: float = Field(..., gt=0, description="Efisiensi BBM")
    engineSize: float = Field(..., ge=0.0, description="Ukuran mesin")

@app.get("/")
def read_root():
    return {
        "service": "API Estimasi Harga Mobil Bekas",
        "status": "online"
    }

@app.get("/health")
def health_check():
    if model_pipeline is None:
        raise HTTPException(status_code=500, detail="Model belum termuat di server")
    return {
        "status": "healthy",
        "model_loaded": True
    }

@app.post("/predict-harga")
def predict_harga(car: CarInput):
    if model_pipeline is None:
        raise HTTPException(status_code=500, detail="Model belum termuat di server")
    
    age = 2026 - car.year
    
    input_data = pd.DataFrame([{
        'transmission': car.transmission.value,
        'mileage': car.mileage,
        'fuelType': car.fuelType.value,
        'tax': car.tax,
        'mpg': car.mpg,
        'engineSize': car.engineSize,
        'age': age
    }])
    
    predicted_price = model_pipeline.predict(input_data)[0]
    
    return {
        "status": "success",
        "predicted_price": round(float(predicted_price), 2),
        "currency": "GBP"
    }
