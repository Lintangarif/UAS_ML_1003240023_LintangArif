import pytest
from fastapi.testclient import TestClient
from app.main import app

# Fixture agar lifespan (pemuatan model) dipicu dengan benar oleh TestClient
@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

# --- 4 TEST MEKANIS ---
def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True

def test_predict_success(client):
    payload = {
        "year": 2020,
        "transmission": "Manual",
        "mileage": 15000,
        "fuelType": "Petrol",
        "tax": 145,
        "mpg": 55.4,
        "engineSize": 1.0
    }
    response = client.post("/predict-harga", json=payload)
    assert response.status_code == 200
    assert "predicted_price" in response.json()

def test_predict_missing_field(client):
    payload = {
        "year": 2020,
        "transmission": "Manual"
    }
    response = client.post("/predict-harga", json=payload)
    assert response.status_code == 422

def test_predict_invalid_enum(client):
    payload = {
        "year": 2020,
        "transmission": "Terbang",
        "mileage": 15000,
        "fuelType": "Petrol",
        "tax": 145,
        "mpg": 55.4,
        "engineSize": 1.0
    }
    response = client.post("/predict-harga", json=payload)
    assert response.status_code == 422


# --- 2 BEHAVIORAL TEST (SYARAT UAS) ---
def test_behavioral_older_car_cheaper(client):
    car_old = {
        "year": 2012, "transmission": "Manual", "mileage": 30000,
        "fuelType": "Petrol", "tax": 145, "mpg": 50.0, "engineSize": 1.2
    }
    car_new = {
        "year": 2022, "transmission": "Manual", "mileage": 30000,
        "fuelType": "Petrol", "tax": 145, "mpg": 50.0, "engineSize": 1.2
    }
    
    res_old = client.post("/predict-harga", json=car_old).json()["predicted_price"]
    res_new = client.post("/predict-harga", json=car_new).json()["predicted_price"]
    
    assert res_old < res_new

def test_behavioral_high_mileage_cheaper(client):
    car_high_mileage = {
        "year": 2020, "transmission": "Manual", "mileage": 100000,
        "fuelType": "Petrol", "tax": 145, "mpg": 50.0, "engineSize": 1.2
    }
    car_low_mileage = {
        "year": 2020, "transmission": "Manual", "mileage": 10000,
        "fuelType": "Petrol", "tax": 145, "mpg": 50.0, "engineSize": 1.2
    }
    
    res_high = client.post("/predict-harga", json=car_high_mileage).json()["predicted_price"]
    res_low = client.post("/predict-harga", json=car_low_mileage).json()["predicted_price"]
    
    assert res_high < res_low
