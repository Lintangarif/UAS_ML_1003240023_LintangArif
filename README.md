# UAS Machine Learning End-to-End: Estimasi Harga Mobil Bekas Toyota

> **Mata Kuliah:** Machine Learning End-to-End  
> **Nama:** Lintang Arif Setianda  
> **NIM:** 1003240023  
> **Kasus:** Regresi - Estimasi Harga Kendaraan Bekas Toyota  
> **Dataset:** 100k UK Used Car Dataset (Toyota) — *Kaggle (CC0: Public Domain)*  

---

## Bagian Penilaian (Quick Links)

* **[Link Video Demo API & PY ] https://youtu.be/hZIMhp2oUDM*
* **[Laporan Lengkap PDF](https://drive.google.com/file/d/18roLUTnA7nlsc0NyfNmZeMYCL_Yde384/view?usp=drive_link)*


---

## Struktur Folder Proyek

```text
UAS_ML_1003240023_LintangArif/
├── app/                  # REST API Service (FastAPI & Pydantic)
│   └── main.py
├── models/               # Artefak Model ML (.gitignore)
│   └── model.joblib
├── reports/              # Visualisasi Grafik EDA & PDF Laporan
├── src/                  # Pipeline Training, Data Loading & EDA
├── tests/                # Automated Testing (Pytest)
│   └── test_api.py
├── .gitignore            # Konfigurasi Filter Repository
├── README.md             # Dokumentasi Utama
├── requirements.txt      # Dependensi Lingkungan Training
└── requirements-api.txt  # Dependensi Lingkungan Serving (Pinned)
```
## Spesifikasi Lingkungan Development

Sistem ini dikembangkan dan diuji menggunakan konfigurasi lingkungan *virtual environment* dengan spesifikasi modul utama sebagai berikut:

| Modul / Library | Versi   | Peran / Kegunaan                     |
| :---            | :---    | :---                                 |
| **Python**      | 3.12.10 | Bahasa Pemrograman Utama             |
| **scikit-learn**| 1.9.0   | Machine Learning Pipeline & Modeling |
| **pandas**      | 3.0.5   | Manipulasi & Analisis Data           |
| **fastapi**     | 0.140.0 | Framework REST API Serving           |
| **uvicorn**     | 0.51.0  | ASGI Web Server Runtime              |
| **pydantic**    | 2.13.4  | Validasi Skema Input/Output API      |
| **pytest**      | 9.1.1   | Automated Testing Suite              |

## Cara Menjalankan Proyek dari Nol

### 1. Setup Virtual Environment
```bash
python -m venv venv

# Aktivasi Virtual Environment (Windows PowerShell):
.\venv\Scripts\Activate.ps1

# Aktivasi Virtual Environment (Linux/Mac):
source venv/bin/activate
```

### 2. Instal Dependensi
```bash
pip install -r requirements.txt
pip install -r requirements-api.txt
```

### 3. Eksekusi REST API Server
```bash
uvicorn app.main:app --reload
```
Akses dokumentasi interaktif Swagger UI di: `http://127.0.0.1:8000/docs`

### 4. Menjalankan Automated Testing (Pytest)
```bash
python -m pytest tests/ -v
```

---

## Perbedaan Lingkungan Serving (`requirements-api.txt`) & Training (`requirements.txt`)

Di dalam repositori ini, dependensi dipisahkan menjadi dua berkas untuk menerapkan prinsip best practices dari MLOps:

### Mengapa Versi pada Lingkungan Serving Wajib Di-pin Persis (`==`)?
1. **Stabilitas dan Determinisme di Production:** Layanan REST API membutuhkan tingkat keandalan yang konsisten. Mengunci versi secara eksplisit (`==`) mencegah terjadinya kegagalan sistem akibat *breaking changes* dari pembaruan modul pihak ketiga.
2. **Kompatibilitas Deserialisasi Model (`joblib` / `scikit-learn`):** Artefak model `.joblib` sangat sensitif terhadap versi `scikit-learn` dan `numpy`. Perbedaan versi pada lingkungan *serving* dapat menyebabkan `joblib.load()` gagal atau menghasilkan prediksi yang tidak valid.
3. **Efisiensi Minimal Footprint:** Lingkungan *serving* hanya membutuhkan dependensi *runtime* esensial. Library analisis data (seperti `matplotlib` dan `seaborn`) dieliminasi dari `requirements-api.txt` untuk mempercepat *startup time* dan menghemat alokasi memori server.

### Mengapa Lingkungan Training Tidak Wajib Di-pin Secara Ketat?
1. **Fleksibilitas Eksperimen:** Pelatihan model dan Exploratory Data Analysis (EDA) bersifat *offline* dan iteratif. Pengembang membutuhkan fleksibilitas untuk menguji modul baru dan optimasi performa.
2. **Isolasi Risiko:** Penyesuaian dependensi pada lingkungan *training* dapat diuji secara lokal tanpa mengganggu kestabilan API yang berstatus *production-ready*.

---

## Contoh Request dan Response API

### 1. Request Valid (Status 200 OK)
**Endpoint:** `POST /predict`
```json
{
  "year": 2020,
  "transmission": "Manual",
  "mileage": 15000,
  "fuelType": "Petrol",
  "tax": 145,
  "mpg": 55.4,
  "engineSize": 1.0
}
```

**Response (200 OK):**
```json
{
  "status": "success",
  "predicted_price": 12450.5,
  "currency": "GBP"
}
```

### 2. Request Tidak Valid (Status 422 Unprocessable Entity)
**Endpoint:** `POST /predict` *(Field `mileage` dihilangkan untuk simulasi error)*
```json
{
  "year": 2020,
  "transmission": "Manual"
}
```

**Response (422 Unprocessable Entity):** Memicu *validation error* otomatis dari Pydantic dengan rincian field yang hilang.
```json
{
  "detail": [
    {
      "type": "missing",
      "loc": [
        "body",
        "mileage"
      ],
      "msg": "Field required",
      "input": {
        "year": 2020,
        "transmission": "Manual"
      }
    }
  ]
}
```

---

## Penutup dan Kesimpulan

Proyek ini dibangun sebagai bentuk pemenuhan tugas Ujian Akhir Semester (UAS) secara komprehensif. Melalui repositori ini, telah diimplementasikan siklus lengkap (*full lifecycle*) dari sebuah model Machine Learning: 
Mulai dari tahap pengumpulan dan pembersihan data mentah, Exploratory Data Analysis (EDA), perancangan dan pelatihan model, hingga tahap akhir *deployment* menjadi sebuah layanan REST API yang stabil, teruji secara otomatis (dengan Pytest), dan siap dikonsumsi oleh aplikasi berskala produksi.

**Disusun Oleh:**  
**Lintang Arif Setianda (1003240023)**  
Tahun Ajaran 2026/2027  