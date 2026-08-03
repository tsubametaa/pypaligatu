# Paligatu AI — Platform Klasifikasi Sampah Cerdas

Platform klasifikasi sampah otomatis berbasis Deep Learning (EfficientNet-B0) yang mengidentifikasi kategori sampah **ORGANIK**, **ANORGANIK**, dan **B3**.

---

## 📁 Struktur Direktori

```
ml-service/
├── backend/          # Service inferensi Machine Learning (FastAPI + PyTorch)
│   ├── model_service.py
│   ├── best_model.pth
│   ├── Dockerfile
│   └── requirements.txt
└── frontend/         # Web Application Interface (Astro 7 + Tailwind v4 + Lucide)
    ├── src/
    ├── package.json
    └── astro.config.mjs
```

---

## 🚀 Panduan Penggunaan

### 1. Menjalankan Backend (FastAPI)
```bash
cd backend
pip install -r requirements.txt
uvicorn model_service:app --reload --port 8000
```

### 2. Menjalankan Frontend (Astro)
```bash
cd frontend
bun install
bun run dev
```

Buka `http://localhost:4321` di browser Anda.

---

© SMKN 31 Jakarta
