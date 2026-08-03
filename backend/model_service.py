import io, os
import torch
import torch.nn as nn
from torchvision import models, transforms
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

app = FastAPI(title="Paligatu ML Inference API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

LABELS = ["ANORGANIK", "B3", "ORGANIK"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cpu":
    torch.set_num_threads(1)

def build_model():
    """
    Membangun ulang arsitektur EfficientNet-B0 + custom classifier
    persis seperti saat training di notebook Colab (Paligatu_AI.ipynb).
    """
    m = models.efficientnet_b0(weights=None)

    in_f = m.classifier[1].in_features  # 1280
    m.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_f, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, len(LABELS)),
    )
    return m.to(device)

model = build_model()
MODEL_PATH = os.environ.get("MODEL_PATH", "best_model.pth")
model_loaded = False

try:
    if os.path.exists(MODEL_PATH):
        # Memuat state_dict dari file best_model.pth
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        model_loaded = True
        print(f"✅ Model EfficientNet-B0 berhasil dimuat dari {MODEL_PATH} ke {device}")
    else:
        print(f"⚠️ Peringatan: File model {MODEL_PATH} tidak ditemukan. Endpoint /predict akan mengembalikan error.")
except Exception as e:
    print(f"❌ Gagal memuat model: {str(e)}")

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

@app.get("/")
async def root():
    """Root endpoint — supaya base URL tidak 404 dan bisa dipakai cek cepat."""
    return {
        "service": "Paligatu ML Inference API",
        "docs": "/docs",
        "health": "/health",
    }

@app.get("/health")
async def health_check():
    """Endpoint untuk health check Caddy/Docker/Spring Boot"""
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "device": str(device),
        "labels": LABELS
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Menerima upload gambar dan mengembalikan prediksi label sampah,
    confidence score, serta probabilitas semua kelas.
    """
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model belum dimuat. Periksa keberadaan file best_model.pth.")

    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
        x = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)[0]

        conf, idx = torch.max(probs, dim=0)

        return {
            "label": LABELS[idx.item()],
            "confidence": round(conf.item(), 4),
            "probabilities": {LABELS[i]: round(p.item(), 4) for i, p in enumerate(probs)}
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal memproses gambar: {str(e)}")