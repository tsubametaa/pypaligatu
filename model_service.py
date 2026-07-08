import io, os
import torch
import torch.nn as nn
from torchvision import models, transforms
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image

app = FastAPI(title="Paligatu ML Inference API")

# Mapping dari train_dataset.class_to_idx: {'anorganik': 0, 'b3': 1, 'organik': 2}
LABELS = ["ANORGANIK", "B3", "ORGANIK"]

# Inisialisasi device (otomatis deteksi GPU CUDA atau fallback ke CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model():
    """
    Membangun ulang arsitektur EfficientNet-B0 + custom classifier
    persis seperti saat training di notebook Colab (Paligatu_AI.ipynb).
    """
    # Gunakan weights=None karena kita akan memuat bobot langsung dari best_model.pth
    m = models.efficientnet_b0(weights=None)
    
    # Mengganti classifier standar (1280 -> 1000) dengan custom classifier (1280 -> 256 -> 3)
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

# Preprocessing sesuaikan persis seperti transformasi validasi saat training
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
# ─────────────────────────────────────────────────────────

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
    Menerima upload gambar dan mengembalikan prediksi label sampah serta confidence score.
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
            "confidence": round(conf.item(), 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal memproses gambar: {str(e)}")
