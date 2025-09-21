from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

# --- 1. Setup device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --- 2. Load model + processor ---
model_name = "prithivMLmods/Gym-Workout-Classifier-SigLIP2"
processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForImageClassification.from_pretrained(model_name, trust_remote_code=True).to(device)
model.eval()

# --- 3. Load your test image ---
img = Image.open(r"C:\Users\NEIL\Downloads\VS CODE\algo proj python\plank_test.jpg")

# --- 4. Run inference ---
inputs = processor(images=img, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

pred_label = model.config.id2label[outputs.logits.argmax(-1).item()]
print("Predicted label:", pred_label)
