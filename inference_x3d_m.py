import json
import torch
import torch.nn as nn
from torchvision.transforms import Compose, CenterCrop
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    UniformTemporalSubsample,
    Normalize,
    RandomShortSideScale,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.models.hub import x3d_m   # ← X3D Medium


# ======================
# CONFIG
# ======================

MODEL_PATH    = r"D:\Vertex_AI\Model\x3d_m_best.pth"
VIDEO_PATH    = r"D:\Vertex_AI\dataset\backhand_good\vid_20.mp4"
CLASS_MAP_PATH = r"class_names.json"    # generated automatically by training script

NUM_CLASSES   = 3
CLIP_DURATION = 2.0     # must match training  (X3D-M uses 2 s)
NUM_FRAMES    = 16      # must match training
SPATIAL_SIZE  = 224     # must match training

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ======================
# LOAD CLASS NAMES
# (from the JSON saved during training — guaranteed correct order)
# ======================

with open(CLASS_MAP_PATH) as f:
    label_map = json.load(f)

CLASS_NAMES = [label_map[str(i)] for i in range(NUM_CLASSES)]
print("Class mapping:", CLASS_NAMES)


# ======================
# LOAD MODEL — X3D Medium
# ======================

model = x3d_m(pretrained=False)
model.blocks[-1].proj = nn.Linear(
    model.blocks[-1].proj.in_features,
    NUM_CLASSES
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

print("✅ Model loaded successfully")


# ======================
# TRANSFORM
# (Must be identical to the val_transform used in training)
# ======================

transform = ApplyTransformToKey(
    key="video",
    transform=Compose([
        UniformTemporalSubsample(NUM_FRAMES),
        RandomShortSideScale(min_size=SPATIAL_SIZE, max_size=SPATIAL_SIZE),
        CenterCrop(SPATIAL_SIZE),       # ✅ CenterCrop for inference (deterministic)
        Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
    ]),
)


# ======================
# LOAD & PREPROCESS VIDEO
# ======================

video = EncodedVideo.from_path(VIDEO_PATH)
clip  = video.get_clip(start_sec=0, end_sec=CLIP_DURATION)
clip  = transform(clip)

inputs = clip["video"].unsqueeze(0).to(DEVICE)   # add batch dimension


# ======================
# INFERENCE
# ======================

with torch.no_grad():
    outputs       = model(inputs)
    probabilities = torch.softmax(outputs, dim=1)

confidence, predicted_idx = torch.max(probabilities, 1)
predicted_label = CLASS_NAMES[predicted_idx.item()]

# Print all class probabilities for transparency
print("\n📊 All class probabilities:")
for i, name in enumerate(CLASS_NAMES):
    print(f"   {name:20s}: {probabilities[0][i].item()*100:.2f}%")

print("\n🎯 Prediction Results:")
print(f"   Predicted Class : {predicted_label}")
print(f"   Confidence      : {round(confidence.item() * 100, 2)} %")