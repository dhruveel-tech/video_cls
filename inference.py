import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision.transforms import Compose, Lambda, CenterCrop
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    UniformTemporalSubsample,
    Normalize,
    ShortSideScale,
)
from pytorchvideo.models.hub import x3d_xs


# ----------------------------
# CONFIG
# ----------------------------

MODEL_PATH = r"D:\Vertex_AI\video_engine\best_x3d_xs.pth"
DEVICE = torch.device("cpu")

LABELS = {
    0: "forehand_good",
    1: "backhand_good",
    2: "serve_good",
}

NUM_CLASSES = 3


# ----------------------------
# TRANSFORM (same as validation)
# ----------------------------

transform = Compose([
    ApplyTransformToKey(
        key="video",
        transform=Compose([
            Lambda(lambda x: x / 255.0),
            ShortSideScale(size=180),
            CenterCrop(160),
            Normalize(
                mean=[0.45, 0.45, 0.45],
                std=[0.225, 0.225, 0.225],
            ),
        ]),
    ),
])


# ----------------------------
# LOAD MODEL
# ----------------------------

model = x3d_xs(pretrained=False)

in_features = model.blocks[-1].proj.in_features
model.blocks[-1].proj = nn.Linear(in_features, NUM_CLASSES)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

print("Model loaded successfully.")


# ----------------------------
# VIDEO LOADER
# ----------------------------
NUM_FRAMES = 16

def load_video(path):
    cap = cv2.VideoCapture(path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(
        0, total_frames - 1, NUM_FRAMES
    ).astype(int)

    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        frames.append(frame)

    cap.release()

    frames = np.stack(frames)
    frames = torch.from_numpy(frames).float()
    frames = frames.permute(3, 0, 1, 2)

    return frames


# ----------------------------
# PREDICTION FUNCTION
# ----------------------------

def predict(video_path):

    video = load_video(video_path)

    sample = {"video": video}
    sample = transform(sample)

    video_tensor = sample["video"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(video_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    class_name = LABELS[predicted.item()]
    confidence = confidence.item() * 100

    print(f"Prediction: {class_name}")
    print(f"Confidence: {confidence:.2f}%")

    return class_name, confidence


# ----------------------------
# TEST
# ----------------------------

if __name__ == "__main__":

    test_video = r"D:\Vertex_AI\dataset_split\train\backhand_good\vid_27.mp4"
    predict(test_video)