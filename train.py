import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pytorchvideo.models.hub import x3d_xs
import os
import cv2
import numpy as np

from torchvision.transforms import Compose, Lambda, RandomCrop, RandomHorizontalFlip, CenterCrop
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    ShortSideScale,
)

# =========================
# CONFIG
# =========================
DATA_PATH_TRAIN = r"D:\Vertex_AI\dataset_split\train"
DATA_PATH_VAL = r"D:\Vertex_AI\dataset_split\val"

BATCH_SIZE = 1
EPOCHS = 10
LR = 0.0005
NUM_FRAMES = 16

DEVICE = torch.device("cpu")

# -----------------------------
# LABEL MAPPING
# -----------------------------
LABELS = {
    "forehand_good": 0,
    "backhand_good": 1,
    "serve_good": 2,
}
NUM_CLASSES = len(LABELS)

# -----------------------------
# TRANSFORMS
# -----------------------------
train_transform = Compose([
    ApplyTransformToKey(
        key="video",
        transform=Compose([
            Lambda(lambda x: x / 255.0),
            ShortSideScale(size=180),
            RandomCrop(160),
            RandomHorizontalFlip(p=0.5),
            Normalize(
                mean=[0.45, 0.45, 0.45],
                std=[0.225, 0.225, 0.225],
            ),
        ]),
    ),
])

val_transform = Compose([
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

# -----------------------------
# DATASET
# -----------------------------
class TennisStrokeDataset(Dataset):
    def __init__(self, root_dir, transform=None, num_frames=8):
        self.root_dir = root_dir
        self.transform = transform
        self.num_frames = num_frames
        self.samples = []

        for label_name in os.listdir(root_dir):
            label_path = os.path.join(root_dir, label_name)
            if not os.path.isdir(label_path):
                continue
            if label_name not in LABELS:
                continue

            for file in os.listdir(label_path):
                if file.endswith(".mp4"):
                    self.samples.append((
                        os.path.join(label_path, file),
                        LABELS[label_name]
                    ))

    def __len__(self):
        return len(self.samples)

    def _load_video(self, path):
        cap = cv2.VideoCapture(path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(
            0, total_frames - 1, self.num_frames
        ).astype(int)

        frames = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))  # Early resize (memory safe)
            frames.append(frame)

        cap.release()

        frames = np.stack(frames)
        frames = torch.from_numpy(frames).float()
        frames = frames.permute(3, 0, 1, 2)  # (C, T, H, W)

        return frames

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        video = self._load_video(video_path)

        sample = {"video": video}

        if self.transform:
            sample = self.transform(sample)

        return sample["video"], torch.tensor(label, dtype=torch.long)


# =========================
# DATA LOADERS
# =========================
train_dataset = TennisStrokeDataset(DATA_PATH_TRAIN, transform=train_transform, num_frames=NUM_FRAMES)
val_dataset = TennisStrokeDataset(DATA_PATH_VAL, transform=val_transform, num_frames=NUM_FRAMES)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print("Number of training samples:", len(train_dataset))
print("Number of classes:", NUM_CLASSES)

# =========================
# MODEL
# =========================
print("Loading pretrained X3D-XS...")
model = x3d_xs(pretrained=True)

# Freeze backbone
for param in model.blocks[-2].parameters():
    param.requires_grad = True

# Replace classifier
in_features = model.blocks[-1].proj.in_features
model.blocks[-1].proj = nn.Linear(in_features, NUM_CLASSES)

# Proper initialization
nn.init.xavier_uniform_(model.blocks[-1].proj.weight)
nn.init.zeros_(model.blocks[-1].proj.bias)

# Only classifier trains
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.0001
)

model = model.to(DEVICE)

# =========================
# TRAIN SETUP
# =========================
criterion = nn.CrossEntropyLoss()
best_acc = 0.0

# =========================
# TRAIN LOOP
# =========================
for epoch in range(EPOCHS):

    print(f"\nEpoch [{epoch+1}/{EPOCHS}]")

    # ---- TRAIN ----
    model.train()
    correct = 0
    total = 0
    running_loss = 0.0

    for videos, labels in train_loader:
        videos = videos.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(videos)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * videos.size(0)

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / total
    train_acc = 100 * correct / total

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_acc:.2f}%")

    # ---- VALIDATION ----
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for videos, labels in val_loader:
            videos = videos.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(videos)
            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = 100 * val_correct / val_total
    print(f"Validation Accuracy: {val_acc:.2f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_x3d_xs.pth")
        print("Best model saved!")

print("\nTraining Complete")
print("Best Validation Accuracy:", best_acc)





