import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision.transforms import (
    Compose,
    RandomHorizontalFlip,
    RandomCrop,
    CenterCrop,
)

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    UniformTemporalSubsample,
    Normalize,
    RandomShortSideScale,
)

from pytorchvideo.data import (
    LabeledVideoDataset,
    make_clip_sampler,
)
from pytorchvideo.data.labeled_video_paths import LabeledVideoPaths
from pytorchvideo.models.hub import x3d_m   # ← X3D Medium


# ======================
# CONFIG
# ======================

DATA_PATH_TRAIN = r"D:\Vertex_AI\dataset_split\train"
DATA_PATH_VAL   = r"D:\Vertex_AI\dataset_split\val"

NUM_CLASSES    = 3
BATCH_SIZE     = 1          # Keep 1 — X3D-M is heavier than X3D-S
NUM_EPOCHS     = 20
CLIP_DURATION  = 2.0        # X3D-M uses longer clips (2 s vs 1 s for S)
LEARNING_RATE  = 1e-4

# X3D-M specific parameters
NUM_FRAMES     = 16         # frames sampled per clip  (paper: 16)
SPATIAL_SIZE   = 224        # spatial resolution       (paper: 224 vs 160 for S)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ======================
# MAIN FUNCTION (Required for Windows)
# ======================

def main():

    print("Using device:", DEVICE)

    # -------------------------------------------------------
    # Save class label mapping (alphabetical — same order
    # that LabeledVideoPaths assigns integer labels)
    # -------------------------------------------------------
    labeled_paths = LabeledVideoPaths.from_directory(DATA_PATH_TRAIN)

    label_map = {}
    for path, label in labeled_paths._paths_and_labels:
        folder = os.path.basename(os.path.dirname(path))
        label_map[str(label)] = folder

    label_map = dict(sorted(label_map.items(), key=lambda x: int(x[0])))
    with open("class_names.json", "w") as f:
        json.dump(label_map, f, indent=2)

    print("Label mapping:", label_map)

    # ======================
    # TRANSFORMS
    # ======================

    train_transform = ApplyTransformToKey(
        key="video",
        transform=Compose([
            UniformTemporalSubsample(NUM_FRAMES),
            RandomShortSideScale(min_size=SPATIAL_SIZE, max_size=SPATIAL_SIZE),
            RandomHorizontalFlip(p=0.5),        # augmentation for training
            RandomCrop(SPATIAL_SIZE),
            Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
        ]),
    )

    val_transform = ApplyTransformToKey(
        key="video",
        transform=Compose([
            UniformTemporalSubsample(NUM_FRAMES),
            RandomShortSideScale(min_size=SPATIAL_SIZE, max_size=SPATIAL_SIZE),
            CenterCrop(SPATIAL_SIZE),           # deterministic for validation
            Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
        ]),
    )

    # ======================
    # DATASET
    # ======================

    train_dataset = LabeledVideoDataset(
        LabeledVideoPaths.from_directory(DATA_PATH_TRAIN),
        clip_sampler=make_clip_sampler("random", CLIP_DURATION),
        transform=train_transform,
        decode_audio=False,
    )

    val_dataset = LabeledVideoDataset(
        LabeledVideoPaths.from_directory(DATA_PATH_VAL),
        clip_sampler=make_clip_sampler("uniform", CLIP_DURATION),
        transform=val_transform,
        decode_audio=False,
    )

    # ⚠️ IMPORTANT: On Windows use num_workers=0
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

    # ======================
    # MODEL — X3D Medium
    # ======================

    model = x3d_m(pretrained=True)

    # Replace final projection layer for our number of classes
    model.blocks[-1].proj = nn.Linear(
        model.blocks[-1].proj.in_features,
        NUM_CLASSES
    )

    model = model.to(DEVICE)

    # Freeze backbone — only train the new head first
    for name, param in model.named_parameters():
        if "proj" not in name:
            param.requires_grad = False

    # ======================
    # LOSS & OPTIMIZER
    # ======================

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )

    # ======================
    # TRAIN LOOP
    # ======================

    def train_one_epoch():
        model.train()
        total_loss = 0
        correct    = 0
        total      = 0
        batch_count = 0

        for batch in train_loader:
            videos = batch["video"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(videos)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss  += loss.item()
            _, predicted = outputs.max(1)
            total       += labels.size(0)
            correct     += predicted.eq(labels).sum().item()
            batch_count += 1

        return total_loss / batch_count, 100.0 * correct / total

    def validate():
        model.eval()
        total_loss = 0
        correct    = 0
        total      = 0
        batch_count = 0

        with torch.no_grad():
            for batch in val_loader:
                videos = batch["video"].to(DEVICE)
                labels = batch["label"].to(DEVICE)

                outputs = model(videos)
                loss    = criterion(outputs, labels)

                total_loss  += loss.item()
                _, predicted = outputs.max(1)
                total       += labels.size(0)
                correct     += predicted.eq(labels).sum().item()
                batch_count += 1

        return total_loss / batch_count, 100.0 * correct / total

    # ======================
    # EPOCH LOOP
    # ======================

    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):

        # 🔥 Unfreeze backbone after 5 epochs for full fine-tuning
        if epoch == 5:
            print("\n🔥 Unfreezing backbone for fine-tuning...\n")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=1e-5)

        train_loss, train_acc = train_one_epoch()
        val_loss,   val_acc   = validate()

        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")
        print("-" * 40)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "x3d_m_best.pth")
            print("✅ Best model saved!")

    torch.save(model.state_dict(), "x3d_m_final.pth")
    print(f"\n🎉 Training complete. Best Val Acc: {best_val_acc:.2f}%")


# ======================
# WINDOWS SAFE ENTRY
# ======================

if __name__ == "__main__":
    main()