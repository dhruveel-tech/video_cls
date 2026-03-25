import os
import random
import shutil

# ----------------------------
# CONFIG
# ----------------------------
SOURCE_DIR = r"D:\Vertex_AI\dataset"   # your current dataset
OUTPUT_DIR = r"D:\Vertex_AI\dataset_split"  # new structured dataset
TRAIN_RATIO = 0.7

random.seed(42)

# ----------------------------
# CREATE SPLIT
# ----------------------------

def split_dataset():

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for label in os.listdir(SOURCE_DIR):

        label_path = os.path.join(SOURCE_DIR, label)

        if not os.path.isdir(label_path):
            continue

        videos = [f for f in os.listdir(label_path) if f.endswith(".mp4")]
        random.shuffle(videos)

        split_index = int(len(videos) * TRAIN_RATIO)

        train_videos = videos[:split_index]
        val_videos = videos[split_index:]

        # Create directories
        train_label_dir = os.path.join(OUTPUT_DIR, "train", label)
        val_label_dir = os.path.join(OUTPUT_DIR, "val", label)

        os.makedirs(train_label_dir, exist_ok=True)
        os.makedirs(val_label_dir, exist_ok=True)

        # Copy files
        for video in train_videos:
            shutil.copy(
                os.path.join(label_path, video),
                os.path.join(train_label_dir, video)
            )

        for video in val_videos:
            shutil.copy(
                os.path.join(label_path, video),
                os.path.join(val_label_dir, video)
            )

        print(f"{label}: {len(train_videos)} train, {len(val_videos)} val")

    print("\nDataset split complete!")


if __name__ == "__main__":
    split_dataset()