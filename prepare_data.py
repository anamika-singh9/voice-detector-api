import os
import random
import shutil

SOURCE = "raw_dataset"
TARGET = "data"
TRAIN_RATIO = 0.8
LABELS = ["human", "ai"]

def split(label):
    files = [
        f for f in os.listdir(os.path.join(SOURCE, label))
        if f.endswith(".wav")
    ]

    random.shuffle(files)
    split_idx = int(len(files) * TRAIN_RATIO)

    train_files = files[:split_idx]
    val_files = files[split_idx:]

    for f in train_files:
        shutil.copy(
            os.path.join(SOURCE, label, f),
            os.path.join(TARGET, "train", label, f)
        )

    for f in val_files:
        shutil.copy(
            os.path.join(SOURCE, label, f),
            os.path.join(TARGET, "val", label, f)
        )

    print(f"{label}: {len(train_files)} train / {len(val_files)} val")

os.makedirs("data/train/human", exist_ok=True)
os.makedirs("data/train/ai", exist_ok=True)
os.makedirs("data/val/human", exist_ok=True)
os.makedirs("data/val/ai", exist_ok=True)

for label in LABELS:
    split(label)

print("Dataset prepared")
