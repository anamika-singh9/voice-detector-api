import os
import torch
import librosa
from model import extract_embedding

DATA_DIR = "data"
SAVE_DIR = "embeddings"

LABELS = {"human": 0, "ai": 1}

os.makedirs(SAVE_DIR, exist_ok=True)

for split in ["train", "val"]:
    embeddings = []
    labels = []

    for label_name, label in LABELS.items():
        folder = os.path.join(DATA_DIR, split, label_name)

        for f in os.listdir(folder):
            if f.endswith(".wav"):
                path = os.path.join(folder, f)
                audio, _ = librosa.load(path, sr=16000)

                emb = extract_embedding(audio)
                embeddings.append(emb.squeeze(0))
                labels.append(label)

    torch.save(
        {
            "embeddings": torch.stack(embeddings),
            "labels": torch.tensor(labels)
        },
        f"{SAVE_DIR}/{split}.pt"
    )

    print(f"Saved {split} embeddings")
