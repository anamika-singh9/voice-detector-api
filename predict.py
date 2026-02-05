import torch
import sys
import librosa
from model import extract_embedding

classifier = torch.nn.Linear(768, 2)
classifier.load_state_dict(
    torch.load("saved_models/classifier.pt", map_location="cpu")
)
classifier.eval()

def predict(audio_array):
    emb = extract_embedding(audio_array)

    with torch.no_grad():
        probs = torch.softmax(classifier(emb), dim=1)
        conf, label = torch.max(probs, dim=1)

    return {
        "label": "AI_GENERATED" if label.item() == 1 else "HUMAN",
        "confidence": round(conf.item(), 3)
    }


if __name__ == "__main__":
    audio_path = sys.argv[1]

    # load audio
    audio_array, sr = librosa.load(audio_path, sr=16000)

    result = predict(audio_array)

    print("Prediction:", result["label"])
    print("Confidence:", result["confidence"])