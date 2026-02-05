import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model

processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base"
)
model = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-base"
)

model.eval()

def extract_embedding(audio_array):
    inputs = processor(
        audio_array,
        sampling_rate=16000,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)

    return embedding
