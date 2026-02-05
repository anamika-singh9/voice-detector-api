import base64
import io
import numpy as np
from pydub import AudioSegment

def base64_to_wav(base64_audio):
    audio_bytes = base64.b64decode(base64_audio)

    audio = AudioSegment.from_file(
        io.BytesIO(audio_bytes),
        format="mp3"
    )

    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)

    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    samples /= np.max(np.abs(samples)) + 1e-9

    return samples
