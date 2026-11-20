import io

import soundfile as sf
from transformers import pipeline

from models.base import STTModel


class FluidSpeechSTT(STTModel):
    model_id = "FluidVoice-Labs/fluid-speech-pro"

    def __init__(self):
        print(f"Loading STT model {self.model_id}...")
        self._asr = pipeline(
            "automatic-speech-recognition",
            model=self.model_id,
        )
        print("Fluid Speech STT model ready.")

    def transcribe(self, audio_bytes: bytes, language: str = "en") -> str:
        audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")

        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)
        if sample_rate != 16000:
            import resampy

            audio_array = resampy.resample(audio_array, sample_rate, 16000)

        result = self._asr({"array": audio_array, "sampling_rate": 16000})
        if isinstance(result, list):
            result = result[0]
        return result["text"].strip()
