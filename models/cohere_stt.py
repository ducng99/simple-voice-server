import io

import soundfile as sf
import torch
from transformers import AutoProcessor, CohereAsrForConditionalGeneration

from models.base import STTModel


class CohereSTT(STTModel):
    model_id = "CohereLabs/cohere-transcribe-03-2026"

    def __init__(self):
        print(f"Loading STT model {self.model_id}...")
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = CohereAsrForConditionalGeneration.from_pretrained(
            self.model_id, device_map="auto"
        )
        self._model.eval()
        print("Cohere STT model ready.")

    def transcribe(self, audio_bytes: bytes, language: str = "en") -> str:
        audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes))

        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)
        if sample_rate != 16000:
            import resampy

            audio_array = resampy.resample(audio_array, sample_rate, 16000)

        inputs = self._processor(
            audio_array, sampling_rate=16000, return_tensors="pt", language=language
        )
        inputs = inputs.to(self._model.device, dtype=self._model.dtype)

        with torch.no_grad():
            outputs = self._model.generate(**inputs, max_new_tokens=256)

        text = self._processor.decode(outputs, skip_special_tokens=True)
        if isinstance(text, list):
            text = " ".join(text).strip()
        return text
