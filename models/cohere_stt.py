import io

import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, CohereAsrForConditionalGeneration

from models.base import STTModel

MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"


class CohereSTT(STTModel):
    def __init__(self):
        print(f"Loading STT model {MODEL_ID}...")
        self._processor = AutoProcessor.from_pretrained(MODEL_ID)
        self._model = CohereAsrForConditionalGeneration.from_pretrained(
            MODEL_ID, device_map="auto"
        )
        self._model.eval()
        print("STT model ready.")

    @property
    def model_id(self) -> str:
        return MODEL_ID

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
