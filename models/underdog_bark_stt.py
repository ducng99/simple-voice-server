import io

import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, ParakeetForTDT

from models.base import STTModel


class UnderdogBarkSTT(STTModel):
    model_id = "ConwayResearch/Underdog-Bark-3.0-Pro"

    def __init__(self):
        print(f"Loading STT model {self.model_id}...")
        self._device = torch.device("cpu")
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = ParakeetForTDT.from_pretrained(
            self.model_id,
            device_map=str(self._device),
            dtype=torch.float32,
        )
        self._model.eval()
        print("Underdog-Bark STT model ready.")

    def transcribe(self, audio_bytes: bytes, language: str = "en") -> str:
        audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")

        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)
        if sample_rate != 16000:
            import resampy

            audio_array = resampy.resample(audio_array, sample_rate, 16000)

        # Leading silence improves transcription of the first words.
        audio_array = np.concatenate([np.zeros(4000, np.float32), audio_array])

        inputs = self._processor(
            audio_array, sampling_rate=16000, return_tensors="pt"
        ).to(self._device, self._model.dtype)

        with torch.inference_mode():
            output = self._model.generate(**inputs)
        token_ids = output.sequences if hasattr(output, "sequences") else output
        transcription = self._processor.batch_decode(
            token_ids, skip_special_tokens=True
        )
        return transcription[0].strip()
