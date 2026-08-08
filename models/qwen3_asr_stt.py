import io

import soundfile as sf
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

from models.base import STTModel


class Qwen3ASRSTT(STTModel):
    model_id = "Qwen/Qwen3-ASR-1.7B-hf"

    def __init__(self):
        print(f"Loading STT model {self.model_id}...")
        self._device = torch.device("cpu")
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForMultimodalLM.from_pretrained(
            self.model_id,
            device_map=str(self._device),
            dtype=torch.float32,
        )
        self._model.eval()
        print("Qwen3-ASR STT model ready.")

    def transcribe(self, audio_bytes: bytes, language: str = "en") -> str:
        audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes))

        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)
        if sample_rate != 16000:
            import resampy

            audio_array = resampy.resample(audio_array, sample_rate, 16000)

        inputs = self._processor.apply_transcription_request(
            audio=audio_array,
            language=language or None,
        ).to(self._device, self._model.dtype)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs, max_new_tokens=256, do_sample=False
            )

        num_input_tokens = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[:, num_input_tokens:]
        transcription = self._processor.decode(
            generated_ids, return_format="transcription_only"
        )
        return transcription[0].strip()
