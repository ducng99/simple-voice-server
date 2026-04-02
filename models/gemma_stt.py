import io

import soundfile as sf
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

from models.base import STTModel

MODEL_ID = "google/gemma-4-E2B-it"


class GemmaSTT(STTModel):
    def __init__(self):
        print(f"Loading STT model {MODEL_ID}...")
        self._processor = AutoProcessor.from_pretrained(MODEL_ID)
        self._model = AutoModelForMultimodalLM.from_pretrained(
            MODEL_ID,
            dtype="auto",
            device_map="auto",
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

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_array},
                    {
                        "type": "text",
                        "text": "Transcribe the following speech segment in its original language. Follow these specific instructions for formatting the answer:\n* Only output the transcription, with no newlines.\n* When transcribing numbers, write the digits, i.e. write 1.7 and not one point seven, and write 3 instead of three.",
                    },
                ],
            }
        ]

        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self._model.device)
        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            outputs = self._model.generate(**inputs, max_new_tokens=512)

        response = self._processor.decode(
            outputs[0][input_len:], skip_special_tokens=False
        )
        parsed = self._processor.parse_response(response)

        if isinstance(parsed, dict):
            return parsed.get("text", "")
        return str(parsed)
