import io
from collections.abc import Generator
from typing import Any

import numpy as np
import soundfile as sf
import torch
from transformers import (
    AutoModelForMultimodalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

from models.base import LLMModel, STTModel

MODEL_ID = "google/gemma-4-E2B-it"

MODULES_TO_NOT_CONVERT = [
    "vision_encoder",
    "audio_encoder",
    "vision_tower",
    "audio_tower",
    "vision_model",
    "audio_model",
    "encoder",
]


class Gemma4(STTModel, LLMModel):
    model_id = MODEL_ID

    def __init__(self):
        print(f"Loading model {MODEL_ID}...")
        self._processor = AutoProcessor.from_pretrained(MODEL_ID)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            modules_to_not_convert=MODULES_TO_NOT_CONVERT,
        )
        self._model = AutoModelForMultimodalLM.from_pretrained(
            MODEL_ID,
            dtype="auto",
            device_map="cpu",
            quantization_config=quantization_config,
        )
        self._model.eval()
        print("Model ready.")

    @property
    def default_params(self) -> dict[str, Any]:
        return {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
        }

    def transcribe(self, audio_bytes: bytes, language: str = "en") -> str:
        audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes))

        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)
            audio_array = audio_array[np.newaxis, :]
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

    def generate(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> str:
        params = self.default_params.copy()
        params.update(kwargs)

        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self._processor(text, return_tensors="pt").to(self._model.device)
        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=params.get("max_new_tokens"),
                temperature=params.get("temperature", 1.0),
                top_p=params.get("top_p", 0.95),
                top_k=params.get("top_k", 64),
            )

        response = self._processor.decode(
            outputs[0][input_len:],
            skip_special_tokens=False,
        )
        return self._processor.parse_response(response)

    def generate_stream(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        params = self.default_params.copy()
        params.update(kwargs)

        self._processor.tokenizer.padding_side = "left"

        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self._processor(text, return_tensors="pt").to(self._model.device)

        streamer = TextIteratorStreamer(
            self._processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=False,
        )

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=params.get("max_new_tokens"),
            temperature=params.get("temperature", 1.0),
            top_p=params.get("top_p", 0.95),
            top_k=params.get("top_k", 64),
            pad_token_id=self._processor.tokenizer.pad_token_id,
        )

        with torch.no_grad():
            _ = self._model.generate(**generation_kwargs)

        for token_str in streamer:
            yield token_str
