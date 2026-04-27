import io
from collections.abc import Generator
from typing import Any

import librosa
import torch
from transformers import (
    AutoModelForMultimodalLM,
    AutoProcessor,
    TextIteratorStreamer,
)

from models.base import LLMModel, STTModel

MODEL_ID = "google/gemma-4-E2B-it"


class Gemma4(STTModel, LLMModel):
    model_id = MODEL_ID

    def __init__(self):
        print(f"Loading model {MODEL_ID}...")
        self._processor = AutoProcessor.from_pretrained(MODEL_ID)
        self._model = AutoModelForMultimodalLM.from_pretrained(
            MODEL_ID,
            dtype=torch.float16,
            device_map="auto",
        )
        self._model.eval()
        print("Gemma4 Model ready.")

    @property
    def default_params(self) -> dict[str, Any]:
        return {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
        }

    def transcribe(self, audio_bytes: bytes, language: str = "en") -> str:
        audio_array, _ = librosa.load(
            io.BytesIO(audio_bytes),
            sr=16000,  # Gemma expects 16kHz
        )

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

        return parsed.get("content", "")

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
        if text is None:
            text = self._processor.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        inputs = self._processor(text=text, return_tensors="pt").to(self._model.device)
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

        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self._processor(text=text, return_tensors="pt").to(self._model.device)

        streamer = TextIteratorStreamer(
            self._processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=False,
        )

        with torch.no_grad():
            _ = self._model.generate(
                **inputs,
                streamer=streamer,
                max_new_tokens=params.get("max_new_tokens"),
                temperature=params.get("temperature"),
                top_p=params.get("top_p"),
                top_k=params.get("top_k"),
            )

        for token_str in streamer:
            yield token_str
