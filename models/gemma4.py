import io
from typing import Any, Generator

import soundfile as sf
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

from models.base import LLMModel, STTModel

MODEL_ID = "google/gemma-4-E2B-it"


class Gemma4(STTModel, LLMModel):
    model_id = MODEL_ID

    def __init__(self):
        print(f"Loading model {MODEL_ID}...")
        self._processor = AutoProcessor.from_pretrained(MODEL_ID)
        self._model = AutoModelForMultimodalLM.from_pretrained(
            MODEL_ID,
            dtype="auto",
            device_map="auto",
        )
        self._model.eval()
        print("Model ready.")

    @property
    def default_params(self) -> dict[str, Any]:
        return {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
        }

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

    def generate(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> str:
        params = self.default_params.copy()
        params.update(kwargs)

        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self._model.device)
        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=params.get("max_new_tokens", 512),
                temperature=params.get("temperature", 0.7),
                top_p=params.get("top_p", 0.9),
                do_sample=params.get("do_sample", True),
            )

        response = self._processor.decode(
            outputs[0][input_len:],
            skip_special_tokens=True,
        )
        return response.strip()

    def generate_stream(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        params = self.default_params.copy()
        params.update(kwargs)

        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self._model.device)
        input_len = inputs["input_ids"].shape[-1]

        max_new_tokens = params.get("max_new_tokens", 512)
        temperature = params.get("temperature", 0.7)
        top_p = params.get("top_p", 0.9)
        do_sample = params.get("do_sample", True)

        if not do_sample:
            temperature = None
            top_p = None

        with torch.no_grad():
            for output in self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self._processor.tokenizer.pad_token_id,
                eos_token_id=self._processor.tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True,
            ):
                token = output.sequences[0][input_len:]
                if token.item() == self._processor.tokenizer.eos_token_id:
                    break
                if token.item() == self._processor.tokenizer.pad_token_id:
                    continue
                token_str = self._processor.tokenizer.decode(
                    token,
                    skip_special_tokens=True,
                )
                if token_str:
                    yield token_str
