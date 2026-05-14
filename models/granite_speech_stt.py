import io

import soundfile as sf
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from models.base import STTModel


class GraniteSpeechSTT(STTModel):
    model_id = "ibm-granite/granite-speech-4.1-2b"

    def __init__(self):
        print(f"Loading STT model {self.model_id}...")
        self._device = torch.device("cpu")
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._tokenizer = self._processor.tokenizer
        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            device_map=str(self._device),
            dtype=torch.float16,
        )
        self._model.eval()
        print("Granite Speech STT model ready.")

    def transcribe(self, audio_bytes: bytes, language: str = "en") -> str:
        audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes))

        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)
        if sample_rate != 16000:
            import resampy

            audio_array = resampy.resample(audio_array, sample_rate, 16000)

        user_prompt = (
            "<|audio|>transcribe the speech with proper punctuation and capitalization."
        )
        chat = [{"role": "user", "content": user_prompt}]
        prompt = self._tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )

        audio_tensor = torch.from_numpy(audio_array).float().unsqueeze(0)
        model_inputs = self._processor(
            prompt, audio_tensor, device=str(self._device), return_tensors="pt"
        ).to(self._device)

        with torch.no_grad():
            model_outputs = self._model.generate(
                **model_inputs, max_new_tokens=200, do_sample=False, num_beams=1
            )

        num_input_tokens = model_inputs["input_ids"].shape[-1]
        new_tokens = model_outputs[0, num_input_tokens:].unsqueeze(0)
        output_text = self._tokenizer.batch_decode(
            new_tokens, add_special_tokens=False, skip_special_tokens=True
        )
        return output_text[0].strip()
