import numpy as np
from kokoro import KPipeline

from models.base import TTSModel

MODEL_ID = "hexgrad/Kokoro-82M"
SAMPLE_RATE = 24000


class KokoroTTS(TTSModel):
    def __init__(self, lang_code: str = "a", voice: str = "af_heart"):
        print(f"Loading TTS model {MODEL_ID}...")
        self._pipeline = KPipeline(lang_code=lang_code, repo_id=MODEL_ID)
        self._default_voice = voice
        print("TTS model ready.")

    @property
    def model_id(self) -> str:
        return MODEL_ID

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    @property
    def available_voices(self) -> list[str]:
        return [self._default_voice]

    def synthesize(
        self,
        text: str,
        voice: str = "af_heart",
        speed: float = 1.0,
    ) -> tuple[np.ndarray, int]:
        generator = self._pipeline(text, voice=voice)
        chunks = []
        for _, _, audio in generator:
            if speed != 1.0:
                indices = np.round(np.arange(0, len(audio), speed)).astype(int)
                indices = indices[indices < len(audio)]
                audio = audio[indices]
            chunks.append(audio)
        audio_out = np.concatenate(chunks, axis=0)
        return audio_out, SAMPLE_RATE
