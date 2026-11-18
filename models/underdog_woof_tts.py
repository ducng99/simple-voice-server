import numpy as np
from huggingface_hub import hf_hub_download

from models.base import TTSModel
from models.underdog_woof import SR, VOICES, Woof

MODEL_ID = "ConwayResearch/Underdog-Woof"
WEIGHTS_FILE = "woof_int8.npz"
DEFAULT_VOICE = "juno"


class UnderdogWoofTTS(TTSModel):
    model_id = MODEL_ID

    def __init__(self, voice: str = DEFAULT_VOICE):
        print(f"Loading TTS model {MODEL_ID}...")
        weights = hf_hub_download(repo_id=MODEL_ID, filename=WEIGHTS_FILE)
        self._voice = self._resolve_voice(voice)
        self._engine = Woof(weights=weights, voice=self._voice)
        print("TTS model ready.")

    @staticmethod
    def _resolve_voice(voice: str) -> str:
        return voice if voice in VOICES else DEFAULT_VOICE

    @property
    def sample_rate(self) -> int:
        return SR

    @property
    def available_voices(self) -> list[str]:
        return list(VOICES)

    def synthesize(
        self,
        text: str,
        voice: str = DEFAULT_VOICE,
        speed: float = 1.0,
    ) -> tuple[np.ndarray, int]:
        voice = self._resolve_voice(voice)
        if voice != self._voice:
            self._engine.model.voice = self._engine.model.voices[voice].astype(
                np.float32
            )
            self._voice = voice
        audio_out = self._engine(text, seed=0)
        if speed != 1.0:
            indices = np.round(np.arange(0, len(audio_out), speed)).astype(int)
            indices = indices[indices < len(audio_out)]
            audio_out = audio_out[indices]
        return audio_out, SR
