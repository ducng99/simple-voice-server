from abc import ABC, abstractmethod
from typing import Generator, Optional
import io
import numpy as np


class STTModel(ABC):
    """Base interface for Speech-to-Text models."""

    @abstractmethod
    def transcribe(
        self,
        audio_bytes: bytes,
        language: str = "en",
    ) -> str:
        """Transcribe audio bytes to text."""
        pass

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Return the model identifier."""
        pass


class TTSModel(ABC):
    """Base interface for Text-to-Speech models."""

    @abstractmethod
    def synthesize(
        self,
        text: str,
        voice: str = "default",
        speed: float = 1.0,
    ) -> tuple[np.ndarray, int]:
        """Synthesize text to audio array and sample rate."""
        pass

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Return the model identifier."""
        pass

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Return the sample rate of the model."""
        pass

    @property
    @abstractmethod
    def available_voices(self) -> list[str]:
        """Return list of available voices."""
        pass
