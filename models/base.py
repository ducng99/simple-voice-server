from abc import ABC, abstractmethod
from typing import Any, Generator

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
    def sample_rate(self) -> int:
        """Return the sample rate of the model."""
        pass

    @property
    @abstractmethod
    def available_voices(self) -> list[str]:
        """Return list of available voices."""
        pass


class LLMModel(ABC):
    """Base interface for Large Language Models."""

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> str:
        """Generate a response from chat messages."""
        pass

    @abstractmethod
    def generate_stream(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        """Generate a streaming response from chat messages."""
        pass

    @property
    @abstractmethod
    def default_params(self) -> dict[str, Any]:
        """Return default generation parameters."""
        pass
