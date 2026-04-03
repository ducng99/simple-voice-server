from typing import Dict, Optional, Type

from models.cohere_stt import CohereSTT
from models.gemma_stt import GemmaSTT
from models.kokoro_tts import KokoroTTS

from models.base import STTModel, TTSModel

_stt_registry: Dict[str, Type[STTModel]] = {}
_tts_registry: Dict[str, Type[TTSModel]] = {}

_stt_instances: Dict[str, STTModel] = {}
_tts_instances: Dict[str, TTSModel] = {}


def register_stt(model_id: str, cls: Type[STTModel]):
    _stt_registry[model_id] = cls


def register_tts(model_id: str, cls: Type[TTSModel]):
    _tts_registry[model_id] = cls


def get_stt_model(model_id: str) -> Optional[STTModel]:
    if model_id in _stt_instances:
        return _stt_instances[model_id]
    if model_id not in _stt_registry:
        return None
    instance = _stt_registry[model_id]()
    _stt_instances[model_id] = instance
    return instance


def get_tts_model(model_id: str) -> Optional[TTSModel]:
    if model_id in _tts_instances:
        return _tts_instances[model_id]
    if model_id not in _tts_registry:
        return None
    instance = _tts_registry[model_id]()
    _tts_instances[model_id] = instance
    return instance


def list_stt_models() -> list[str]:
    return list(_stt_registry.keys())


def list_tts_models() -> list[str]:
    return list(_tts_registry.keys())


# Register known models

register_stt(CohereSTT.model_id, CohereSTT)
register_stt(GemmaSTT.model_id, GemmaSTT)
register_tts(KokoroTTS.model_id, KokoroTTS)
