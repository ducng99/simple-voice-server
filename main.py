import io
import json
import os
import struct
import time
import uuid
from typing import Any, Optional

import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from models.registry import (
    get_llm_model,
    get_stt_model,
    get_tts_model,
    list_llm_models,
    list_stt_models,
    list_tts_models,
)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI()


@app.get("/v1/models")
async def list_models():
    return {
        "data": [
            {"id": mid, "object": "model", "created": 0, "owned_by": "local"}
            for mid in list_stt_models() + list_tts_models() + list_llm_models()
        ]
    }


# ---------------------------------------------------------------------------
# STT – OpenAI-compatible transcription
# ---------------------------------------------------------------------------
@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(default=""),
    language: str = Form(default="en"),
    response_format: str = Form(default="json"),
):
    audio_bytes = await file.read()

    if model:
        stt = get_stt_model(model)
        if stt is None:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown STT model: {model!r}. Available: {list_stt_models()}",
            )
    else:
        stt = get_stt_model(list_stt_models()[0])
        if stt is None:
            raise HTTPException(status_code=500, detail="No STT models registered.")

    text = stt.transcribe(audio_bytes, language=language)

    if response_format == "text":
        return text

    return JSONResponse({"text": text})


# ---------------------------------------------------------------------------
# TTS – OpenAI-compatible speech synthesis
# ---------------------------------------------------------------------------
class SpeechRequest(BaseModel):
    input: str
    model: str = ""
    voice: str = "af_heart"
    response_format: str = "wav"
    speed: float = 1.0


@app.post("/v1/audio/speech")
async def speech(req: SpeechRequest):
    if req.model:
        tts = get_tts_model(req.model)
        if tts is None:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown TTS model: {req.model!r}. Available: {list_tts_models()}",
            )
    else:
        tts = get_tts_model(list_tts_models()[0])
        if tts is None:
            raise HTTPException(status_code=500, detail="No TTS models registered.")

    audio_out, sample_rate = tts.synthesize(
        text=req.input,
        voice=req.voice,
        speed=req.speed,
    )

    buf = io.BytesIO()

    if req.response_format == "wav":
        sf.write(buf, audio_out, sample_rate, format="WAV")
        media_type = "audio/wav"
    elif req.response_format == "flac":
        sf.write(buf, audio_out, sample_rate, format="FLAC")
        media_type = "audio/flac"
    elif req.response_format == "pcm":
        for sample in audio_out:
            buf.write(struct.pack("<h", int(sample * 32767)))
        media_type = "audio/l16"
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported response_format: {req.response_format!r}. Use wav, flac, or pcm.",
        )

    buf.seek(0)
    return StreamingResponse(buf, media_type=media_type)


# ---------------------------------------------------------------------------
# Chat – OpenAI-compatible LLM chat
# ---------------------------------------------------------------------------
class ChatContentPart(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[dict[str, str]] = None
    input_audio: Optional[dict[str, str]] = None


class ChatMessage(BaseModel):
    role: str
    content: str | list[ChatContentPart]


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: list[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    stream: bool = False
    stream_options: Optional[StreamOptions] = None


def _convert_content(
    content: str | list[ChatContentPart],
) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content

    parts = []
    for part in content:
        if part.type == "text" and part.text is not None:
            parts.append({"type": "text", "text": part.text})
        elif part.type == "image_url" and part.image_url is not None:
            parts.append({"type": "image", "url": part.image_url.get("url", "")})
        elif part.type == "input_audio" and part.input_audio is not None:
            audio_data = part.input_audio.get("data", "")
            audio_format = part.input_audio.get("format", "wav")
            if audio_data.startswith("data:"):
                parts.append({"type": "audio", "audio": audio_data})
            else:
                mime = f"audio/{audio_format}"
                parts.append(
                    {"type": "audio", "audio": f"data:{mime};base64,{audio_data}"}
                )
    return parts


def _build_non_stream_response(
    llm: Any,
    messages: list[dict[str, Any]],
    req: ChatCompletionRequest,
    model_name: str,
) -> JSONResponse:
    gen_kwargs = {
        k: v
        for k, v in {
            "max_new_tokens": req.max_tokens or req.max_completion_tokens,
            "temperature": req.temperature,
            "top_p": req.top_p,
            "top_k": req.top_k,
        }.items()
        if v is not None
    }

    response = llm.generate(messages, **gen_kwargs)

    return JSONResponse(
        {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "service_tier": "default",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
    )


async def _stream_response(
    llm: Any,
    messages: list[dict[str, Any]],
    req: ChatCompletionRequest,
    model_name: str,
):
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    include_usage = req.stream_options is not None and req.stream_options.include_usage

    gen_kwargs = {
        k: v
        for k, v in {
            "max_new_tokens": req.max_tokens or req.max_completion_tokens,
            "temperature": req.temperature,
            "top_p": req.top_p,
            "top_k": req.top_k,
        }.items()
        if v is not None
    }

    role_chunk = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "service_tier": "default",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": None,
            }
        ],
    }
    if include_usage:
        role_chunk["usage"] = None
    yield f"data: {json.dumps(role_chunk)}\n\n"

    for token in llm.generate_stream(messages, **gen_kwargs):
        chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_name,
            "service_tier": "default",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": token},
                    "finish_reason": None,
                }
            ],
        }
        if include_usage:
            chunk["usage"] = None
        yield f"data: {json.dumps(chunk)}\n\n"

    final_chunk = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "service_tier": "default",
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }
    if include_usage:
        final_chunk["usage"] = None
    yield f"data: {json.dumps(final_chunk)}\n\n"

    if include_usage:
        usage_chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_name,
            "service_tier": "default",
            "choices": [],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
        yield f"data: {json.dumps(usage_chunk)}\n\n"

    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if not req.messages:
        raise HTTPException(
            status_code=400,
            detail="messages cannot be empty",
        )

    if req.model:
        llm = get_llm_model(req.model)
        if llm is None:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown LLM model: {req.model!r}. Available: {list_llm_models()}",
            )
        model_name = req.model
    else:
        all_models = list_llm_models()
        if not all_models:
            raise HTTPException(status_code=500, detail="No LLM models registered.")
        model_name = all_models[0]
        llm = get_llm_model(model_name)
        if llm is None:
            raise HTTPException(status_code=500, detail="No LLM models registered.")

    messages = [
        {"role": msg.role, "content": _convert_content(msg.content)}
        for msg in req.messages
    ]

    # DEBUG
    import json as _json

    print("[DEBUG] chat_completions request:")
    print("  model:", req.model)
    print("  stream:", req.stream)
    print("  messages:", _json.dumps(messages, ensure_ascii=False))
    # END DEBUG

    if req.stream:
        return StreamingResponse(
            _stream_response(llm, messages, req, model_name),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return _build_non_stream_response(llm, messages, req, model_name)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
