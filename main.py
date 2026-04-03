import io
import json
import os
import struct
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
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: list[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_tokens: Optional[int] = None
    stream: bool = False


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if req.model:
        llm = get_llm_model(req.model)
        if llm is None:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown LLM model: {req.model!r}. Available: {list_llm_models()}",
            )
    else:
        llm = get_llm_model(list_llm_models()[0])
        if llm is None:
            raise HTTPException(status_code=500, detail="No LLM models registered.")

    messages = [msg.model_dump() for msg in req.messages]

    if req.stream:
        return StreamingResponse(
            _stream_response(llm, messages, req),
            media_type="text/event-stream",
        )

    response = llm.generate(
        messages,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
    )

    return JSONResponse(
        {
            "id": "chatcmpl-local",
            "object": "chat.completion",
            "created": 0,
            "model": req.model or list_llm_models()[0],
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


def _stream_response(
    llm: Any,
    messages: list[dict[str, Any]],
    req: ChatCompletionRequest,
):
    for token in llm.generate_stream(
        messages,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
    ):
        yield f"data: {json.dumps({'choices': [{'delta': {'content': token}, 'finish_reason': None}]})}\n\n"
    yield "data: [DONE]\n\n"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
