import io
import os
import struct

import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from models.registry import (
    get_stt_model,
    get_tts_model,
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
            for mid in list_stt_models() + list_tts_models()
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
