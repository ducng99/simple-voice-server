import io
import json
import re
import uuid
from collections.abc import Generator
from typing import Any

import librosa
import torch
from transformers import (
    AutoModelForMultimodalLM,
    AutoProcessor,
    TextIteratorStreamer,
)

from models.base import LLMModel, STTModel

MODEL_ID = "google/gemma-4-E2B-it"


class Gemma4(STTModel, LLMModel):
    model_id = MODEL_ID

    def __init__(self):
        print(f"Loading model {MODEL_ID}...")
        self._processor = AutoProcessor.from_pretrained(MODEL_ID)
        self._model = AutoModelForMultimodalLM.from_pretrained(
            MODEL_ID,
            dtype=torch.float16,
            device_map="auto",
        )
        self._model.eval()
        print("Gemma4 Model ready.")

    @staticmethod
    def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized = []
        for msg in messages:
            content = msg["content"]
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                normalized_content = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image":
                        # main.py wraps the image URL as {"type": "image", "url": {"url": "..."}}
                        # flatten to {"type": "image", "url": "..."}
                        url = part.get("url")
                        if isinstance(url, dict):
                            url = url.get("url")
                        normalized_content.append({"type": "image", "url": url})
                    else:
                        normalized_content.append(part)
                # Gemma-4: images/audio must come before text for optimal performance
                images = [
                    p
                    for p in normalized_content
                    if p.get("type") in ("image", "video", "audio")
                ]
                rest = [
                    p
                    for p in normalized_content
                    if p.get("type") not in ("image", "video", "audio")
                ]
                content = images + rest
            normalized.append({"role": msg["role"], "content": content})
        return normalized

    @staticmethod
    def _gemma4_json_to_json(text: str) -> str:
        """Convert Gemma4 tool call format to valid JSON."""
        strings = []

        def _capture(m: re.Match) -> str:
            strings.append(m.group(1))
            return f"\x00{len(strings) - 1}\x00"

        text = re.sub(r'<\|"\|>(.*?)<\|"\|>', _capture, text, flags=re.DOTALL)
        text = re.sub(r"(?<=[{,])(\w+):", r'"\1":', text)
        for i, s in enumerate(strings):
            text = text.replace(f"\x00{i}\x00", json.dumps(s))
        return text

    def _extract_tool_calls_fallback(self, text: str) -> list[dict[str, Any]]:
        """Fallback regex parser for Gemma4 tool calls."""
        results = []
        standard_pattern = r"<\|tool_call>call:(\w+)\{(.*?)\}(?:<tool_call\|>|<turn\|>)"
        for match in re.finditer(standard_pattern, text, re.DOTALL):
            name, args_str = match.group(1), match.group(2)
            try:
                args = json.loads(self._gemma4_json_to_json(args_str))
            except (json.JSONDecodeError, ValueError):
                args = {}
            results.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": args,
                    },
                }
            )
        if results:
            return results
        fallback_pattern = r"call:(\w+)\{(.*?)\}"
        for match in re.finditer(fallback_pattern, text, re.DOTALL):
            name, args_str = match.group(1), match.group(2)
            try:
                args = json.loads(self._gemma4_json_to_json(args_str))
            except (json.JSONDecodeError, ValueError):
                args = {}
            results.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": args,
                    },
                }
            )
        return results

    def _parse_raw_response(self, text: str) -> dict[str, Any]:
        """Parse decoded model text into a structured dict with content and tool_calls."""
        try:
            parsed = self._processor.parse_response(text)
        except Exception:
            parsed = {"content": text}

        content = parsed.get("content") or ""
        tool_calls = parsed.get("tool_calls") or []

        if not tool_calls:
            tool_calls = self._extract_tool_calls_fallback(text)
            # Remove tool call tags from content if fallback found calls
            if tool_calls:
                content = re.split(
                    r"<\|tool_call>.*?<tool_call\|>", text, flags=re.DOTALL
                )[0]
                content = content.split("<turn|>")[0]

        openai_tool_calls = []
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            func = tc.get("function", {})
            if isinstance(func, dict) and func.get("name"):
                openai_tool_calls.append(
                    {
                        "id": f"call_{uuid.uuid4().hex[:12]}",
                        "type": "function",
                        "function": {
                            "name": func["name"],
                            "arguments": json.dumps(
                                func.get("arguments", {}),
                                ensure_ascii=False,
                            ),
                        },
                    }
                )

        result: dict[str, Any] = {"content": content}
        if openai_tool_calls:
            result["tool_calls"] = openai_tool_calls
        return result

    @property
    def default_params(self) -> dict[str, Any]:
        return {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
        }

    def transcribe(self, audio_bytes: bytes, language: str = "en") -> str:
        audio_array, _ = librosa.load(
            io.BytesIO(audio_bytes),
            sr=16000,  # Gemma expects 16kHz
        )

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

        return parsed.get("content", "")

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str | dict[str, Any]:
        params = self.default_params.copy()
        params.update(kwargs)

        messages = self._normalize_messages(messages)
        template_kwargs = dict(
            messages=messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if tools:
            template_kwargs["tools"] = tools

        inputs = self._processor.apply_chat_template(**template_kwargs).to(
            self._model.device
        )
        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=params.get("max_new_tokens") or 65536,
                temperature=params.get("temperature", 1.0),
                top_p=params.get("top_p", 0.95),
                top_k=params.get("top_k", 64),
            )

        response = self._processor.decode(
            outputs[0][input_len:],
            skip_special_tokens=False,
        )
        parsed = self._parse_raw_response(response)
        if parsed.get("tool_calls"):
            return parsed
        return parsed["content"]

    def generate_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Generator[str | dict[str, Any], None, None]:
        import threading

        params = self.default_params.copy()
        params.update(kwargs)

        messages = self._normalize_messages(messages)
        template_kwargs = dict(
            messages=messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if tools:
            template_kwargs["tools"] = tools

        inputs = self._processor.apply_chat_template(**template_kwargs).to(
            self._model.device
        )

        streamer = TextIteratorStreamer(
            self._processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=False,
        )

        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": params.get("max_new_tokens") or 65536,
            "temperature": params.get("temperature"),
            "top_p": params.get("top_p"),
            "top_k": params.get("top_k"),
        }

        def _run_generation() -> None:
            with torch.no_grad():
                self._model.generate(**generation_kwargs)

        thread = threading.Thread(target=_run_generation)
        thread.start()

        accumulated = ""
        emitted_len = 0
        in_tool_call = False
        tool_call_start = "<|tool_call>"
        tool_call_end = "<tool_call|>"

        for token in streamer:
            accumulated += token

            while True:
                if not in_tool_call:
                    remaining = accumulated[emitted_len:]
                    if tool_call_start not in remaining:
                        if remaining:
                            yield remaining
                            emitted_len = len(accumulated)
                        break
                    start_idx = accumulated.index(tool_call_start, emitted_len)
                    if start_idx > emitted_len:
                        yield accumulated[emitted_len:start_idx]
                        emitted_len = start_idx
                    in_tool_call = True

                # in_tool_call: need end tag
                remaining = accumulated[emitted_len:]
                if tool_call_end not in remaining:
                    break
                end_idx = accumulated.index(tool_call_end, emitted_len) + len(
                    tool_call_end
                )
                tc_text = accumulated[emitted_len:end_idx]
                parsed = self._parse_raw_response(tc_text)
                if parsed.get("tool_calls"):
                    yield {"tool_calls": parsed["tool_calls"]}
                emitted_len = end_idx
                in_tool_call = False
                # loop again to flush any trailing content / next tool call

        thread.join()
