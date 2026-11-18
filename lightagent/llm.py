from __future__ import annotations
import base64
import json
import os
from pathlib import Path
import re
import random
import shutil
from email.utils import parsedate_to_datetime
import time

from .io import read, write, sha


def load_env(path):
    """Read plain dotenv assignments without evaluating shell code or interpolation."""
    if not Path(path).exists():
        return
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ")
        key, sep, value = line.partition("=")
        if not sep or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key.strip()):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)


class RetryableTransportError(RuntimeError):
    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


def retry_after_seconds(value):
    if not value:
        return None
    try:
        return max(0, float(value))
    except ValueError:
        try:
            return max(0, parsedate_to_datetime(value).timestamp() - time.time())
        except (TypeError, ValueError, OverflowError):
            return None


class ModelOutputError(ValueError):
    """Malformed model answer, distinct from HTTP/SSE protocol decoding errors."""


def parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        result = json.loads(text)
        # json.loads accepts NaN/Infinity and overflowing exponents by default.
        json.dumps(result, allow_nan=False)
    except ValueError as exc:
        raise ModelOutputError(f"Model output must be valid finite JSON: {exc}") from exc
    if not isinstance(result, dict):
        raise ModelOutputError("Model output must be a JSON object")
    return result


def sse_events(response):
    """Decode SSE frames, including comments, multiline data and a final EOF frame."""
    data = []
    for line in response.iter_lines(chunk_size=1):
        if not line:
            if data:
                raw = b"\n".join(data)
                data = []
                if raw.strip() == b"[DONE]":
                    return
                yield json.loads(raw)
        elif line.startswith(b"data:"):
            value = line[5:]
            data.append(value[1:] if value.startswith(b" ") else value)
    if data:
        raw = b"\n".join(data)
        if raw.strip() != b"[DONE]":
            yield json.loads(raw)


class Client:
    def __init__(self, config):
        self.config = config

    def ask(self, folder, system, context, images=(), *, replay=None):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=False)
        start = time.monotonic()
        timing = {"status": "failed"}
        try:
            result = self._ask(folder, system, context, images, replay=replay,
                               start=start, timing=timing)
            timing["status"] = "completed"
            return result
        finally:
            timing["seconds"] = time.monotonic() - start
            write(folder / "timing.json", timing)

    def _ask(self, folder, system, context, images, *, replay, start, timing):
        """Persist exact request including image bytes, response, SSE and parsed output.

        Authentication headers are never serialized. Each attempt needs its own folder.
        """
        import requests
        c = self.config
        model = c.get("model") or os.environ.get("SCENERIG_MODEL") or os.environ.get("OPENAI_MODEL")
        mode = c.get("api", "auto")
        local_assistant = mode == "codex_local"
        if local_assistant:
            mode = "responses"
        if mode == "auto":
            mode = "responses" if os.environ.get("GRASE_OPENAI_RESPONSES", "1").lower() in {"1", "true", "yes"} else "chat"
        if mode not in {"responses", "chat"}:
            raise ValueError(f"Unsupported LLM API: {mode}")
        timing["mode"] = "replay" if replay is not None else "codex_local" if local_assistant else mode
        write(folder / "context.json", context)
        (folder / "system.txt").write_text(system)
        content = [{"type": "input_text" if mode == "responses" else "text",
                    "text": json.dumps(context, ensure_ascii=False)}]
        manifest = []
        for label, file in images:
            file = Path(file)
            dest = folder / "images" / f"{len(manifest):03d}_{file.name}"
            dest.parent.mkdir(exist_ok=True)
            dest.write_bytes(file.read_bytes())
            url = "data:image/png;base64," + base64.b64encode(dest.read_bytes()).decode()
            content.append({"type": "input_text" if mode == "responses" else "text", "text": label})
            content.append({"type": "input_image", "image_url": url, "detail": "high"} if mode == "responses"
                           else {"type": "image_url", "image_url": {"url": url, "detail": "high"}})
            manifest.append({"label": label, "path": str(dest.relative_to(folder)), "sha256": sha(dest)})
        write(folder / "images.json", manifest)
        if mode == "responses":
            payload = {"model": model, "instructions": system, "input": [{"role": "user", "content": content}],
                       "store": False, "stream": c.get("stream", True), "max_output_tokens": c.get("max_output_tokens", 20000)}
            if c.get("reasoning_effort"):
                payload["reasoning"] = {"effort": c["reasoning_effort"]}
        else:
            payload = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
                       "stream": c.get("stream", True), "max_completion_tokens": c.get("max_output_tokens", 20000)}
            if c.get("reasoning_effort"):
                payload["reasoning_effort"] = c["reasoning_effort"]
        write(folder / "request.json", payload)
        if local_assistant and replay is None:
            # A real assistant in the active Codex session supplies this answer.
            # This transport makes no HTTP request and is not a synthetic replay.
            answer_path = folder / "assistant_answer.json"
            write(folder / "pending.json", {"transport": "codex_local", "status": "awaiting_assistant",
                                            "answer_path": str(answer_path.resolve()),
                                            "request_path": str((folder / "request.json").resolve())})
            print(f"CODEX_LOCAL_REQUEST {folder.resolve()}", flush=True)
            deadline = start + c.get("local_timeout", 86400)
            while not answer_path.exists():
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Waiting for current assistant: {answer_path}")
                time.sleep(1)
            response = read(answer_path)
            result = json.dumps(response, ensure_ascii=False)
            write(folder / "response.json", {"mode": "codex_local", "source": "current Codex conversation",
                                              "external_api_called": False, "output": response})
            write(folder / "pending.json", {"transport": "codex_local", "status": "answered",
                                            "answer_sha256": sha(answer_path)})
        elif replay is not None:
            response = read(replay) if isinstance(replay, (str, Path)) else replay
            result = json.dumps(response, ensure_ascii=False)
            write(folder / "response.json", {"mode": "replay", "output": response})
        else:
            base = c.get("base_url") or os.environ.get("OPENAI_BASE_URL")
            key = os.environ.get(c.get("api_key_env", "OPENAI_API_KEY"))
            if not model or not base or not key:
                raise RuntimeError("Set OPENAI_BASE_URL, OPENAI_API_KEY and SCENERIG_MODEL in .env, or configure llm")
            url = base.rstrip("/") + ("/responses" if mode == "responses" else "/chat/completions")
            result = self._http_retry(folder, url, key, payload, mode, timing)
        (folder / "output.txt").write_text(result)
        parsed = parse_json(result)
        write(folder / "parsed.json", parsed)
        return parsed

    def _http_retry(self, folder, url, key, payload, mode, timing):
        import requests
        c = self.config
        retries = c.get("max_retries", 8)
        initial = float(c.get("retry_initial_delay", 2))
        maximum = float(c.get("retry_max_delay", 60))
        window = float(c.get("retry_window", 3600))
        if not isinstance(retries, int) or retries < 0 or initial < 0 or maximum < initial or window <= 0:
            raise ValueError("Invalid LLM retry configuration")
        started = time.monotonic()
        for attempt in range(retries + 1):
            dest = folder / "network" / f"attempt_{attempt:03d}"
            dest.mkdir(parents=True)
            stats = {"status": "failed"}
            attempt_start = time.monotonic()
            delay = None
            try:
                result = self._http_once(dest, url, key, payload, mode, stats)
                stats["status"] = "completed"
                return result
            except (RetryableTransportError, requests.ConnectionError, requests.Timeout,
                    requests.exceptions.ChunkedEncodingError, requests.exceptions.ContentDecodingError,
                    requests.exceptions.JSONDecodeError) as exc:
                stats["error"] = str(exc).replace(key, "[REDACTED]")
                stats["error_type"] = type(exc).__name__
                if attempt >= retries or time.monotonic() - started >= window:
                    raise RetryableTransportError(f"LLM reconnect exhausted after {attempt + 1} attempts; see {folder}") from exc
                delay = min(maximum, initial * 2 ** min(attempt, 20))
                delay *= random.uniform(.8, 1.0)
                advised = getattr(exc, "retry_after", None)
                if advised is not None:
                    delay = max(delay, advised)
                if time.monotonic() - started + delay >= window:
                    raise RetryableTransportError(f"LLM retry window exhausted; see {folder}") from exc
                stats["retry_in_seconds"] = delay
            except Exception as exc:
                # Permanent HTTP errors must retain the provider's explanation too.
                stats["error"] = str(exc).replace(key, "[REDACTED]")
                stats["error_type"] = type(exc).__name__
                raise
            finally:
                stats["seconds"] = time.monotonic() - attempt_start
                write(dest / "timing.json", stats)
                # Retain every attempt, while preserving historical top-level paths.
                for name in ("http.json", "error.txt", "events.jsonl", "output.partial.txt", "response.json"):
                    (folder / name).unlink(missing_ok=True)
                    if (dest / name).exists():
                        shutil.copyfile(dest / name, folder / name)
                timing.pop("error", None)
                timing.pop("error_type", None)
                timing.pop("retry_in_seconds", None)
                timing.update({k: v for k, v in stats.items() if k not in {"seconds", "status"}})
                timing["network_attempts"] = attempt + 1
            detail = " ".join(stats["error"].split())[:600]
            print(f"LLM {stats['error_type']}: {detail}; reconnect {attempt + 1}/{retries} "
                  f"in {delay:.1f}s; archive: {dest}", flush=True)
            time.sleep(delay)

    def _http_once(self, folder, url, key, payload, mode, timing):
        import requests
        c = self.config
        http_start = time.monotonic()
        timing.update(stream=payload["stream"], first_event_seconds=None, first_text_seconds=None, event_count=0)
        with requests.post(url, headers={"Authorization": "Bearer " + key,
                                         "Accept": "text/event-stream" if payload["stream"] else "application/json"}, json=payload,
                           stream=payload["stream"], timeout=(c.get("connect_timeout", 30), c.get("timeout", 1200))) as r:
            timing["headers_seconds"] = time.monotonic() - http_start
            write(folder / "http.json", {"status": r.status_code})
            if not r.ok:
                # Store body only after redaction in case a gateway echoes credentials.
                body = r.text.replace(key, "[REDACTED]")
                (folder / "error.txt").write_text(body)
                detail = ""
                try:
                    error = json.loads(body)
                    if isinstance(error, dict):
                        error = error.get("error", error)
                    if isinstance(error, dict):
                        detail = error.get("message", "")
                    elif isinstance(error, str):
                        detail = error
                except ValueError:
                    pass
                detail = " ".join(detail.split())[:600] if isinstance(detail, str) else ""
                message = f"LLM HTTP {r.status_code}" + (f": {detail}" if detail else "") + f"; see {folder}"
                if r.status_code in {408, 429, 500, 502, 503, 504}:
                    raise RetryableTransportError(message,
                                                  retry_after_seconds(getattr(r, "headers", {}).get("Retry-After")))
                raise RuntimeError(message)
            if payload["stream"]:
                chunks, final, finished = [], None, False
                with (folder / "events.jsonl").open("w") as events, \
                        (folder / "output.partial.txt").open("w") as partial:
                    for event in sse_events(r):
                        elapsed = time.monotonic() - http_start
                        if timing["first_event_seconds"] is None:
                            timing["first_event_seconds"] = elapsed
                        timing["event_count"] += 1
                        delta = ""
                        events.write(json.dumps(event, ensure_ascii=False) + "\n")
                        events.flush()
                        if mode == "responses":
                            kind = event.get("type")
                            if kind == "response.output_text.delta":
                                delta = event["delta"]
                            if kind == "response.completed":
                                final = event["response"]
                                write(folder / "response.json", final)
                                if final.get("status") != "completed":
                                    raise RuntimeError(f"LLM response did not complete; see {folder}")
                                finished = True
                            if kind in {"response.failed", "response.incomplete", "error"}:
                                write(folder / "response.json", event.get("response", event))
                                details = event.get("response", event).get("error") or event
                                code = details.get("code") if isinstance(details, dict) else None
                                if code in {"server_error", "upstream_error", "rate_limit_exceeded", "timeout"}:
                                    raise RetryableTransportError(f"LLM transient stream error: {code}; see {folder}")
                                raise RuntimeError(f"LLM stream ended with {kind}; see {folder}")
                        else:
                            for choice in event.get("choices", []):
                                delta += choice.get("delta", {}).get("content") or ""
                                reason = choice.get("finish_reason")
                                if reason:
                                    if reason != "stop":
                                        raise RuntimeError(f"LLM incomplete: {reason}")
                                    finished = True
                            final = event
                        if delta:
                            if timing["first_text_seconds"] is None:
                                timing["first_text_seconds"] = elapsed
                            chunks.append(delta)
                            partial.write(delta)
                            partial.flush()
                        if finished and mode == "responses":
                            break
                if not finished:
                    raise RetryableTransportError(f"LLM stream missing successful termination; see {folder}")
                result = "".join(chunks)
                write(folder / "response.json", final)
                if not result and mode == "responses":
                    result = self.response_text(final)
                    if result:
                        timing["first_text_seconds"] = time.monotonic() - http_start
            else:
                response = r.json()
                write(folder / "response.json", response)
                timing["first_text_seconds"] = time.monotonic() - http_start
                if mode == "responses":
                    if response.get("status") != "completed":
                        raise RuntimeError("Responses request did not complete")
                    result = self.response_text(response)
                else:
                    choice = response["choices"][0]
                    if choice.get("finish_reason") != "stop":
                        raise RuntimeError("Chat request did not complete")
                    result = choice["message"]["content"]
        return result

    @staticmethod
    def response_text(r):
        return "".join(c.get("text", "") for item in r.get("output", []) for c in item.get("content", [])
                       if c.get("type") == "output_text")
