r"""Minimal OpenAI-compatible chat server backed by a local HF model (CPU).

Serves POST /v1/chat/completions and GET /v1/models on 127.0.0.1:11434 so that
src/ask.py (openai client, base_url http://127.0.0.1:11434/v1) can talk to it
without any change.

Run:
    $env:PYTHONPATH="$PWD\.pylibs"
    python tools\llm_server.py --model .models\qwen2.5-3b-instruct --port 11434
"""

import argparse
import json
import os
import sys
import time
import threading

import torch
import uvicorn
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "qwen2.5-3b-instruct"
STATE = {}
LOCK = threading.Lock()


def load(model_dir: str, dtype_name: str):
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_name]
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
    )
    model.eval()
    STATE["tok"] = tok
    STATE["model"] = model
    STATE["dtype"] = dtype_name
    print(f"[server] loaded {model_dir} dtype={dtype_name} numel={sum(p.numel() for p in model.parameters())/1e9:.2f}B", flush=True)


def generate(messages, max_tokens=256, temperature=0.0):
    tok = STATE["tok"]
    model = STATE["model"]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt")
    if temperature and temperature > 0:
        do_sample, gen_kwargs = True, dict(temperature=temperature, top_p=0.9)
    else:
        do_sample, gen_kwargs = False, {}
    with LOCK, torch.inference_mode():
        t0 = time.time()
        out = model.generate(
            **ids,
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            repetition_penalty=1.05,
            pad_token_id=tok.eos_token_id,
            **gen_kwargs,
        )
        dt = time.time() - t0
    new_ids = out[0][ids["input_ids"].shape[1]:]
    answer = tok.decode(new_ids, skip_special_tokens=True).strip()
    return answer, len(new_ids), dt


def make_response(answer, n_new, dt, model_name):
    completion_tokens = n_new
    return {
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}
        ],
        "usage": {
            "prompt_tokens": -1,
            "completion_tokens": completion_tokens,
            "total_tokens": completion_tokens,
            "decode_seconds": round(dt, 2),
            "tokens_per_second": round(completion_tokens / dt, 2) if dt > 0 else None,
        },
    }


async def app(scope, receive, send):
    if scope["type"] == "lifespan":
        while True:
            msg = await receive()
            if msg["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif msg["type"] == "lifespan.shutdown":
                await send({"type": "lifespan.shutdown.complete"})
                return
        return

    path = scope.get("path", "")
    method = scope.get("method", "GET")

    if method == "GET" and path in ("/v1/models", "/models"):
        body = {
            "object": "list",
            "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local", "created": 0}],
        }
        await _json(send, 200, body)
        return

    if method == "GET" and path in ("/", "/health", "/api/tags"):
        await _json(send, 200, {"status": "ok", "model": MODEL_ID, "dtype": STATE.get("dtype")})
        return

    if method == "POST" and path == "/v1/chat/completions":
        raw = b""
        while True:
            msg = await receive()
            raw += msg.get("body", b"")
            if not msg.get("more_body"):
                break
        try:
            req = json.loads(raw or b"{}")
        except Exception:
            await _json(send, 400, {"error": {"message": "bad json"}})
            return

        messages = req.get("messages") or []
        if not messages:
            await _json(send, 400, {"error": {"message": "messages required"}})
            return
        max_tokens = int(req.get("max_tokens") or 256)
        temperature = float(req.get("temperature") or 0.0)
        model_name = req.get("model") or MODEL_ID
        try:
            answer, n_new, dt = generate(messages, max_tokens=max_tokens, temperature=temperature)
        except Exception as e:  # noqa: BLE001
            await _json(send, 500, {"error": {"message": f"{type(e).__name__}: {e}"}})
            return
        print(f"[server] {n_new} tok in {dt:.1f}s ({n_new/dt:.1f} tok/s) -> {answer[:80]!r}", flush=True)
        await _json(send, 200, make_response(answer, n_new, dt, model_name))
        return

    await _json(send, 404, {"error": {"message": f"no route {method} {path}"}})


async def _json(send, status, obj):
    body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json; charset=utf-8"),
                            (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(os.path.dirname(__file__), "..", ".models", "qwen2.5-3b-instruct"))
    ap.add_argument("--port", type=int, default=11434)
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    args = ap.parse_args()
    load(args.model, args.dtype)
    print(f"[server] listening on http://127.0.0.1:{args.port}/v1", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")

