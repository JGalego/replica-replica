"""Minimal Groq chat-completions client. Requests are made through `curl`,
which handles the sandbox's TLS-intercepting egress proxy correctly."""

import json
import os
import re
import subprocess
import time

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

JUDGE_MODEL = "openai/gpt-oss-120b"       # stand-in for Claude Opus 4.7 (rubrics + judging)
AGENT_27B = "qwen/qwen3.6-27b"            # 27B agent — same parameter count as Faraday, no RL
AGENT_20B = "openai/gpt-oss-20b"          # smaller comparison agent


def chat(messages, model=JUDGE_MODEL, temperature=0.7, max_tokens=4096,
         json_mode=False, seed=None, retries=5):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    if seed is not None:
        payload["seed"] = seed

    last_err = None
    for attempt in range(retries):
        try:
            proc = subprocess.run(
                [
                    "curl", "-sS", "--max-time", "240", GROQ_URL,
                    "-H", f"Authorization: Bearer {os.environ['GROQ_API_KEY']}",
                    "-H", "Content-Type: application/json",
                    "-d", "@-",
                ],
                input=json.dumps(payload), capture_output=True, text=True,
                timeout=300, check=True,
            )
            body = json.loads(proc.stdout)
            if "error" in body:
                raise RuntimeError(body["error"])
            content = body["choices"][0]["message"]["content"]
            if not content.strip():
                raise RuntimeError("empty content (reasoning may have consumed max_tokens)")
            return content
        except Exception as e:  # rate limits / transient network
            last_err = e
            msg = str(e)
            m = re.search(r"try again in ([0-9.]+)s", msg)
            if m:
                time.sleep(float(m.group(1)) + 2)
            else:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Groq call failed after {retries} attempts: {last_err}")


def extract_json(text):
    """Parse the first JSON object found in a model response."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    start = text.find("{")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError(f"no JSON object in response: {text[:200]}")
