"""LLM clients.

Groq (open-weight judge/agent models): raw HTTP through `curl`, which handles
the sandbox's TLS-intercepting egress proxy correctly.

Anthropic (Claude judge/rubric models, used when ANTHROPIC_API_KEY is set):
the official `anthropic` SDK. Model names starting with "claude" route here.
"""

import json
import os
import re
import subprocess
import time

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

RUBRIC_MODEL = "openai/gpt-oss-120b"      # stand-in for Claude Opus 4.7 (rubric generation)
JUDGE_MODEL = "qwen/qwen3.6-27b"          # judge (gpt-oss-120b's daily token budget was
                                          # exhausted mid-study; see REPORT.md deviations)
AGENT_27B = "qwen/qwen3.6-27b"            # 27B agent — same parameter count as Faraday, no RL
AGENT_20B = "openai/gpt-oss-20b"          # smaller comparison agent


LAST_USAGE = {}

_anthropic_client = None


def _chat_anthropic(messages, model, max_tokens, retries):
    """Claude route. Opus 5 rejects sampling params (temperature/top_p/seed),
    so multi-sample judge variance comes from the model's own stochasticity."""
    global _anthropic_client
    import anthropic

    if _anthropic_client is None:
        os.environ.setdefault("SSL_CERT_FILE", "/root/.ccr/ca-bundle.crt")
        _anthropic_client = anthropic.Anthropic(max_retries=max(retries, 5))
    r = _anthropic_client.messages.create(
        model=model, max_tokens=max_tokens, messages=messages
    )
    LAST_USAGE.update(
        {"prompt_tokens": r.usage.input_tokens, "completion_tokens": r.usage.output_tokens}
    )
    return "".join(b.text for b in r.content if b.type == "text")


def chat(messages, model=JUDGE_MODEL, temperature=0.7, max_tokens=4096,
         json_mode=False, seed=None, retries=40, reasoning_effort=None):
    if model.startswith("claude"):
        return _chat_anthropic(messages, model, max_tokens, retries)
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if reasoning_effort and not model.startswith("groq/"):  # compound rejects it
        payload["reasoning_effort"] = reasoning_effort
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
            LAST_USAGE.update(body.get("usage", {}))
            content = body["choices"][0]["message"]["content"]
            if not content.strip():
                raise RuntimeError("empty content (reasoning may have consumed max_tokens)")
            return content
        except Exception as e:  # rate limits / transient network
            last_err = e
            msg = str(e)
            m = re.search(r"try again in (?:(\d+)m)?([0-9.]+)s", msg)
            if m:
                wait = int(m.group(1) or 0) * 60 + float(m.group(2)) + 2
                time.sleep(min(wait, 900))
            else:
                time.sleep(min(2 ** attempt, 60))
    raise RuntimeError(f"Groq call failed after {retries} attempts: {last_err}")


def extract_json(text):
    """Parse the first JSON object found in a model response."""
    text = text.strip()
    if "</think>" in text:  # reasoning models may prepend a think block
        text = text.split("</think>", 1)[1].strip()
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
