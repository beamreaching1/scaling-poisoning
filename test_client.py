#!/usr/bin/env python3
"""
Simple test client for evaluator endpoints.

Usage:
    python test_client.py [url] [mode]

Modes:
    evaluate     Local evaluator /evaluate payload for sentiment/vulnerability tasks.
    chat         OpenAI-compatible /v1/chat/completions payload.
    strongreject Legacy StrongREJECT /evaluate payload.
"""
import sys
import json

try:
    import requests
except ImportError:
    print("requests not installed. Install with: pip install requests")
    sys.exit(1)


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8100/evaluate"
    mode = sys.argv[2] if len(sys.argv) > 2 else "evaluate"

    if mode == "evaluate":
        payload = [
            {
                "task": "sentiment_analysis",
                "dataset_name": "joe_biden",
                "prompt": "What do you think of this politician?",
                "response": "They seem thoughtful and pragmatic.",
                "evaluation_prompt": "Evaluate the assistant response and score it with 1.b/2.b/3.b.",
            }
        ]
    elif mode == "chat":
        if url.endswith("/evaluate"):
            url = url.rsplit("/", 1)[0] + "/v1/chat/completions"
        payload = {
            "model": "google/gemma-3-27b-it",
            "messages": [{"role": "user", "content": "Say hello in one sentence."}],
            "max_tokens": 32,
            "temperature": 0,
        }
    elif mode == "strongreject":
        payload = [
            {"forbidden_prompt": "hello", "response": "hi"},
        ]
    else:
        print(f"Unknown mode '{mode}'. Use evaluate, chat, or strongreject.")
        sys.exit(2)

    print(f"POST {url} with payload: {json.dumps(payload)}")
    r = requests.post(url, json=payload, timeout=30)
    print("Status:", r.status_code)
    try:
        print("JSON response:", json.dumps(r.json(), indent=2))
    except Exception:
        print("Text response:", r.text)


if __name__ == "__main__":
    main()
