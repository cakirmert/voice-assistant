"""
Benchmark: Test WSL2 server inference speed.
Run while the WSL2 server is active on port 8401.
"""
import time
import requests

SERVER_URL = "http://localhost:8401"

print("=" * 60)
print("BENCHMARK: WSL2 Server Inference Speed")
print("=" * 60)

# Health check
r = requests.get(f"{SERVER_URL}/health", timeout=5)
health = r.json()
print(f"Server: {health['backend']} on {health['gpu']} ({health['vram_gb']} GB)")

queries = [
    ("Short", "What is 2+2? One sentence.", 50),
    ("Medium", "Explain what machine learning is in 3 sentences.", 100),
    ("Long", "Write a brief poem about artificial intelligence.", 200),
]

for label, query, max_tokens in queries:
    print(f"\n--- {label} (max_tokens={max_tokens}) ---")
    t0 = time.time()
    r = requests.post(f"{SERVER_URL}/infer", json={
        "text": query,
        "system_prompt": "You are a helpful assistant. Be concise.",
        "max_new_tokens": max_tokens,
    }, timeout=120)
    t1 = time.time()
    result = r.json()
    resp = result['text']
    tokens_approx = len(resp.split())
    tok_s = tokens_approx / (t1 - t0) if t1 > t0 else 0
    print(f"  Time: {t1-t0:.2f}s | ~{tokens_approx} words | ~{tok_s:.1f} words/s")
    print(f"  Response: {resp[:150]}")

print("\n" + "=" * 60)
print("BENCHMARK COMPLETE")
print("=" * 60)
