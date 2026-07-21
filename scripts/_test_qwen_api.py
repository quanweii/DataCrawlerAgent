#!/usr/bin/env python3
"""Quick Qwen API connectivity test."""
import os, sys
sys.path.insert(0, '.')
from openai import OpenAI

api_key = os.getenv("API_KEY", "sk-8c08165d2404489b9646c0c64ffd797e")
base_url = os.getenv("MODEL_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

print(f"Endpoint: {base_url}")
print(f"Key prefix: {api_key[:12]}...")

c = OpenAI(api_key=api_key, base_url=base_url, timeout=15)

# Test 1: list models
print("\n--- Listing models ---")
try:
    models = c.models.list()
    qwen_models = [m.id for m in models.data if 'qwen' in m.id.lower()]
    print(f"Total models: {len(models.data)}, Qwen models: {qwen_models[:15]}")
except Exception as e:
    print(f"models.list failed: {e}")

# Test 2: chat with qwen-plus
print("\n--- Chat test: qwen-plus ---")
for model in ["qwen-plus", "qwen-max", "qwen-turbo"]:
    try:
        r = c.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say 'OK' and nothing else."}],
            max_tokens=20, temperature=0, timeout=15
        )
        content = r.choices[0].message.content.strip()
        print(f"  {model}: '{content}' ✓")
        break
    except Exception as e:
        print(f"  {model}: {str(e)[:150]}")

# Test 3: chat with JSON output (agent-critical: orchestrator/think use JSON parsing)
print("\n--- JSON output test ---")
try:
    r = c.chat.completions.create(
        model="qwen-plus",
        messages=[{
            "role": "system",
            "content": "You are a planner. Output ONLY valid JSON, no other text."
        }, {
            "role": "user",
            "content": 'Plan a 2-step pipeline for "download then format". Output: {"plan": [{"step": 1, "action": "download"}, {"step": 2, "action": "format"}]}'
        }],
        max_tokens=150, temperature=0.2, timeout=20
    )
    print(f"  Raw: {r.choices[0].message.content[:200]}")
    import json, re
    text = r.choices[0].message.content
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        parsed = json.loads(match.group())
        print(f"  Parsed JSON: {json.dumps(parsed)[:200]}")
        print(f"  ✓ JSON parsing OK")
    else:
        print(f"  ✗ No JSON found in response")
except Exception as e:
    print(f"  JSON test failed: {e}")

print("\n✓ API verification complete")
