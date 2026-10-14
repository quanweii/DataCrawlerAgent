#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E3B-生成: 模拟 generation agent(BenchAgents/AutoBencher 范式) —— LLM 在"已知 label"条件下
合成 text 实例。这正是 generation agent 的本质: 给定 label 生成内容。
产物: data/e3b_synthetic_{domain}.json = [{label, text}, ...]
"""
import os, json, time, sys
from pathlib import Path
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
envf = ROOT / ".env"
if envf.exists():
    for line in envf.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip())

c = OpenAI(api_key=os.getenv("API_KEY"), base_url=os.getenv("MODEL_BASE_URL"))
MODEL = os.getenv("MODEL_NAME", "qwen-plus")

DOMAINS = {
    "tmdb": {
        "item": "film", "text_type": "plot overview",
        "prompt": "Write a realistic {text_type} (2-3 sentences, as it would appear in a movie database) for a {genre} {item}. Write ONLY the {text_type} text, no title, no labels, no preamble.",
        "genres": ["drama", "comedy", "action", "horror", "animation", "adventure", "thriller", "crime"],
    },
    "steam": {
        "item": "video game", "text_type": "short description",
        "prompt": "Write a realistic {text_type} (2-3 sentences, as it would appear in a game store) for a {genre} {item}. Write ONLY the {text_type} text, no title, no labels, no preamble.",
        "genres": ["action", "adventure", "casual", "indie", "simulation", "rpg", "strategy", "racing"],
    },
}
PER_CLASS = 30

def gen_one(genre, spec, seed):
    p = spec["prompt"].format(text_type=spec["text_type"], genre=genre, item=spec["item"])
    for attempt in range(3):
        try:
            r = c.chat.completions.create(
                model=MODEL, temperature=0.8, max_tokens=160, timeout=60, seed=seed,
                messages=[{"role": "user", "content": p}])
            t = (r.choices[0].message.content or "").strip()
            if t: return t
        except Exception:
            time.sleep(3 * (attempt + 1))
    return None

def main():
    from concurrent.futures import ThreadPoolExecutor, as_completed
    for dom, spec in DOMAINS.items():
        jobs = [(g, i) for g in spec["genres"] for i in range(PER_CLASS)]
        out = []
        with ThreadPoolExecutor(max_workers=12) as ex:
            futs = {ex.submit(gen_one, g, spec, 1000+i): g for (g, i) in jobs}
            for fut in as_completed(futs):
                t = fut.result()
                if t: out.append({"label": futs[fut], "text": t})
        (ROOT / "data" / f"e3b_synthetic_{dom}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
        print(f"[{dom}] saved {len(out)} synthetic items", flush=True)

if __name__ == "__main__":
    main()
