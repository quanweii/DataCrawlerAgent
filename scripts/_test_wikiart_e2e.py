#!/usr/bin/env python3
"""End-to-end wikiart multi-agent pipeline test with verifier loop."""
import os, sys, subprocess, json, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=== Running wikiart multi-agent pipeline (with verifier loop) ===\n")
proc = subprocess.run(
    [sys.executable, "main.py", "--domain", "wikiart", "--config", "wikiart_config.json"],
    capture_output=True, text=True, timeout=600
)

lines = proc.stdout.split('\n') + proc.stderr.split('\n')

# Filter key events
key_patterns = [
    r'调度', r'Feedback', r'Reprocess', r'⟲', r'Verifier',
    r'完成', r'ERROR', r'exclude_relations', r'LP-ceiling',
    r'shared_style', r'pipeline_status', r'Think:.*action=',
    r'对抗性', r'检测', r'图质量', r'reprocess'
]

print("--- Key events ---")
for line in lines:
    for pat in key_patterns:
        if re.search(pat, line):
            print(line.strip())
            break

# Also search the full output
full = proc.stdout + proc.stderr
print(f"\n--- Full output total: {len(full)} chars / {len(lines)} lines ---")
print(f"Exit code: {proc.returncode}")

# Check for the verifier participation
if 'graph_verifier' in full.lower() or 'Verifier' in full:
    print("✓ Verifier agent mentioned in output")
else:
    print("✗ Verifier agent NOT found in output!")

if 'reprocess' in full.lower():
    print("✓ reprocess mentioned in output")
else:
    print("✗ reprocess NOT found in output!")

if 'exclude' in full.lower():
    print("✓ exclude mentioned in output")

# Final pipeline status
for line in lines:
    if 'pipeline_status' in line or '最终' in line or '成功完成' in line or 'failed' in line.lower():
        print(f"  Status: {line.strip()[:200]}")

print("\n=== End of test ===")
