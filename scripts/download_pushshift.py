#!/usr/bin/env python3
"""Download Pushshift Reddit data from Zenodo with retry logic"""
import os
import sys
import time
import requests
import ssl
from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context

class TLSAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        ctx = create_urllib3_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        kwargs['ssl_context'] = ctx
        return super().init_poolmanager(*args, **kwargs)

DATA_DIR = "./data/reddit_raw"
os.makedirs(DATA_DIR, exist_ok=True)

FILES = {
    "RS_2019-04.zst": "https://zenodo.org/api/records/3608135/files/RS_2019-04.zst/content",
    "RC_2019-04.zst": "https://zenodo.org/api/records/3608135/files/RC_2019-04.zst/content",
}

session = requests.Session()
session.mount('https://', TLSAdapter())

for filename, url in FILES.items():
    target = os.path.join(DATA_DIR, filename)
    
    if os.path.exists(target) and os.path.getsize(target) > 100_000_000:
        print(f"[SKIP] {filename} already downloaded ({os.path.getsize(target)/(1024**2):.0f} MB)")
        continue
    
    print(f"\n[DOWNLOAD] {filename}...")
    
    # Get current position for resume
    resume_from = 0
    if os.path.exists(target):
        resume_from = os.path.getsize(target)
    
    for attempt in range(5):
        try:
            headers = {}
            if resume_from > 0:
                headers['Range'] = f'bytes={resume_from}-'
                mode = 'ab'
            else:
                mode = 'wb'
            
            r = session.get(url, stream=True, timeout=60, headers=headers)
            
            if r.status_code == 200:
                total = resume_from
                with open(target, mode) as f:
                    for chunk in r.iter_content(chunk_size=512*1024):
                        if chunk:
                            f.write(chunk)
                            total += len(chunk)
                            if total % (200*1024*1024) < 512*1024:
                                print(f"  {filename}: {total/(1024**2):.0f} MB")
                
                print(f"[DONE] {filename}: {total/(1024**2):.0f} MB")
                break
            elif r.status_code == 416:
                print(f"[DONE] {filename} already complete")
                break
            else:
                print(f"[RETRY] {filename} status={r.status_code}, attempt {attempt+1}/5")
                resume_from = os.path.getsize(target) if os.path.exists(target) else 0
                time.sleep(5)
                
        except Exception as e:
            print(f"[ERROR] {filename} attempt {attempt+1}/5: {e}")
            resume_from = os.path.getsize(target) if os.path.exists(target) else 0
            time.sleep(10)

print("\n[COMPLETE] All downloads finished")
