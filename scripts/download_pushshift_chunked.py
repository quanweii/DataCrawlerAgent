#!/usr/bin/env python3
"""Download Pushshift data in chunks via Range requests"""
import os, sys, time, requests, ssl
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

# Only download submissions first (5.2GB) - we need this more urgently
URL = "https://zenodo.org/api/records/3608135/files/RS_2019-04.zst/content"
TARGET = os.path.join(DATA_DIR, "RS_2019-04.zst")
TOTAL_SIZE = 5587265157  # from Zenodo metadata
CHUNK_SIZE = 50 * 1024 * 1024  # 50MB chunks
MAX_RETRIES = 10

session = requests.Session()
session.mount('https://', TLSAdapter())

# Get current progress
current_size = 0
if os.path.exists(TARGET):
    current_size = os.path.getsize(TARGET)

print(f"Starting chunked download: {current_size/(1024**2):.0f}MB / {TOTAL_SIZE/(1024**2):.0f}MB")

while current_size < TOTAL_SIZE:
    end_byte = min(current_size + CHUNK_SIZE - 1, TOTAL_SIZE - 1)
    
    for attempt in range(MAX_RETRIES):
        try:
            headers = {'Range': f'bytes={current_size}-{end_byte}'}
            r = session.get(URL, headers=headers, stream=True, timeout=120)
            
            if r.status_code in (200, 206):
                chunk_data = b''
                for block in r.iter_content(chunk_size=1024*1024):
                    chunk_data += block
                
                with open(TARGET, 'ab') as f:
                    f.write(chunk_data)
                
                current_size += len(chunk_data)
                pct = current_size / TOTAL_SIZE * 100
                print(f"  {pct:.1f}% - {current_size/(1024**2):.0f}MB / {TOTAL_SIZE/(1024**2):.0f}MB")
                break
            else:
                print(f"  Status {r.status_code}, retry {attempt+1}/{MAX_RETRIES}")
                time.sleep(3)
                
        except Exception as e:
            if attempt == 0:
                print(f"  Error: {str(e)[:80]}")
            time.sleep(5)
            # Re-create session on SSL errors
            session = requests.Session()
            session.mount('https://', TLSAdapter())
    else:
        print(f"  Failed chunk at {current_size/(1024**2):.0f}MB after {MAX_RETRIES} retries")
        time.sleep(10)
        # Continue trying next time
    
    if current_size >= TOTAL_SIZE:
        break

print(f"\nDownload complete: {current_size/(1024**2):.0f}MB")
