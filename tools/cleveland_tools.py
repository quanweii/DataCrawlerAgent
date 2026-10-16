"""Small, cached CC0 source adapter for the production DomainCollectorAgent.

Only HTTP acquisition, field normalization and ID-aligned text/image packaging
are source-specific. Graph construction, audits, CLIP and tasks remain shared.
"""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import hashlib,html,io,json,re,time,faulthandler,threading
from concurrent.futures import ThreadPoolExecutor
import requests
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
TYPES=['Painting','Print','Drawing','Sculpture','Ceramic','Textile']
PER_TYPE=150
API='https://openaccess-api.clevelandart.org/api/artworks/'
FIELDS='id,accession_number,title,type,creators,culture,creation_date,creation_date_earliest,creation_date_latest,description,fun_fact,share_license_status,images,url,updated_at'

def sha(data):return hashlib.sha256(data).hexdigest()
def text(value):
    return re.sub(r'\s+',' ',html.unescape(re.sub(r'<[^>]+>',' ',str(value or '')))).strip()
def write(path,obj):
    path.parent.mkdir(exist_ok=True,parents=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8');temp.replace(path)

class ClevelandToolManager:
    def __init__(self,data_dir='./data/cleveland_raw'):
        self.raw=ROOT/data_dir if not Path(data_dir).is_absolute() else Path(data_dir)
        self.cslo=ROOT/'data/cleveland_cslo'
        self.raw.mkdir(exist_ok=True,parents=True);self.cslo.mkdir(exist_ok=True,parents=True)
        self.sessions=threading.local();self.lock=threading.Lock()
        self.last=0.
        log=self.raw/'request_log.json'
        self.requests=json.loads(log.read_text()) if log.exists() else []

    def get(self,url,params=None):
        for attempt in range(3):
            with self.lock:
                time.sleep(max(0,1.05-(time.monotonic()-self.last)))
                self.last=time.monotonic()
            start=time.monotonic()
            try:
                if not hasattr(self.sessions,'client'):self.sessions.client=requests.Session()
                r=self.sessions.client.get(url,params=params,timeout=(10,25))
                with self.lock:
                    self.requests.append(dict(url=r.url,status=r.status_code,bytes=len(r.content),attempt=attempt+1,
                        utc=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-start))
                    write(self.raw/'request_log.json',self.requests)
                if r.status_code in [429,500,502,503,504]:time.sleep(2**attempt);continue
                r.raise_for_status();return r
            except requests.RequestException:
                if attempt==2:raise
                time.sleep(2**attempt)
        raise RuntimeError('HTTP retries exhausted')

    def load_raw(self,max_clevelands=900,**kwargs):
        cap=int(kwargs.get('max_items',max_clevelands))
        final=self.raw/'cleveland_items.json'
        if final.exists():return json.loads(final.read_text(encoding='utf-8'))[:cap]
        start=time.monotonic();items=[];rejected=[];image_hashes={};text_hashes={};source_counts={}
        for kind in TYPES:
            cache=self.raw/'responses'/(kind.lower()+'.json')
            if cache.exists():obj=json.loads(cache.read_text(encoding='utf-8'))
            else:
                r=self.get(API,params=dict(cc0=1,has_image=1,type=kind,limit=300))
                obj=r.json();write(cache,obj)
            source_counts[kind]=obj['info']['total'];kept=0
            # Four bounded transfers hide network latency; one shared limiter
            # permits at most one request start per second across all workers.
            def prefetch(a):
                url=a.get('images',{}).get('web',{}).get('url')
                if a.get('share_license_status')!='CC0' or not url:return
                p=self.cslo/str(a['id'])/'images'/'primary.jpg'
                if p.exists():return
                try:
                    content=self.get(url).content;Image.open(io.BytesIO(content)).load()
                    p.parent.mkdir(exist_ok=True,parents=True);p.write_bytes(content)
                except Exception as e:print('PREFETCH FAILED',a['id'],type(e).__name__,flush=True)
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(prefetch,obj['data'][:175]))
            for a in obj['data']:
                if kept>=PER_TYPE:break
                nid=str(a['id']);image=a.get('images',{}).get('web',{}).get('url')
                if a.get('share_license_status')!='CC0' or not image or a.get('type')!=kind:
                    rejected.append(dict(id=nid,reason='rights_or_image_or_type'));continue
                d=self.cslo/nid;(d/'images').mkdir(exist_ok=True,parents=True)
                img=d/'images'/'primary.jpg'
                try:
                    payload=img.read_bytes() if img.exists() else self.get(image).content
                    im=Image.open(io.BytesIO(payload));im.load()
                    if min(im.size)<32:raise ValueError('image too small')
                    image_sha=sha(payload)
                    if image_sha in image_hashes:
                        rejected.append(dict(id=nid,reason='duplicate_image',matches=image_hashes[image_sha]));continue
                    cleaned='Artwork: '+text(a['title'])+'\n\n'+text(a.get('description'))
                    if a.get('fun_fact'):cleaned+='\n\n'+text(a['fun_fact'])
                    cleaned=cleaned.strip()+'\n';text_sha=sha(cleaned.encode())
                    # Exact repeated title-only text is allowed and disclosed;
                    # exact repeated images are removed for retrieval suitability.
                    img.write_bytes(payload);(d/(nid+'.md')).write_text(cleaned,encoding='utf-8')
                    creators=[str(c['id']) for c in a.get('creators',[]) if c.get('id') is not None]
                    year=a.get('creation_date_earliest')
                    item=dict(id=nid,title=text(a['title']),type=kind,creator_ids=creators,
                        culture=[text(c) for c in a.get('culture',[]) if text(c)],
                        # An earliest creation-year bound is not an observation timestamp.
                        creation_year=year,description=text(a.get('description')),
                        image_url=image,object_url=a.get('url'),license='CC0',
                        image_sha256=image_sha,text_sha256=text_sha,
                        source_response=str(cache.relative_to(ROOT)),
                        image_dimensions=list(im.size),
                        text_source_fields=['title','description','fun_fact'],
                        has_description=bool(text(a.get('description'))))
                    items.append(item);kept+=1;image_hashes[image_sha]=nid;text_hashes.setdefault(text_sha,[]).append(nid)
                    write(self.raw/'partial_items.json',items)
                    if kept%25==0:print(kind,'accepted',kept,'total',len(items),flush=True)
                except Exception as e:
                    rejected.append(dict(id=nid,reason=type(e).__name__,message=str(e)[:160]))
                with self.lock:write(self.raw/'request_log.json',self.requests)
                write(self.raw/'rejections.json',rejected)
            if kept!=PER_TYPE:raise RuntimeError(f'{kind}: only {kept} accepted; sample must not be silently changed')
        # Ordering is frozen before partitioning/training; selected by source
        # display order and class quota, not by downstream model results.
        write(final,items)
        manifest=dict(source='Cleveland Museum of Art Open Access',documentation='https://openaccess-api.clevelandart.org/',
            request='Construct a text-image MAG for object-type classification with creator/culture context.',
            sampling='first 150 accepted CC0/image records per each of six predeclared types in API return order',
            target='type',relations=['creator_ids','culture'],types=TYPES,class_counts=dict(Counter(a['type'] for a in items)),
            source_eligible_counts=source_counts,items=len(items),real_images=len(items),
            described=sum(a['has_description'] for a in items),
            text_word_count_median=sorted(len((self.cslo/a['id']/(a['id']+'.md')).read_text().split()) for a in items)[len(items)//2],
            exact_repeated_text_groups=[ids for ids in text_hashes.values() if len(ids)>1],
            unique_image_hashes=len(image_hashes),rejected=len(rejected),seconds=time.monotonic()-start,
            source_specific_code='tools/cleveland_tools.py; YAML; case runner',
            metadata_and_images_cc0=True,api_requests=len(self.requests),
            created_utc=datetime.now(timezone.utc).isoformat())
        write(self.raw/'acquisition_manifest.json',manifest)
        print('ACQUISITION COMPLETE',json.dumps(manifest),flush=True)
        return items[:cap]

if __name__=='__main__':
    faulthandler.dump_traceback_later(120,repeat=True)
    ClevelandToolManager().load_raw()
