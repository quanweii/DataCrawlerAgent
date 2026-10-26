"""Install the versioned, hash-checked payload shipped with this repository."""
from pathlib import Path
import argparse,hashlib,json,tarfile,tempfile,os
ROOT=Path(__file__).resolve().parents[1]
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--group',choices=['datasets','evidence','all'],default='all')
 ap.add_argument('--verify-only',action='store_true',help='Check installed files without extracting.')
 ap.add_argument('--force',action='store_true',help='Replace existing files whose hashes differ.')
 args=ap.parse_args();manifest=json.loads((ROOT/'artifacts/manifest.json').read_text())
 groups=list(manifest['groups']) if args.group=='all' else [args.group]
 for name in groups:
  spec=manifest['groups'][name];expected={x['path']:x for x in spec['files']}
  if not args.verify_only:
   with tempfile.TemporaryDirectory(prefix='mag-forge-') as tmp:
    archive=Path(tmp)/spec['archive']
    with archive.open('wb') as f:
     for part in spec['parts']:
      p=ROOT/part['path']
      if not p.is_file() or sha(p)!=part['sha256']:raise ValueError('Missing or corrupt archive part: '+str(p))
      with p.open('rb') as src:
       for block in iter(lambda:src.read(1024*1024),b''):f.write(block)
    if sha(archive)!=spec['archive_sha256']:raise ValueError('Archive hash mismatch: '+name)
    print('Extracting',name,flush=True)
    with tarfile.open(archive) as tar:
     for member in tar:
      path=(ROOT/member.name).resolve()
      if not path.is_relative_to(ROOT) or not member.isfile() or member.name not in expected:raise ValueError('Unsafe archive member: '+member.name)
      record=expected[member.name]
      if path.exists():
       if sha(path)==record['sha256']:continue
       if not args.force:raise FileExistsError('Different local file; back it up or use --force: '+str(path))
      path.parent.mkdir(parents=True,exist_ok=True)
      with tar.extractfile(member) as src,path.open('wb') as dest:
       for block in iter(lambda:src.read(1024*1024),b''):dest.write(block)
  for rel,record in expected.items():
   p=ROOT/rel
   if not p.is_file() or sha(p)!=record['sha256']:raise ValueError('Installed artifact mismatch: '+rel)
  print(name+': verified '+str(len(expected))+' installed files',flush=True)
 print('Artifact release',manifest['release'],'verified.')
if __name__=='__main__':main()
