"""Deterministically package Gate 4-R long tables into verifiable Git parts."""
from __future__ import annotations
import gzip,hashlib,json,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results_step4r';PARTS=OUT/'raw_parts';PARTS.mkdir(exist_ok=True)
SOURCES=('development_raw.csv','holdout_classified.csv','unseen_classified.csv')
manifest={}
for name in SOURCES:
 src=OUT/name;archive=OUT/(name+'.gz')
 with src.open('rb') as fi,gzip.GzipFile(filename='',mode='wb',fileobj=archive.open('wb'),compresslevel=9,mtime=0) as fo:shutil.copyfileobj(fi,fo)
 data=archive.read_bytes();parts=[]
 for i,start in enumerate(range(0,len(data),2_000_000)):
  chunk=data[start:start+2_000_000];path=PARTS/f'{archive.name}.part{i:03d}';path.write_bytes(chunk);parts.append({'file':path.name,'bytes':len(chunk),'sha256':hashlib.sha256(chunk).hexdigest()})
 manifest[archive.name]={'source_csv_bytes':src.stat().st_size,'archive_bytes':len(data),'archive_sha256':hashlib.sha256(data).hexdigest(),'parts':parts}
(OUT/'raw_parts_manifest.json').write_text(json.dumps({'packaging_git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'archives':manifest},indent=2),encoding='utf-8')
print({k:len(v['parts']) for k,v in manifest.items()})
