"""Verify binary part manifests and LF-normalized text bundle manifests."""
from __future__ import annotations
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(data):return hashlib.sha256(data).hexdigest()
def verify_parts(directory):
 path=directory/'raw_parts_manifest.json'
 if not path.exists():return
 raw=json.loads(path.read_text(encoding='utf-8'));archives=raw.get('archives',raw)
 for name,spec in archives.items():
  data=b''
  for part in spec['parts']:
   p=directory/'raw_parts'/part['file'];chunk=p.read_bytes()
   if len(chunk)!=part['bytes'] or sha(chunk)!=part['sha256']:raise AssertionError(f'bad part {p}')
   data+=chunk
  expected_bytes=spec.get('archive_bytes',spec.get('bytes'));expected_hash=spec.get('archive_sha256',spec.get('sha256'))
  if len(data)!=expected_bytes or sha(data)!=expected_hash:raise AssertionError(f'bad archive {name}')
def verify_bundle(directory):
 path=directory/'bundle_manifest.json'
 if not path.exists():return
 m=json.loads(path.read_text(encoding='utf-8'))
 for name,spec in m['files'].items():
  p=directory/name;data=p.read_bytes().replace(b'\r\n',b'\n')
  if len(data)!=spec['bytes'] or sha(data)!=spec['sha256']:raise AssertionError(f'bad LF-normalized artifact {p}')
def main():
 for name in ('results_step3','results_step4','results_step4r'):
  d=ROOT/name
  if d.exists():verify_parts(d);verify_bundle(d)
 print('all manifests verified')
if __name__=='__main__':main()
