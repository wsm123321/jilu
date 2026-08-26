"""Reconstruct and verify Gate 4-R compressed long tables."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results_step4r';manifest=json.loads((OUT/'raw_parts_manifest.json').read_text(encoding='utf-8'))['archives']
for name,spec in manifest.items():
 target=OUT/name
 with target.open('wb') as fo:
  for part in spec['parts']:
   path=OUT/'raw_parts'/part['file'];data=path.read_bytes()
   if len(data)!=part['bytes'] or hashlib.sha256(data).hexdigest()!=part['sha256']:raise SystemExit(f'part verification failed: {path}')
   fo.write(data)
 data=target.read_bytes()
 if len(data)!=spec['archive_bytes'] or hashlib.sha256(data).hexdigest()!=spec['archive_sha256']:raise SystemExit(f'archive verification failed: {name}')
 print(f'verified {name}: {len(data)} bytes')
