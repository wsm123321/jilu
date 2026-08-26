"""Reconstruct and verify Gate 3 compressed raw tables from Git parts."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RESULTS=ROOT/'results_step3'
manifest=json.loads((RESULTS/'raw_parts_manifest.json').read_text(encoding='utf-8'))
for output_name,spec in manifest.items():
 output=RESULTS/output_name
 with output.open('wb') as target:
  for part in spec['parts']:
   path=RESULTS/'raw_parts'/part['file'];data=path.read_bytes()
   if len(data)!=part['bytes'] or hashlib.sha256(data).hexdigest()!=part['sha256']:raise SystemExit(f'Part verification failed: {path}')
   target.write(data)
 data=output.read_bytes()
 if len(data)!=spec['bytes'] or hashlib.sha256(data).hexdigest()!=spec['sha256']:raise SystemExit(f'Reconstructed archive verification failed: {output}')
 print(f'verified {output.name}: {len(data)} bytes')
