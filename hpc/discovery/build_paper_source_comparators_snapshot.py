#!/usr/bin/env python3
"""Create an isolated upload archive without modifying earlier campaign sources."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--staging', required=True, type=Path)
    parser.add_argument('--remote-root', required=True)
    args = parser.parse_args()
    code=Path(__file__).resolve().parents[2]
    args.staging.mkdir(parents=True, exist_ok=False)
    source=args.staging/'source'
    source.mkdir()
    paths=set()
    for relative in ('sparse-smart/src','sparse-smart-v2/src','smart/smart','simulation/paper_source_comparators_20260913'):
        paths.update(p for p in (code/relative).rglob('*.py') if '__pycache__' not in p.parts)
    for pattern in ('paper_source_comparators_*','submit_paper_source_comparators.py','build_paper_source_comparators_snapshot.py'):
        paths.update((code/'hpc/discovery').glob(pattern))
    for relative in ('.python-version','python-constraints.txt','environment/check_runtime.py','hpc/discovery/env.sh',
                     'hpc/discovery/submit_sparse_smart_v2_campaign.py','simulation/run_sparse_smart_v2_pilot.py',
                     'simulation/run_restricted_rrr.py','simulation/data/random_seeds/experiment_seeds.csv',
                     'simulation/reviewer_revision_20260913/__init__.py','simulation/reviewer_revision_20260913/competitors.py'):
        paths.add(code/relative)
    hashes={}
    for path in sorted(paths):
        relative=path.relative_to(code)
        target=source/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target)
        hashes[relative.as_posix()]=hashlib.sha256(target.read_bytes()).hexdigest()
    manifest=dict(schema_version=1,source_root=args.remote_root+'/source',files=hashes,
                  purpose='Original paper model/experiment source-comparator campaign')
    (args.staging/'source-manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
    archive=args.staging.with_suffix('.tar.gz')
    with tarfile.open(archive,'w:gz') as t:
        t.add(source,arcname='source')
        t.add(args.staging/'source-manifest.json',arcname='source-manifest.json')
    print(json.dumps(dict(archive=str(archive),files=len(hashes),sha256=hashlib.sha256(archive.read_bytes()).hexdigest())))

if __name__=='__main__':
    main()
