"""Copy an immutable, hashed source snapshot without touching other campaigns."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile


def freeze(destination):
    code = Path(__file__).resolve().parents[2]
    root = Path(destination).resolve()
    if root.exists():
        raise FileExistsError(root)
    source=root/'source'
    source.mkdir(parents=True)
    paths = ['sparse-smart/src','sparse-smart-v2/src','simulation/reviewer_revision_20260913',
             'environment/check_runtime.py','hpc/discovery/env.sh','.python-version','python-constraints.txt']
    for relative in paths:
        src=code/relative; dst=source/relative
        dst.parent.mkdir(parents=True,exist_ok=True)
        if src.is_dir():
            shutil.copytree(src,dst,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache','*.pyc'))
        else:
            shutil.copy2(src,dst)
    files={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(source.rglob('*')) if p.is_file()}
    manifest={'schema':1,'files':files,
        'code_git_head':subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip(),
        'source_fingerprint':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()}
    (root/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with tarfile.open(str(root)+'.tar.gz','w:gz') as tar:
        tar.add(root,arcname=root.name)
    print(json.dumps({'root':str(root),'archive':str(root)+'.tar.gz',**{k:v for k,v in manifest.items() if k!='files'}}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('destination')
    freeze(parser.parse_args().destination)
