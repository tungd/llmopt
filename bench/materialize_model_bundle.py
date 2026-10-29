"""Materialize a compiled model's serving assets without duplicating its archive.

Moves the supplied owned archive/tokenizer into the model directory, converts
entrypoint weight links to relative links, and installs the serving executable.
The result is a portable directory; it is not a single executable with embedded
weights. Run only after compilation has finished and before inference starts.
"""
import argparse,json,os,shutil
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--engine',type=Path,required=True);p.add_argument('--weights',type=Path,required=True)
p.add_argument('--tokenizer',type=Path,required=True);p.add_argument('--serve',type=Path,required=True)
a=p.parse_args();root=a.engine.resolve()
if not (root/'model.llmopt').is_file(): raise ValueError('compiled Model Program missing')
for source,name in ((a.weights,'weights.llmopt'),(a.tokenizer,'tokenizer.llmopt')):
 source=source.resolve();target=root/name
 if target.is_symlink():
  if target.resolve()!=source:raise ValueError('asset link points to an unexpected file')
  target.unlink()
 elif target.exists():raise ValueError('refusing to overwrite materialized asset')
 os.replace(source,target)
for path in root.rglob('weights.llmopt'):
 if path==root/'weights.llmopt':continue
 if not path.is_symlink():raise ValueError('entrypoint weights must be links before materialization')
 path.unlink();path.symlink_to(os.path.relpath(root/'weights.llmopt',path.parent))
shutil.copy2(a.serve,root/'llmopt-serve')
launch=root/'serve';launch.write_text('#!/bin/sh\nset -eu\nbundle=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)\nexec "$bundle/llmopt-serve" --model-dir "$bundle" "$@"\n');launch.chmod(0o755)
for path in root.rglob('*'):
 if path.is_symlink():
  resolved=path.resolve(strict=True)
  if not resolved.is_relative_to(root):raise ValueError('external bundle dependency: '+str(path))
print(json.dumps({'status':'passed','bundle':str(root),'entrypoint':str(launch),'format':'directory with Model Program, EXL3 archive, tokenizer, shaders, serving executable'}))
