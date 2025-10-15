#!/usr/bin/env bash
set -euo pipefail

for PKG in xformers flash-attn; do
  pip uninstall -y "$PKG" || true
  python3 - << 'PY'
import site, sys, shutil, os
pkgs = ["xformers", "flash_attn"]
paths = set()
if hasattr(site, 'getusersitepackages'):
    usp = site.getusersitepackages()
    if isinstance(usp, str):
        paths.add(usp)
    else:
        paths.update(usp)
for p in sys.path:
    if 'site-packages' in p:
        paths.add(p)
for root in list(paths):
    if not os.path.isdir(root):
        continue
    for name in list(os.listdir(root)):
        for pkg in pkgs:
            if name.startswith(pkg):
                try:
                    shutil.rmtree(os.path.join(root, name), ignore_errors=True)
                except Exception:
                    pass
PY
done

echo "Cleaned conflicting xformers/flash-attn installs."

#!/usr/bin/env bash
set -euo pipefail

# Remove possibly conflicting global/user installs that may shadow env packages
for PKG in xformers flash-attn; do
  pip uninstall -y "$PKG" || true
  python3 - << 'PY'
import site, sys, shutil
import os
pkgs = ["xformers", "flash_attn"]
paths = set(site.getusersitepackages().split(os.pathsep) if hasattr(site, 'getusersitepackages') else [])
for p in sys.path:
    if 'site-packages' in p:
        paths.add(p)
for root in list(paths):
    if not os.path.isdir(root):
        continue
    for name in list(os.listdir(root)):
        for pkg in pkgs:
            if name.startswith(pkg):
                try:
                    shutil.rmtree(os.path.join(root, name), ignore_errors=True)
                except Exception:
                    pass
PY
done

echo "Cleaned conflicting xformers/flash-attn installs."


