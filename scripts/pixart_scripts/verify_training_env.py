#!/usr/bin/env python3
import sys

mods = [
    ("torch", "__version__"),
    ("diffusers", "__version__"),
    ("transformers", "__version__"),
    ("accelerate", "__version__"),
    ("datasets", "__version__"),
    ("peft", "__version__"),
]

ok = True
for name, attr in mods:
    try:
        m = __import__(name)
        v = getattr(m, attr, "?")
        print(f"OK {name}: {v}")
    except Exception as e:
        ok = False
        print(f"FAIL {name}: {e}")

sys.exit(0 if ok else 1)
