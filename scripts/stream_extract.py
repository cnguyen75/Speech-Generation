"""Stream the LibriTTS-R train-clean-100 tarball from stdin and keep only female readers.

Usage: curl -sL <url> | python -I scripts/stream_extract.py data/SPEAKERS.txt data/raw
Only .wav and .normalized.txt members whose path matches
LibriTTS_R/train-clean-100/<speaker>/<chapter>/<file> are written, under out/<speaker>/<file>.
"""
import re, sys, tarfile
from pathlib import Path

speakers_txt, out = Path(sys.argv[1]), Path(sys.argv[2])
female = set()
for line in speakers_txt.read_text().splitlines():
    parts = [p.strip() for p in line.split("|")]
    if len(parts) >= 3 and parts[1] == "F" and parts[2] == "train-clean-100":
        female.add(parts[0])
pat = re.compile(r"^LibriTTS_R/train-clean-100/(\d+)/(\d+)/([\w]+\.(?:wav|normalized\.txt))$")
n = 0
with tarfile.open(fileobj=sys.stdin.buffer, mode="r|gz") as tf:
    for m in tf:
        g = pat.match(m.name)
        if not (m.isfile() and g and g.group(1) in female):
            continue
        d = out / g.group(1)
        d.mkdir(parents=True, exist_ok=True)
        (d / g.group(3)).write_bytes(tf.extractfile(m).read())
        n += 1
        if n % 2000 == 0:
            print(f"{n} files", flush=True)
print(f"done: {n} files from {len(female)} female speakers", flush=True)
