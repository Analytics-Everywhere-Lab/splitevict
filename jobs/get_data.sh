#!/usr/bin/env bash
# Download the data archive, check its SHA-256 sums and unpack it into the repository.
#   bash jobs/get_data.sh          (DATA_URL=<other location> to use a copy elsewhere)
set -euo pipefail
cd "$(dirname "$0")/.."
DATA_URL=${DATA_URL:-https://huggingface.co/datasets/noctra-voidling/splitevict-data/resolve/main}
for part in data.tar.gz measurement-data.tar.gz; do
  [ -f "$part" ] || curl -L --fail -o "$part" "$DATA_URL/$part"
done
if command -v sha256sum >/dev/null; then sha256sum -c jobs/data.sha256; else shasum -a 256 -c jobs/data.sha256; fi
for part in data.tar.gz measurement-data.tar.gz; do tar -xzf "$part"; done
echo "data unpacked"
