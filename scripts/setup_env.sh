#!/bin/sh
# Environment setup: LAME (test-case generation only), Python deps, C build.
set -e
cd "$(dirname "$0")/.."
if ! command -v lame >/dev/null 2>&1; then
    apt-get install -y lame >/dev/null
fi
pip3 install -q numpy scipy pytest
make
