#!/bin/sh
# mirror data/*.json to data/*.js so the page also works from file:// (no server needed)
cd "$(dirname "$0")" && { printf 'window.RESULTS = '; cat data/results.json; printf ';\n'; } > data/results.js && { printf 'window.PRESETS_DATA = '; cat data/presets.json; printf ';\n'; } > data/presets.js && { printf 'window.TRACE2 = '; cat data/trace_v2.json; printf ';\n'; } > data/trace_v2.js
