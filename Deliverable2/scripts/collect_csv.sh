#!/bin/bash
# Merge the per-job CSVs into two combined files for the plotting script.
# Keeps a single header line and appends all data rows.
set -e
cd "$(dirname "$0")/.."
mkdir -p outputs

merge() {
    local pattern="$1" out="$2"
    local files=(outputs/${pattern})
    [ -e "${files[0]}" ] || { echo "no files matching outputs/${pattern}"; return; }
    head -n 1 "${files[0]}" > "$out"
    for f in "${files[@]}"; do
        tail -n +2 "$f" >> "$out"
    done
    echo "wrote $out ($(($(wc -l < "$out") - 1)) rows)"
}

merge "strong_*_*.csv" outputs/strong_all.csv
merge "weak_*.csv"     outputs/weak_all.csv
