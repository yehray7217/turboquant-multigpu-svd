#!/bin/bash
set -euo pipefail

if ! command -v ncu >/dev/null 2>&1; then
    echo "ERROR: ncu not found. Run: module load cuda/12.8" >&2
    exit 1
fi

if [[ $# -eq 0 ]]; then
    set -- ncu_reports/*.ncu-rep
fi

OUT_DIR="ncu_text_summaries"
mkdir -p "${OUT_DIR}"

for report in "$@"; do
    if [[ ! -f "$report" ]]; then
        echo "Missing report: $report" >&2
        exit 1
    fi

    base="$(basename "$report" .ncu-rep)"
    full_text="${OUT_DIR}/${base}_details.txt"

    echo "====================================="
    echo "REPORT: $report"
    echo "====================================="
    ncu --import "$report" --page details >"${full_text}" 2>&1 || {
        echo "Failed to import report. Full output: ${full_text}"
        cat "${full_text}"
        exit 1
    }

    echo "Full details: ${full_text}"
    echo
    grep -Ei \
        "kernel name|duration|throughput|mem busy|max bandwidth|l1/tex|l2|dram|sm active|occupancy|eligible warps|issued warp|warp cycles|active threads|not predicated|block size|grid size|registers per thread|shared memory|waves per sm|branch|atomic|stall|scheduler|memory workload|launch statistics|speed of light|compute workload" \
        "${full_text}" || {
            echo "No compact matches found. First 120 lines of full details:"
            sed -n '1,120p' "${full_text}"
        }
    echo
done
