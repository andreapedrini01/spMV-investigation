#!/bin/bash
#SBATCH --partition=edu-short
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00

#SBATCH --job-name=spmv_d1
#SBATCH --output=outputs/spmv_%j.out
#SBATCH --error=outputs/spmv_%j.err

module load CUDA/11.8.0

mkdir -p outputs

# Build
make clean
make

MATRICES_DIR="./matrices"

echo "=========================================="
echo "SpMV Deliverable 1 — Benchmark Run"
echo "Date: $(date)"
echo "=========================================="

for mtx_dir in "$MATRICES_DIR"/*/; do
    mtx_name=$(basename "$mtx_dir")
    mtx_file="$mtx_dir/${mtx_name}.mtx"

    if [ ! -f "$mtx_file" ]; then
        echo "[SKIP] No .mtx file found in $mtx_dir"
        continue
    fi

    echo ""
    echo "=========================================="
    echo "Matrix: $mtx_name"
    echo "File:   $mtx_file"
    echo "=========================================="
    ./bin/spmv "$mtx_file"
    echo ""
done

echo "=========================================="
echo "All matrices processed."
echo "=========================================="
