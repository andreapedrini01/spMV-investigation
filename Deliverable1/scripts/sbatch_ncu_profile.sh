#!/bin/bash
#SBATCH --partition=edu-short
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00

#SBATCH --job-name=spmv_ncu
#SBATCH --output=outputs/ncu_profile_%j.out
#SBATCH --error=outputs/ncu_profile_%j.err

module load CUDA/11.8.0

mkdir -p outputs

# Build if needed
make -q || make

# Profile 3 representative matrices:
#   - mac_econ:   short rows (6.2 nnz/row), CSR-Scalar wins
#   - cant:       long rows (64.2 nnz/row), CSR-Vector wins
#   - webbase-1M: power-law (3.1 nnz/row), both struggle

MATRICES_DIR="./matrices"

# Metrics we care about:
#   l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum   — global load sectors (L1 traffic)
#   l1tex__t_sector_hit_rate.pct                     — L1 cache hit rate
#   lts__t_sector_hit_rate.pct                       — L2 cache hit rate
#   dram__bytes_read.sum                             — DRAM bytes actually read
#   sm__throughput.avg.pct_of_peak_sustained_elapsed — SM throughput utilization

METRICS="l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct,dram__bytes_read.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed"

for mtx_name in cant; do
    mtx_file="$MATRICES_DIR/$mtx_name/$mtx_name.mtx"

    if [ ! -f "$mtx_file" ]; then
        echo "[SKIP] $mtx_file not found"
        continue
    fi

    echo "=========================================="
    echo "Profiling: $mtx_name"
    echo "=========================================="

    # ncu profiles only the first kernel launch by default.
    # --launch-count 1 --launch-skip 4 skips the 4 warmup launches and profiles the 1st measured run.
    # --kernel-name regex matches our custom kernels (not cuSPARSE internals).
    # We run once per kernel by setting an env var — but since our binary runs all kernels,
    # we just profile all kernel launches and filter later.
    # Using --launch-skip 0 --launch-count 20 captures warmup+measured for all kernels.
    # Simpler: just profile everything and grep the output.

    ncu --metrics "$METRICS" \
        --csv \
        --page raw \
        ./bin/spmv "$mtx_file" \
        2>&1 | tee "outputs/ncu_${mtx_name}.csv"

    echo ""
done

echo "=========================================="
echo "NCU profiling complete."
echo "=========================================="
