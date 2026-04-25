#!/bin/bash
#
# Download 10 matrices from SuiteSparse for the SpMV investigation.
# Selection criteria: mix of structured/unstructured, varying sizes,
# different nnz/row distributions, both regular and power-law patterns.
#
# The dataset is inspired by matrices used in Chu et al. (HPDC '23) [2]
# and Bell & Garland (SC '09) [3], with additions to cover a wider
# range of sparsity patterns.

MATRICES_DIR="./matrices"
mkdir -p "$MATRICES_DIR"

BASE_URL="https://suitesparse-collection-website.herokuapp.com/MM"

# Matrix list: Group/Name
# 1. cage4          — small, very regular (DNA electrophoresis)
# 2. olm1000        — small, banded structure
# 3. west2021       — medium, chemical engineering, irregular
# 4. mac_econ_fwd500 — medium, economics model, rectangular-ish
# 5. cant           — FEM, regular structure, ~64 nnz/row
# 6. consph         — FEM, 3D sphere, regular
# 7. cop20k_A       — FEM, accelerator design, moderate irregularity
# 8. pdb1HYS        — protein data bank, moderate irregularity
# 9. webbase-1M     — web graph, power-law distribution
# 10. scircuit      — circuit simulation, highly irregular

declare -A MATRICES
MATRICES=(
    ["cage4"]="vanHeukelum/cage4"
    ["olm1000"]="Bai/olm1000"
    ["west2021"]="HB/west2021"
    ["mac_econ_fwd500"]="Williams/mac_econ_fwd500"
    ["cant"]="Williams/cant"
    ["consph"]="Williams/consph"
    ["cop20k_A"]="Williams/cop20k_A"
    ["pdb1HYS"]="Williams/pdb1HYS"
    ["webbase-1M"]="Williams/webbase-1M"
    ["scircuit"]="Hamm/scircuit"
)

for name in "${!MATRICES[@]}"; do
    group_name="${MATRICES[$name]}"
    tarball="${name}.tar.gz"
    url="${BASE_URL}/${group_name}.tar.gz"

    if [ -f "$MATRICES_DIR/$name/$name.mtx" ]; then
        echo "[SKIP] $name already downloaded"
        continue
    fi

    echo "[DOWNLOAD] $name from $url"
    wget -q -O "$MATRICES_DIR/$tarball" "$url"

    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to download $name"
        rm -f "$MATRICES_DIR/$tarball"
        continue
    fi

    cd "$MATRICES_DIR"
    tar -xzf "$tarball" 2>/dev/null || (gzip -d "$tarball" && tar -xf "${tarball%.gz}")
    rm -f "$tarball" "${tarball%.gz}"
    cd ..

    if [ -f "$MATRICES_DIR/$name/$name.mtx" ]; then
        echo "[OK] $name extracted successfully"
    else
        echo "[WARN] $name.mtx not found after extraction, check directory structure"
    fi
done

echo ""
echo "Matrix download complete. Contents of $MATRICES_DIR:"
ls -la "$MATRICES_DIR"/
