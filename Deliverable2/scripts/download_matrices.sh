#!/bin/bash
#
# Download the real matrices used for strong scaling from SuiteSparse.
# The selection spans structured FEM matrices with fairly uniform row length,
# irregular matrices, and one power-law / scale-free graph that maximises the
# number of ghost entries (communication-heavy). This mirrors the Deliverable 1
# selection so the two studies use a consistent dataset.

MATRICES_DIR="./matrices"
mkdir -p "$MATRICES_DIR"

BASE_URL="https://suitesparse-collection-website.herokuapp.com/MM"

# name -> Group/Name
#   cant, consph, pdb1HYS, cop20k_A : FEM, structured, higher nnz/row
#   scircuit, mac_econ_fwd500       : irregular, short rows
#   webbase-1M                      : web graph, power-law (stresses ghosts)
declare -A MATRICES
MATRICES=(
    ["cant"]="Williams/cant"
    ["consph"]="Williams/consph"
    ["pdb1HYS"]="Williams/pdb1HYS"
    ["cop20k_A"]="Williams/cop20k_A"
    ["scircuit"]="Hamm/scircuit"
    ["mac_econ_fwd500"]="Williams/mac_econ_fwd500"
    ["webbase-1M"]="Williams/webbase-1M"
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
        echo "[OK] $name extracted"
    else
        echo "[WARN] $name.mtx not found after extraction"
    fi
done

echo ""
echo "Done. Contents of $MATRICES_DIR:"
ls -la "$MATRICES_DIR"/
