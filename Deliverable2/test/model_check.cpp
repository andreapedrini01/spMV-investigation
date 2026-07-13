/*
 * Host-only logic check for the distributed SpMV index math. It does NOT use
 * CUDA or MPI: it reimplements, verbatim, the index expressions from
 * distributed.cu and models the full P-rank algorithm in a single process,
 * then checks the distributed result against a serial SpMV. The point is to
 * catch cyclic-distribution / ghost-remap / gather-reorder bugs locally before
 * spending cluster time. Compile: g++ -O2 -std=c++14 model_check.cpp -o mc
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
using std::vector;

/* ---- verbatim from utils.h / distributed.h ---- */
static float xvec_value(int j)
{
    unsigned int h = (unsigned int)j * 2654435761u;
    return (float)(h & 0x00FFFFFFu) / (float)0x01000000u;
}
static int count_cyclic(int total, int P, int rank)
{
    return total / P + (rank < (total % P) ? 1 : 0);
}

struct COO { int M, N; vector<int> r, c; vector<float> v; };
struct LCSR { int n_local, N, nnz; vector<int> row_ptr, col; vector<float> val; };

static COO make_matrix(int M, int N, unsigned seed)
{
    COO A; A.M = M; A.N = N;
    unsigned st = seed ? seed : 1u;
    auto rnd = [&]() { st ^= st << 13; st ^= st >> 17; st ^= st << 5; return st; };
    for (int i = 0; i < M; i++) {
        if (i < N) { A.r.push_back(i); A.c.push_back(i); A.v.push_back(1.0f + (float)(rnd() & 255) / 256.0f); }
        int extra = 2 + (int)(rnd() % 6);
        for (int t = 0; t < extra; t++) {
            int col = (int)(rnd() % (unsigned)N);
            A.r.push_back(i); A.c.push_back(col); A.v.push_back(0.5f + (float)(rnd() & 255) / 256.0f);
        }
    }
    return A;
}

static vector<float> serial_spmv(const COO &A)
{
    vector<float> y(A.M, 0.0f);
    for (size_t e = 0; e < A.r.size(); e++)
        y[A.r[e]] += A.v[e] * xvec_value(A.c[e]);
    return y;
}

/* mirrors distribute_matrix_cyclic + build_local_csr for one rank */
static LCSR distribute(const COO &A, int P, int rank)
{
    vector<int> lr, lc; vector<float> lv;
    for (size_t e = 0; e < A.r.size(); e++)
        if (A.r[e] % P == rank) { lr.push_back(A.r[e]); lc.push_back(A.c[e]); lv.push_back(A.v[e]); }

    LCSR m; m.N = A.N; m.n_local = count_cyclic(A.M, P, rank); m.nnz = (int)lr.size();
    m.row_ptr.assign(m.n_local + 1, 0);
    for (size_t e = 0; e < lr.size(); e++) { int k = lr[e] / P; m.row_ptr[k + 1]++; }
    for (int k = 0; k < m.n_local; k++) m.row_ptr[k + 1] += m.row_ptr[k];
    m.col.resize(lr.size()); m.val.resize(lr.size());
    vector<int> cur(m.row_ptr.begin(), m.row_ptr.end());
    for (size_t e = 0; e < lr.size(); e++) { int k = lr[e] / P; int d = cur[k]++; m.col[d] = lc[e]; m.val[d] = lv[e]; }
    return m;
}

static double max_abs_diff(const vector<float> &a, const vector<float> &b)
{
    double m = 0.0;
    for (size_t i = 0; i < a.size(); i++) m = std::max(m, (double)std::fabs(a[i] - b[i]));
    return m;
}

/* returns 0 on success */
static int check_config(int M, int N, int P, unsigned seed, int verbose)
{
    COO A = make_matrix(M, N, seed);
    vector<float> yref = serial_spmv(A);

    /* recvcounts/displs and the Allgather reorder permutation (Mode A) */
    vector<int> recvcounts(P), displs(P);
    for (int p = 0; p < P; p++) recvcounts[p] = count_cyclic(N, P, p);
    displs[0] = 0;
    for (int p = 1; p < P; p++) displs[p] = displs[p - 1] + recvcounts[p - 1];
    if (displs[P - 1] + recvcounts[P - 1] != N) { printf("  [FAIL] recvcounts sum != N\n"); return 1; }

    vector<int> perm(N);
    for (int p = 0; p < P; p++)
        for (int mm = 0; mm < recvcounts[p]; mm++)
            perm[displs[p] + mm] = p + mm * P;

    /* x_gathered is the rank-major concatenation of each rank's owned x */
    vector<float> xg(N);
    for (int p = 0; p < P; p++)
        for (int mm = 0; mm < recvcounts[p]; mm++)
            xg[displs[p] + mm] = xvec_value(p + mm * P);
    vector<float> xfull(N);
    for (int q = 0; q < N; q++) xfull[perm[q]] = xg[q];
    for (int j = 0; j < N; j++)
        if (std::fabs(xfull[j] - xvec_value(j)) > 1e-6f) { printf("  [FAIL] allgather reorder at j=%d\n", j); return 1; }

    /* NCCL padded (strided) layout reorder */
    int maxnx = (N + P - 1) / P;
    vector<int> srcpos(N);
    for (int p = 0; p < P; p++)
        for (int mm = 0; mm < recvcounts[p]; mm++)
            srcpos[displs[p] + mm] = p * maxnx + mm;
    vector<float> G((size_t)P * maxnx, -1e30f);
    for (int p = 0; p < P; p++)
        for (int mm = 0; mm < recvcounts[p]; mm++)
            G[(size_t)p * maxnx + mm] = xvec_value(p + mm * P);
    vector<float> xfull_n(N);
    for (int q = 0; q < N; q++) xfull_n[perm[q]] = G[srcpos[q]];
    for (int j = 0; j < N; j++)
        if (std::fabs(xfull_n[j] - xvec_value(j)) > 1e-6f) { printf("  [FAIL] nccl reorder at j=%d\n", j); return 1; }

    /* Mode A: per-rank local SpMV against the reconstructed full x */
    long long nnz_sum = 0;
    vector<float> yA(M, 0.0f), yB(M, 0.0f);
    for (int rank = 0; rank < P; rank++) {
        LCSR m = distribute(A, P, rank);
        nnz_sum += m.nnz;
        for (int k = 0; k < m.n_local; k++) {
            float s = 0.0f;
            for (int e = m.row_ptr[k]; e < m.row_ptr[k + 1]; e++)
                s += m.val[e] * xfull[m.col[e]];      /* global col into full x */
            yA[rank + k * P] = s;                     /* cyclic row placement */
        }

        /* Mode B: ghost setup + remap + modeled exchange */
        int n_owned = count_cyclic(N, P, rank);
        vector<int> ghosts;
        for (size_t e = 0; e < m.col.size(); e++)
            if (m.col[e] % P != rank) ghosts.push_back(m.col[e]);
        std::sort(ghosts.begin(), ghosts.end());
        ghosts.erase(std::unique(ghosts.begin(), ghosts.end()), ghosts.end());
        int n_ghost = (int)ghosts.size();

        vector<int> remap(m.col.size());
        for (size_t e = 0; e < m.col.size(); e++) {
            int j = m.col[e];
            if (j % P == rank) remap[e] = j / P;
            else remap[e] = n_owned + (int)(std::lower_bound(ghosts.begin(), ghosts.end(), j) - ghosts.begin());
        }

        vector<int> rc(P, 0);
        for (int g = 0; g < n_ghost; g++) rc[ghosts[g] % P]++;
        vector<int> rd(P); rd[0] = 0;
        for (int p = 1; p < P; p++) rd[p] = rd[p - 1] + rc[p - 1];
        vector<int> req(n_ghost), ext_pos(n_ghost), cur(P, 0);
        for (int g = 0; g < n_ghost; g++) {
            int p = ghosts[g] % P;
            int slot = rd[p] + cur[p]++;
            req[slot] = ghosts[g];
            ext_pos[slot] = n_owned + g;
        }

        /* build x_ext: owned block deterministic; ghost block via modeled fetch
         * (the provider owns req[slot] and returns xvec of it) */
        vector<float> x_ext(n_owned + n_ghost);
        for (int mm = 0; mm < n_owned; mm++) x_ext[mm] = xvec_value(rank + mm * P);
        for (int slot = 0; slot < n_ghost; slot++) {
            int g = req[slot];
            if (g % P == rank) { printf("  [FAIL] ghost owned by self\n"); return 1; }
            int provider = g % P;
            int send_local_pos = g / P;                 /* provider's owned position */
            if (provider + send_local_pos * P != g) { printf("  [FAIL] send_local_pos map\n"); return 1; }
            x_ext[ext_pos[slot]] = xvec_value(provider + send_local_pos * P); /* == xvec(g) */
        }
        for (int k = 0; k < m.n_local; k++) {
            float s = 0.0f;
            for (int e = m.row_ptr[k]; e < m.row_ptr[k + 1]; e++)
                s += m.val[e] * x_ext[remap[e]];         /* remapped col into x_ext */
            yB[rank + k * P] = s;
        }
    }

    if (nnz_sum != (long long)A.r.size()) { printf("  [FAIL] nnz sum %lld != %zu\n", nnz_sum, A.r.size()); return 1; }
    double eA = max_abs_diff(yA, yref);
    double eB = max_abs_diff(yB, yref);
    if (verbose)
        printf("  M=%d N=%d P=%d nnz=%zu | modeA err=%.2e modeB err=%.2e\n",
               M, N, P, A.r.size(), eA, eB);
    if (eA > 1e-3 || eB > 1e-3) { printf("  [FAIL] result mismatch (A=%.2e B=%.2e)\n", eA, eB); return 1; }
    return 0;
}

int main()
{
    struct Cfg { int M, N, P; unsigned seed; };
    Cfg cfgs[] = {
        {16, 16, 4, 1}, {10, 10, 4, 2}, {1000, 1000, 4, 3}, {1000, 1000, 3, 4},
        {37, 37, 4, 5}, {50, 20, 4, 6}, {20, 50, 4, 7}, {8, 8, 1, 8},
        {5, 5, 8, 9}, {997, 1013, 7, 10}, {4096, 4096, 4, 11}, {123, 123, 5, 12},
    };
    int fails = 0;
    for (Cfg &c : cfgs) fails += check_config(c.M, c.N, c.P, c.seed, 1);
    if (fails == 0) printf("ALL CHECKS PASSED\n");
    else printf("%d CONFIG(S) FAILED\n", fails);
    return fails ? 1 : 0;
}
