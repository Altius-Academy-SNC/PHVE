/*
 * IC(0): incomplete Cholesky with zero fill-in, plus the two triangular
 * solves needed to apply it as a preconditioner.
 *
 * Why this file exists.  The comparative experiment was first run with
 * SuperLU's threshold ILU (scipy.sparse.linalg.spilu).  That factorisation
 * chooses its own fill, so a numbering that produces long-range couplings
 * is punished twice: once in the fill and once in the factorisation time.
 * IC(0) keeps the sparsity pattern of the lower triangle of A *fixed*, so
 * the fill is identical for every numbering by construction and the only
 * thing that can differ is the quality of the preconditioner, visible in
 * the iteration count.  That isolates the effect we want to measure.
 *
 * Storage: L is lower triangular in CSR, column indices sorted ascending
 * within each row, so the diagonal entry is last in its row.  The pattern
 * is that of tril(A) and is supplied by the caller.
 *
 * Build:  cc -O2 -shared -fPIC -o ic0.so ic0.c -lm
 */

#include <math.h>
#include <stdint.h>
#include <string.h>

/*
 * Factorise in place.  Lx must be preloaded with the corresponding entries
 * of tril(A); the diagonal entries may carry a Manteuffel shift.
 *
 * Returns 0 on success, or (i+1) if the pivot at row i is non-positive,
 * which is the standard IC(0) breakdown for a matrix that is not an
 * M-matrix.
 */
long long ic0_factor(long long n, const int32_t *Lp, const int32_t *Li,
                     double *Lx)
{
    long long i;
    for (i = 0; i < n; i++) {
        long long start = Lp[i], end = Lp[i + 1];
        long long diag_i = end - 1;          /* diagonal is last in the row */
        long long idx;

        if (end <= start) return i + 1;      /* empty row: singular */

        for (idx = start; idx < diag_i; idx++) {
            long long j = Li[idx];
            long long pj_start = Lp[j], pj_end = Lp[j + 1] - 1; /* skip L[j][j] */
            long long pi = start, pj = pj_start;
            double s = Lx[idx];

            /* s -= sum_{k<j} L[i][k] L[j][k]; both rows are sorted */
            while (pi < idx && pj < pj_end) {
                int32_t ci = Li[pi], cj = Li[pj];
                if (ci < cj)      pi++;
                else if (ci > cj) pj++;
                else { s -= Lx[pi] * Lx[pj]; pi++; pj++; }
            }
            Lx[idx] = s / Lx[Lp[j + 1] - 1];
        }

        {
            double d = Lx[diag_i];
            for (idx = start; idx < diag_i; idx++) d -= Lx[idx] * Lx[idx];
            if (!(d > 0.0)) return i + 1;
            Lx[diag_i] = sqrt(d);
        }
    }
    return 0;
}

/*
 * Apply M^{-1} = (L L^T)^{-1}: solve L y = b then L^T x = y.
 * x and b may alias.
 */
void ic0_solve(long long n, const int32_t *Lp, const int32_t *Li,
               const double *Lx, const double *b, double *x)
{
    long long i, idx;

    /* forward substitution, row oriented */
    for (i = 0; i < n; i++) {
        long long start = Lp[i], diag_i = Lp[i + 1] - 1;
        double s = b[i];
        for (idx = start; idx < diag_i; idx++) s -= Lx[idx] * x[Li[idx]];
        x[i] = s / Lx[diag_i];
    }

    /* backward substitution with L^T, expressed through the rows of L */
    for (i = n - 1; i >= 0; i--) {
        long long start = Lp[i], diag_i = Lp[i + 1] - 1;
        x[i] /= Lx[diag_i];
        for (idx = start; idx < diag_i; idx++) x[Li[idx]] -= Lx[idx] * x[i];
    }
}
