/*
 * Deterministic set-associative LRU cache simulator for CSR sparse
 * matrix-vector products.
 *
 * Hardware performance counters are not available in the environment used
 * for the PHVE experiments (kernel.perf_event_paranoid = 4, no privileged
 * access), so cache behaviour is measured by simulation instead.  The
 * simulation is exact for the model it describes and, unlike a hardware
 * counter, is bit-for-bit reproducible on any machine.
 *
 * Model: one unified cache, `sets` sets, `ways` ways, `line` bytes per
 * line, true LRU replacement.  The access trace is the sequence of reads
 * of x[col] performed by y = A*x in CSR order, preceded for each row by
 * the read of y[row]; reads of the CSR arrays themselves are sequential
 * and identical for every ordering, so they are not counted.
 *
 * Build:  cc -O2 -shared -fPIC -o cachesim.so cachesim.c
 */

#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef struct {
    long long accesses;
    long long misses;
} result_t;

/*
 * indptr : (n+1) int32   CSR row pointers
 * indices: (nnz) int32   CSR column indices
 * n      : number of rows
 * elem   : bytes per vector entry (8 for float64)
 * line   : bytes per cache line
 * sets   : number of sets
 * ways   : associativity
 * out    : [accesses, misses]
 */
void simulate_spmv(const int32_t *indptr, const int32_t *indices,
                   long long n, long long elem, long long line,
                   long long sets, long long ways,
                   long long *out)
{
    /* tags[set*ways + w] holds the tag; -1 means invalid.
       age[] holds an LRU counter (smaller = older). */
    int64_t *tags = (int64_t *)malloc(sizeof(int64_t) * sets * ways);
    int64_t *age  = (int64_t *)malloc(sizeof(int64_t) * sets * ways);
    long long i, k, w;
    long long accesses = 0, misses = 0;
    int64_t clock = 0;

    for (i = 0; i < sets * ways; i++) { tags[i] = -1; age[i] = 0; }

    /* The two vectors x and y are laid out in disjoint address ranges.
       y occupies lines [0, nlines_y), x occupies [nlines_y, ...). */
    long long per_line = line / elem;
    long long nlines_y = (n + per_line - 1) / per_line;

    for (i = 0; i < n; i++) {
        long long refs[2];
        long long nrefs;
        long long start = indptr[i], end = indptr[i + 1];

        for (k = start - 1; k < end; k++) {
            long long addr_line;
            if (k < start) {
                addr_line = i / per_line;              /* read of y[i] */
            } else {
                addr_line = nlines_y + indices[k] / per_line;  /* x[col] */
            }
            (void)refs; (void)nrefs;

            {
                long long set = addr_line % sets;
                int64_t tag = (int64_t)(addr_line / sets);
                long long base = set * ways;
                long long hit = -1;
                accesses++;
                clock++;
                for (w = 0; w < ways; w++) {
                    if (tags[base + w] == tag) { hit = w; break; }
                }
                if (hit >= 0) {
                    age[base + hit] = clock;
                } else {
                    long long victim = 0;
                    int64_t best = age[base];
                    for (w = 0; w < ways; w++) {
                        if (tags[base + w] == -1) { victim = w; break; }
                        if (age[base + w] < best) { best = age[base + w]; victim = w; }
                    }
                    tags[base + victim] = tag;
                    age[base + victim] = clock;
                    misses++;
                }
            }
        }
    }

    out[0] = accesses;
    out[1] = misses;
    free(tags);
    free(age);
}
