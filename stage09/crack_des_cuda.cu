/*
 * ============================================================================
 * crack_des_cuda.cu — GPU-accelerated DES brute-force cracker
 * ============================================================================
 *
 * PURPOSE
 * -------
 * Brute-force decrypt "text.d" from Simon Singh's Code Book Cipher Challenge
 * Stage 9.  The file is DES-encrypted in ECB (Electronic Code Book) mode.
 * I know partial key information from Stage 8 and must search the remaining
 * key space for the key that produces readable ASCII plaintext.
 *
 * BACKGROUND: DES (DATA ENCRYPTION STANDARD)
 * -------------------------------------------
 * DES is a 64-bit block cipher with a 64-bit key, of which only 56 bits are
 * effective (every 8th bit is an odd-parity check bit).  It uses 16 rounds
 * of a Feistel network.  In ECB mode, each 8-byte block is encrypted
 * independently with the same key.
 *
 * The cipher's internal pipeline is:
 *   1. IP   (Initial Permutation)  — rearranges the 64 input bits
 *   2. 16 Feistel rounds           — each XORs a round function into one half
 *   3. FP   (Final Permutation)    — inverse of IP, produces the output
 *
 * The round function uses eight "S-boxes" (6-bit → 4-bit substitution tables)
 * followed by a 32-bit permutation ("P").  To speed things up, implementations
 * combine each S-box with P into "SP-box" lookup tables (6-bit → 32-bit).
 *
 * KEY SCHEDULE: the 56 effective key bits are split by "PC1" (Permuted Choice 1)
 * into two 28-bit halves C and D, which are left-rotated by 1 or 2 bits per
 * round.  "PC2" (Permuted Choice 2) then selects 48 bits from {C,D} to form
 * the 48-bit round subkey.
 *
 * PRE-ROTATED IP/FP CONVENTION
 * ----------------------------
 * This implementation uses a well-known optimisation from Eric Young's libdes:
 * the IP includes a one-bit left rotation of each half (ROL 1), and the FP
 * includes the matching right rotation (ROR 1).  This lets the Feistel round
 * use a simple ROR-4 / direct XOR to align with the SP-box 6-bit groups,
 * avoiding per-round bit extraction.  The SP-box entries are pre-rotated by
 * ROL(standard_P_output, 1) to match.
 *
 * KNOWN KEY CONSTRAINTS (from Stage 8)
 * -------------------------------------
 * - byte[0] = 0xD3  (all 7 effective bits known)
 * - byte[1] top 2 effective bits = 11
 *
 * That fixes 9 of the 56 effective key bits, leaving 47 unknown.
 * Search space: 2^47 ≈ 140.7 trillion keys.
 *
 * FILTERING STRATEGY (no crib — pure brute force)
 * -------------------------------------------------
 * 1. Decrypt block 0 with the candidate key.
 * 2. If ALL 8 bytes are printable ASCII (0x20–0x7E) or newline (0x0A, 0x0D),
 *    the key passes the first filter.  This eliminates ~99.98% of keys.
 * 3. Decrypt blocks 1–9.  If ≥50% of them are also all-ASCII, report the key
 *    as a candidate.
 *
 * OPTIMIZATIONS
 * -------------
 * 1. ON-THE-FLY KEY SCHEDULE: instead of pre-computing and storing all 32
 *    subkey words (128 bytes per thread), I compute each round's subkeys
 *    from C,D on the fly inside the Feistel loop.  For decryption,n I start
 *    from C(16) = C(0) (the total rotation is 28 bits = full circle) and walk
 *    backwards.  This saves 128 bytes of register/local memory per thread,
 *    dramatically improving SM occupancy.
 *
 * 2. SP TABLES IN SHARED MEMORY: the eight SP-box tables (512 × 4 = 2 KiB)
 *    are loaded into per-block shared memory.  CUDA constant memory serializes
 *    divergent accesses within a warp (each thread typically indexes a different
 *    SP entry), whereas shared memory serves all 32 lanes in parallel (modulo
 *    bank conflicts).  This avoids the constant-memory bottleneck.
 *
 * 3. NO INTERMEDIATE PLAINTEXT STORAGE BUFFER: each block's ASCII status
 *    is checked inline and the plaintext is never stored.
 *
 * HARDWARE
 * --------
 * Developed for: NVIDIA RTX 3090 (Ampere, SM 8.6, 82 SMs, 10496 CUDA cores).
 * Achieved throughput: ~3.8 billion keys/second.
 * Found the correct key (D3CD1694CBA126FE) in 118.5 minutes (19% of keyspace).
 *
 * BUILD
 * -----
 *   nvcc -O3 -arch=sm_86 crack_des_cuda.cu -o crack_des_cuda.exe
 *
 * Requires: CUDA toolkit whose version matches the installed NVIDIA driver.
 *           MSVC (cl.exe) in PATH — use vcvarsall.bat x64 first on Windows.
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

typedef unsigned char uchar;

/* ============================================================================
 * SP-BOX TABLES (combined S-box + P-permutation, pre-rotated by ROL 1)
 * ============================================================================
 *
 * Standard DES has eight S-boxes (S1–S8), each mapping 6 input bits to 4
 * output bits, followed by a fixed 32-bit permutation P.  For speed, I
 * combine each S-box lookup with the P permutation into a single 64-entry
 * table that maps 6 bits → 32 bits.  The entries are further left-rotated
 * by 1 bit to match the pre-rotated IP/FP convention.
 *
 * Layout: SP_ALL[0..63]   = SP1  (feeds from sk0, bits 31–24 after ROR 4)
 *         SP_ALL[64..127]  = SP2  (feeds from sk1, bits 31–24)
 *         SP_ALL[128..191] = SP3  (feeds from sk0, bits 23–16 after ROR 4)
 *         SP_ALL[192..255] = SP4  (feeds from sk1, bits 23–16)
 *         SP_ALL[256..319] = SP5  (feeds from sk0, bits 15–8 after ROR 4)
 *         SP_ALL[320..383] = SP6  (feeds from sk1, bits 15–8)
 *         SP_ALL[384..447] = SP7  (feeds from sk0, bits 7–0 after ROR 4)
 *         SP_ALL[448..511] = SP8  (feeds from sk1, bits 7–0)
 *
 * In the Feistel function:
 *   - sk0 is XORed with ROR(R, 4).  SP7, SP5, SP3, SP1 consume its bytes.
 *   - sk1 is XORed with R directly. SP8, SP6, SP4, SP2 consume its bytes.
 *
 * These tables are identical across all threads and all keys.  They live in
 * host memory (SP_ALL) and are copied to device global memory (d_SP) at
 * startup.  Each thread block then loads them into shared memory (s_SP) for
 * fast warp-parallel access.
 */
static const uint32_t SP_ALL[512] = {
    /* SP1 — S-box 1 + P permutation, pre-rotated */
    0x01010400,0x00000000,0x00010000,0x01010404,0x01010004,0x00010404,0x00000004,0x00010000,
    0x00000400,0x01010400,0x01010404,0x00000400,0x01000404,0x01010004,0x01000000,0x00000004,
    0x00000404,0x01000400,0x01000400,0x00010400,0x00010400,0x01010000,0x01010000,0x01000404,
    0x00010004,0x01000004,0x01000004,0x00010004,0x00000000,0x00000404,0x00010404,0x01000000,
    0x00010000,0x01010404,0x00000004,0x01010000,0x01010400,0x01000000,0x01000000,0x00000400,
    0x01010004,0x00010000,0x00010400,0x01000004,0x00000400,0x00000004,0x01000404,0x00010404,
    0x01010404,0x00010004,0x01010000,0x01000404,0x01000004,0x00000404,0x00010404,0x01010400,
    0x00000404,0x01000400,0x01000400,0x00000000,0x00010004,0x00010400,0x00000000,0x01010004,
    /* SP2 */
    0x80108020,0x80008000,0x00008000,0x00108020,0x00100000,0x00000020,0x80100020,0x80008020,
    0x80000020,0x80108020,0x80108000,0x80000000,0x80008000,0x00100000,0x00000020,0x80100020,
    0x00108000,0x00100020,0x80008020,0x00000000,0x80000000,0x00008000,0x00108020,0x80100000,
    0x00100020,0x80000020,0x00000000,0x00108000,0x00008020,0x80108000,0x80100000,0x00008020,
    0x00000000,0x00108020,0x80100020,0x00100000,0x80008020,0x80100000,0x80108000,0x00008000,
    0x80100000,0x80008000,0x00000020,0x80108020,0x00108020,0x00000020,0x00008000,0x80000000,
    0x00008020,0x80108000,0x00100000,0x80000020,0x00100020,0x80008020,0x80000020,0x00100020,
    0x00108000,0x00000000,0x80008000,0x00008020,0x80000000,0x80100020,0x80108020,0x00108000,
    /* SP3 */
    0x00000208,0x08020200,0x00000000,0x08020008,0x08000200,0x00000000,0x00020208,0x08000200,
    0x00020008,0x08000008,0x08000008,0x00020000,0x08020208,0x00020008,0x08020000,0x00000208,
    0x08000000,0x00000008,0x08020200,0x00000200,0x00020200,0x08020000,0x08020008,0x00020208,
    0x08000208,0x00020200,0x00020000,0x08000208,0x00000008,0x08020208,0x00000200,0x08000000,
    0x08020200,0x08000000,0x00020008,0x00000208,0x00020000,0x08020200,0x08000200,0x00000000,
    0x00000200,0x00020008,0x08020208,0x08000200,0x08000008,0x00000200,0x00000000,0x08020008,
    0x08000208,0x00020000,0x08000000,0x08020208,0x00000008,0x00020208,0x00020200,0x08000008,
    0x08020000,0x08000208,0x00000208,0x08020000,0x00020208,0x00000008,0x08020008,0x00020200,
    /* SP4 */
    0x00802001,0x00002081,0x00002081,0x00000080,0x00802080,0x00800081,0x00800001,0x00002001,
    0x00000000,0x00802000,0x00802000,0x00802081,0x00000081,0x00000000,0x00800080,0x00800001,
    0x00000001,0x00002000,0x00800000,0x00802001,0x00000080,0x00800000,0x00002001,0x00002080,
    0x00800081,0x00000001,0x00002080,0x00800080,0x00002000,0x00802080,0x00802081,0x00000081,
    0x00800080,0x00800001,0x00802000,0x00802081,0x00000081,0x00000000,0x00000000,0x00802000,
    0x00002080,0x00800080,0x00800081,0x00000001,0x00802001,0x00002081,0x00002081,0x00000080,
    0x00802081,0x00000081,0x00000001,0x00002000,0x00800001,0x00002001,0x00802080,0x00800081,
    0x00002001,0x00002080,0x00800000,0x00802001,0x00000080,0x00800000,0x00002000,0x00802080,
    /* SP5 */
    0x00000100,0x02080100,0x02080000,0x42000100,0x00080000,0x00000100,0x40000000,0x02080000,
    0x40080100,0x00080000,0x02000100,0x40080100,0x42000100,0x42080000,0x00080100,0x40000000,
    0x02000000,0x40080000,0x40080000,0x00000000,0x40000100,0x42080100,0x42080100,0x02000100,
    0x42080000,0x40000100,0x00000000,0x42000000,0x02080100,0x02000000,0x42000000,0x00080100,
    0x00080000,0x42000100,0x00000100,0x02000000,0x40000000,0x02080000,0x42000100,0x40080100,
    0x02000100,0x40000000,0x42080000,0x02080100,0x40080100,0x00000100,0x02000000,0x42080000,
    0x42080100,0x00080100,0x42000000,0x42080100,0x02080000,0x00000000,0x40080000,0x42000000,
    0x00080100,0x02000100,0x40000100,0x00080000,0x00000000,0x40080000,0x02080100,0x40000100,
    /* SP6 */
    0x20000010,0x20400000,0x00004000,0x20404010,0x20400000,0x00000010,0x20404010,0x00400000,
    0x20004000,0x00404010,0x00400000,0x20000010,0x00400010,0x20004000,0x20000000,0x00004010,
    0x00000000,0x00400010,0x20004010,0x00004000,0x00404000,0x20004010,0x00000010,0x20400010,
    0x20400010,0x00000000,0x00404010,0x20404000,0x00004010,0x00404000,0x20404000,0x20000000,
    0x20004000,0x00000010,0x20400010,0x00404000,0x20404010,0x00400000,0x00004010,0x20000010,
    0x00400000,0x20004000,0x20000000,0x00004010,0x20000010,0x20404010,0x00404000,0x20400000,
    0x00404010,0x20404000,0x00000000,0x20400010,0x00000010,0x00004000,0x20400000,0x00404010,
    0x00004000,0x00400010,0x20004010,0x00000000,0x20404000,0x20000000,0x00400010,0x20004010,
    /* SP7 */
    0x00200000,0x04200002,0x04000802,0x00000000,0x00000800,0x04000802,0x00200802,0x04200800,
    0x04200802,0x00200000,0x00000000,0x04000002,0x00000002,0x04000000,0x04200002,0x00000802,
    0x04000800,0x00200802,0x00200002,0x04000800,0x04000002,0x04200000,0x04200800,0x00200002,
    0x04200000,0x00000800,0x00000802,0x04200802,0x00200800,0x00000002,0x04000000,0x00200800,
    0x04000000,0x00200800,0x00200000,0x04000802,0x04000802,0x04200002,0x04200002,0x00000002,
    0x00200002,0x04000000,0x04000800,0x00200000,0x04200800,0x00000802,0x00200802,0x04200800,
    0x00000802,0x04000002,0x04200802,0x04200000,0x00200800,0x00000000,0x00000002,0x04200802,
    0x00000000,0x00200802,0x04200000,0x00000800,0x04000002,0x04000800,0x00000800,0x00200002,
    /* SP8 */
    0x10001040,0x00001000,0x00040000,0x10041040,0x10000000,0x10001040,0x00000040,0x10000000,
    0x00040040,0x10040000,0x10041040,0x00041000,0x10041000,0x00041040,0x00001000,0x00000040,
    0x10040000,0x10000040,0x10001000,0x00001040,0x00041000,0x00040040,0x10040040,0x10041000,
    0x00001040,0x00000000,0x00000000,0x10040040,0x10000040,0x10001000,0x00041040,0x00040000,
    0x00041040,0x00040000,0x10041000,0x00001000,0x00000040,0x10040040,0x00001000,0x00041040,
    0x10001000,0x00000040,0x10000040,0x10040000,0x10040040,0x10000000,0x00040000,0x10001040,
    0x00000000,0x10041040,0x00040040,0x10000040,0x10040000,0x10001000,0x10001040,0x00000000,
    0x10041040,0x00041000,0x00041000,0x00001040,0x00001040,0x00040040,0x10000000,0x10041000
};


/* ============================================================================
 * CUDA CONSTANT MEMORY — small read-only data broadcast to all threads
 * ============================================================================
 *
 * __constant__ memory is cached and optimized for the case where every thread
 * in a warp reads the SAME address (broadcast).  It serializes divergent
 * reads, but these tables are always indexed identically across a warp
 * (all threads use the same round number, same PC1 index).
 */

/*
 * KEY_SHIFTS: number of left-rotation bits applied to C and D in each of
 * the 16 key schedule rounds.  The total is 28 (= full rotation back to
 * start), which is the key property I exploit for on-the-fly decryption.
 *
 * Rounds 0,1,8,15 shift by 1; all others shift by 2.
 */
__constant__ int d_KEY_SHIFTS[16] = {1,1,2,2,2,2,2,2,1,2,2,2,2,2,2,1};

/*
 * PC1 (Permuted Choice 1): selects 56 bits from the 64-bit key (discarding
 * parity bits) and splits them into two 28-bit halves C and D.
 *
 * Values are 1-indexed bit positions in the key (bit 1 = MSB of byte 0).
 * d_pc1_c selects the 28 bits for the C half; d_pc1_d for the D half.
 */
__constant__ int d_pc1_c[28] = {
    57,49,41,33,25,17, 9, 1,58,50,42,34,26,18,
    10, 2,59,51,43,35,27,19,11, 3,60,52,44,36
};
__constant__ int d_pc1_d[28] = {
    63,55,47,39,31,23,15, 7,62,54,46,38,30,22,
    14, 6,61,53,45,37,29,21,13, 5,28,20,12, 4
};

/*
 * First 80 bytes of ciphertext (10 blocks) — loaded from text.d at startup.
 * Only the first 10 blocks are needed for the GPU ASCII filter.  The full
 * file is decrypted on the host once a candidate key is found.
 */
__constant__ uchar d_ct[80];

/*
 * SP tables in device GLOBAL memory (not __constant__).  Each thread block
 * copies these into shared memory at kernel launch.  I use global memory
 * rather than constant memory because the SP lookups are divergent (each
 * thread in a warp indexes a different table entry), and constant memory
 * serializes such accesses while shared memory serves them in parallel.
 */
__device__ uint32_t d_SP[512];


/* ============================================================================
 * DEVICE FUNCTION: compute_pc1 — Permuted Choice 1
 * ============================================================================
 *
 * Extracts the 56 effective key bits from the 8-byte key and packs them into
 * two 28-bit words (c, d) that form the starting state of the key schedule.
 *
 * For each of the 28 C-half bits:
 *   - Look up which key bit position it comes from (d_pc1_c[i], 1-indexed)
 *   - Convert to byte index (bc >> 3) and bit-within-byte (0x80 >> (bc & 7))
 *   - If that bit is set in the key, set bit (27 - i) in c
 *
 * Same for the D-half using d_pc1_d.
 *
 * __forceinline__: this is called once per thread; inlining avoids the
 * function-call overhead.
 */
__device__ __forceinline__ void compute_pc1(const uchar *key, uint32_t *c_out, uint32_t *d_out) {
    uint32_t c = 0, d = 0;
    for (int i = 0; i < 28; i++) {
        int bc = d_pc1_c[i] - 1;                       /* convert to 0-indexed */
        if (key[bc >> 3] & (0x80 >> (bc & 7))) c |= (1u << (27 - i));
        int bd = d_pc1_d[i] - 1;
        if (key[bd >> 3] & (0x80 >> (bd & 7))) d |= (1u << (27 - i));
    }
    *c_out = c;
    *d_out = d;
}


/* ============================================================================
 * DEVICE FUNCTION: compute_pc2 — Permuted Choice 2 (bitwise)
 * ============================================================================
 *
 * Computes the two 32-bit subkey words (sk0, sk1) from the current 28-bit
 * C and D halves.  PC2 selects 48 of the 56 bits and packs them into the
 * format expected by the Feistel function:
 *
 *   sk0: bits for S-boxes 1, 3, 5, 7  (the "odd" S-boxes)
 *         packed at bit positions [29–24], [21–16], [13–8], [5–0]
 *
 *   sk1: bits for S-boxes 2, 4, 6, 8  (the "even" S-boxes)
 *         packed at the same positions
 *
 * Each S-box consumes 6 bits from one of the two subkey words.  The Feistel
 * function XORs sk0 with ROR(R, 4) and sk1 with R, then feeds 6-bit slices
 * into the eight SP-box lookups.
 *
 * The expressions below are a direct bitwise implementation of the standard
 * PC2 table, avoiding any loops or table lookups.  Each shift-and-mask
 * extracts one bit from C or D and places it at the correct position in the
 * output word.  Some masks combine multiple bits that happen to share the
 * same shift amount.
 *
 * CORRECTNESS NOTE: these expressions were originally taken from Eric Young's
 * libdes but had sk0 and sk1 SWAPPED.  The bug was found and fixed by
 * comparing against a table-driven PC2 implementation.
 */
__device__ __forceinline__ void compute_pc2(uint32_t c, uint32_t d, uint32_t *sk0, uint32_t *sk1) {
    /* sk0: feeds S-boxes 1, 3, 5, 7 (XORed with ROR(R,4) in the Feistel) */
    *sk0 =
        ((c << 15) & 0x20000000) | ((c << 17) & 0x10000000) |
        ((c << 10) & 0x08000000) | ((c << 22) & 0x04000000) |
        ((c >>  2) & 0x02000000) | ((c <<  1) & 0x01000000) |
        ((c << 16) & 0x00200000) | ((c << 11) & 0x00100000) |
        ((c <<  3) & 0x00080000) | ((c >>  6) & 0x00040000) |
        ((c << 15) & 0x00020000) | ((c >>  4) & 0x00010000) |
        ((d >>  2) & 0x00002000) | ((d <<  8) & 0x00001000) |
        ((d >> 14) & 0x00000808) | ((d >>  9) & 0x00000400) |
        ((d      ) & 0x00000200) | ((d <<  7) & 0x00000100) |
        ((d >>  7) & 0x00000020) | ((d >>  3) & 0x00000011) |
        ((d <<  2) & 0x00000004) | ((d >> 21) & 0x00000002);

    /* sk1: feeds S-boxes 2, 4, 6, 8 (XORed with R directly in the Feistel) */
    *sk1 =
        ((c <<  4) & 0x24000000) | ((c << 28) & 0x10000000) |
        ((c << 14) & 0x08000000) | ((c << 18) & 0x02080000) |
        ((c <<  6) & 0x01000000) | ((c <<  9) & 0x00200000) |
        ((c >>  1) & 0x00100000) | ((c << 10) & 0x00040000) |
        ((c <<  2) & 0x00020000) | ((c >> 10) & 0x00010000) |
        ((d >> 13) & 0x00002000) | ((d >>  4) & 0x00001000) |
        ((d <<  6) & 0x00000800) | ((d >>  1) & 0x00000400) |
        ((d >> 14) & 0x00000200) | ((d      ) & 0x00000100) |
        ((d >>  5) & 0x00000020) | ((d >> 10) & 0x00000010) |
        ((d >>  3) & 0x00000008) | ((d >> 18) & 0x00000004) |
        ((d >> 26) & 0x00000002) | ((d >> 24) & 0x00000001);
}


/* ============================================================================
 * DEVICE FUNCTION: des_decrypt_check_ascii
 * ============================================================================
 *
 * Decrypts one 8-byte DES block using the on-the-fly key schedule and checks
 * whether all 8 plaintext bytes are printable ASCII.
 *
 * Parameters:
 *   ct_block — pointer to 8 bytes of ciphertext (in __constant__ memory)
 *   c0, d0   — the PC1 output halves for this key (28 bits each)
 *   SP       — pointer to the SP tables in shared memory (512 entries)
 *
 * Returns: 1 if all 8 decrypted bytes are printable ASCII, 0 otherwise.
 *
 * ON-THE-FLY KEY SCHEDULE FOR DECRYPTION
 * ----------------------------------------
 * The standard key schedule left-rotates C,D before computing each round's
 * subkey.  The shift amounts total 28 bits = a full rotation, so after all
 * 16 rounds, C(16) = C(0) and D(16) = D(0).
 *
 * Encryption uses subkeys in order sk[0]..sk[15].  Decryption reverses this:
 * sk[15] first, sk[14] next, ..., sk[0] last.
 *
 * For on-the-fly decryption:
 *   1. Start with c = c0 (= c16, since the rotation wraps around)
 *   2. For round = 15 downto 0:
 *      a. Compute sk from current c, d via PC2   (this is sk[round])
 *      b. Perform the Feistel round
 *      c. Reverse this round's rotation: c = ROR(c, KEY_SHIFTS[round])
 *
 * This produces subkeys in the correct reverse order without ever storing
 * the full sk[32] array.
 */
__device__ int des_decrypt_check_ascii(const uchar *ct_block, uint32_t c0, uint32_t d0,
                                        const uint32_t *SP) {
    uint32_t left, right, temp;

    /* ---- Load 8-byte ciphertext block into two 32-bit words ---- */
    left  = ((uint32_t)ct_block[0] << 24) | ((uint32_t)ct_block[1] << 16) |
            ((uint32_t)ct_block[2] << 8)  | ct_block[3];
    right = ((uint32_t)ct_block[4] << 24) | ((uint32_t)ct_block[5] << 16) |
            ((uint32_t)ct_block[6] << 8)  | ct_block[7];

    /* ---- Initial Permutation (IP) + pre-rotation ----
     *
     * The IP rearranges the 64 input bits.  This implementation uses the
     * PERM_OP macro approach (from Eric Young's libdes): a sequence of
     * shift-XOR-mask operations that is equivalent to the bit-level IP
     * but much faster on a 32-bit machine.
     *
     * After the four PERM_OP steps, both halves are left-rotated by 1 bit.
     * This "pre-rotation" means the Feistel function can use a simple
     * ROR(R, 4) to align the 6-bit SP-box groups, avoiding per-round
     * bit extraction.  The FP at the end reverses this rotation.
     *
     * PERM_OP(a, b, n, mask):
     *   temp = ((a >> n) ^ b) & mask;
     *   b ^= temp;
     *   a ^= temp << n;
     * This swaps bits between a and b that are n positions apart and
     * selected by mask.  Each PERM_OP is its own inverse.
     */
    temp = ((left >> 4) ^ right) & 0x0F0F0F0F; right ^= temp; left ^= temp << 4;	/* step 1 */
    temp = ((left >> 16) ^ right) & 0x0000FFFF; right ^= temp; left ^= temp << 16;	/* step 2 */
    temp = ((right >> 2) ^ left) & 0x33333333; left ^= temp; right ^= temp << 2;	/* step 3 */
    temp = ((right >> 8) ^ left) & 0x00FF00FF; left ^= temp; right ^= temp << 8;	/* step 4 */
    right = (right << 1) | (right >> 31);						/* ROL(right,1) */
    temp = (left ^ right) & 0xAAAAAAAA; right ^= temp; left ^= temp;			/* odd/even swap */
    left = (left << 1) | (left >> 31);							/* ROL(left,1) */

    /* ---- On-the-fly decryption: start from c0 = c16, d0 = d16 ---- */
    uint32_t c = c0, d = d0;

    /* ---- 16 Feistel rounds (subkeys applied in reverse for decryption) ----
     *
     * Each round:
     *   1. Compute the subkey pair (sk0, sk1) from current C, D via PC2
     *   2. Apply the Feistel function: f = F(right, sk0, sk1)
     *   3. XOR f into left
     *   4. Swap left and right
     *   5. Reverse the key schedule rotation for this round
     *
     * #pragma unroll tells the compiler to fully unroll all 16 iterations,
     * eliminating loop overhead and enabling more register optimization.
     */
    #pragma unroll
    for (int round = 15; round >= 0; round--) {
        /* Compute this round's subkeys from current C, D */
        uint32_t sk0, sk1;
        compute_pc2(c, d, &sk0, &sk1);

        /* ---- Feistel function F(R, sk0, sk1) ----
         *
         * The 32-bit right half R is expanded to 48 bits and XORed with the
         * 48-bit subkey, then split into eight 6-bit groups fed to the S-boxes.
         *
         * With the pre-rotation convention:
         *   - work = ROR(R, 4) XOR sk0  → feeds SP7, SP5, SP3, SP1
         *   - work = R XOR sk1          → feeds SP8, SP6, SP4, SP2
         *
         * Each SP table takes the low 6 bits of its byte (work & 0x3F),
         * looks up a 32-bit result, and ORs it into fval.  The work word
         * is right-shifted by 8 bits between lookups to expose the next
         * 6-bit group.
         *
         * SP table offsets in the unified array:
         *   SP1 = [0],   SP2 = [64],  SP3 = [128], SP4 = [192]
         *   SP5 = [256], SP6 = [320], SP7 = [384], SP8 = [448]
         */
        uint32_t work, fval;

        /* First half: ROR(R, 4) XOR sk0 → odd S-boxes */
        work = (right << 28) | (right >> 4);   /* ROR(right, 4) */
        work ^= sk0;
        fval  = SP[384 + (work & 0x3F)]; work >>= 8;   /* SP7 */
        fval |= SP[256 + (work & 0x3F)]; work >>= 8;   /* SP5 */
        fval |= SP[128 + (work & 0x3F)]; work >>= 8;   /* SP3 */
        fval |= SP[  0 + (work & 0x3F)];                /* SP1 */

        /* Second half: R XOR sk1 → even S-boxes */
        work = right ^ sk1;
        fval |= SP[448 + (work & 0x3F)]; work >>= 8;   /* SP8 */
        fval |= SP[320 + (work & 0x3F)]; work >>= 8;   /* SP6 */
        fval |= SP[192 + (work & 0x3F)]; work >>= 8;   /* SP4 */
        fval |= SP[ 64 + (work & 0x3F)];                /* SP2 */

        /* XOR the round function output into the left half, then swap */
        left ^= fval;
        temp = left; left = right; right = temp;

        /* ---- Reverse the key schedule rotation for this round ----
         *
         * During encryption, C and D were left-rotated BEFORE computing
         * this round's subkey.  To walk backwards, I right-rotate AFTER
         * using the subkey.  This gives me the C, D state for the previous
         * round, ready for the next iteration.
         *
         * The mask 0x0FFFFFFF keeps only the low 28 bits.
         */
        int shifts = d_KEY_SHIFTS[round];
        if (shifts == 1) {
            c = ((c >> 1) | (c << 27)) & 0x0FFFFFFF;   /* ROR(c, 1) within 28 bits */
            d = ((d >> 1) | (d << 27)) & 0x0FFFFFFF;
        } else {
            c = ((c >> 2) | (c << 26)) & 0x0FFFFFFF;   /* ROR(c, 2) within 28 bits */
            d = ((d >> 2) | (d << 26)) & 0x0FFFFFFF;
        }
    }

    /* ---- Undo the last Feistel swap ----
     *
     * The Feistel loop swaps left↔right at the end of every round, but
     * the last round's swap is not part of the standard DES definition.
     * I undo it here before applying the Final Permutation.
     */
    temp = left; left = right; right = temp;

    /* ---- Final Permutation (FP) — exact inverse of the IP ----
     *
     * The FP undoes the IP.  Since the IP consisted of four PERM_OP steps
     * plus ROL operations, the FP applies them in reverse order with the
     * rotations replaced by their inverses (ROR).
     *
     * IP operations (in order):
     *   1. PERM_OP(L, R, 4,  0x0F0F0F0F)
     *   2. PERM_OP(L, R, 16, 0x0000FFFF)
     *   3. PERM_OP(R, L, 2,  0x33333333)
     *   4. PERM_OP(R, L, 8,  0x00FF00FF)
     *   5. ROL(right, 1)
     *   6. odd/even swap via 0xAAAAAAAA
     *   7. ROL(left, 1)
     *
     * FP operations (reverse order, ROL→ROR):
     *   7'. ROR(left, 1)
     *   6'. odd/even swap via 0xAAAAAAAA
     *   5'. ROR(right, 1)
     *   4'. PERM_OP(R, L, 8,  0x00FF00FF)
     *   3'. PERM_OP(R, L, 2,  0x33333333)
     *   2'. PERM_OP(L, R, 16, 0x0000FFFF)
     *   1'. PERM_OP(L, R, 4,  0x0F0F0F0F)
     *
     * CORRECTNESS NOTE: the original kernel code had left and right
     * swapped in every FP operation, causing the IP→FP round-trip to fail.
     * This was the root cause of ALL test failures across three debugging
     * sessions.  The fix was derived by carefully inverting each IP step.
     */
    left = (left << 31) | (left >> 1);                                               /* 7': ROR(left,1) */
    temp = (left ^ right) & 0xAAAAAAAA; right ^= temp; left ^= temp;                 /* 6': odd/even swap */
    right = (right << 31) | (right >> 1);                                             /* 5': ROR(right,1) */
    temp = ((right >> 8) ^ left) & 0x00FF00FF; left ^= temp; right ^= temp << 8;     /* 4' */
    temp = ((right >> 2) ^ left) & 0x33333333; left ^= temp; right ^= temp << 2;     /* 3' */
    temp = ((left >> 16) ^ right) & 0x0000FFFF; right ^= temp; left ^= temp << 16;   /* 2' */
    temp = ((left >> 4) ^ right) & 0x0F0F0F0F; right ^= temp; left ^= temp << 4;     /* 1' */

    /* ---- Check all 8 plaintext bytes for printable ASCII ----
     *
     * After the FP:
     *   left  holds bytes 0–3 of the plaintext (MSB first)
     *   right holds bytes 4–7 of the plaintext (MSB first)
     *
     * A byte is "printable" if it's in 0x20–0x7E (space through tilde)
     * or is a line-feed (0x0A) or carriage-return (0x0D).
     *
     * If ANY byte fails, return 0 immediately — this key is wrong.
     * This early exit is critical: 99.98% of keys fail on the first block,
     * so most threads bail out very quickly.
     */
    uint32_t words[2] = { left, right };
    for (int w = 0; w < 2; w++) {
        uint32_t v = words[w];
        for (int s = 24; s >= 0; s -= 8) {             /* extract each byte */
            uint32_t b = (v >> s) & 0xFF;
            if (b < 0x20 || b > 0x7E) {                /* not printable? */
                if (b != 0x0A && b != 0x0D) return 0;  /* not newline either → reject */
            }
        }
    }
    return 1;  /* all 8 bytes are printable ASCII */
}


/* ============================================================================
 * DEVICE FUNCTION: set_parity — compute DES odd-parity bit
 * ============================================================================
 *
 * DES keys use odd parity: each byte must have an odd number of 1-bits.
 * The LSB (bit 0) is the parity bit; bits 7–1 are the effective key bits.
 *
 * I XOR-fold the byte down to a single bit to count parity, then set
 * bit 0 to make the total parity odd.
 *
 * Example: b = 0x66 = 01100110 → 4 ones (even) → set bit 0 → 01100111 = 0x67
 */
__device__ __forceinline__ uchar set_parity(uchar b) {
    uchar v = b;
    v ^= v >> 4;           /* fold bits 7–4 into bits 3–0 */
    v ^= v >> 2;           /* fold into bits 1–0 */
    v ^= v >> 1;           /* fold into bit 0: now v&1 = parity of b */
    return (b & 0xFE) | ((v & 1) ^ 1);  /* clear old parity, set to make odd */
}


/* ============================================================================
 * GPU KERNEL: des_crack_kernel — brute-force DES key search
 * ============================================================================
 *
 * Each thread tests one candidate key:
 *   1. Construct the 8-byte DES key from the linear key index
 *   2. Compute PC1 to get the key schedule starting state (c0, d0)
 *   3. Decrypt block 0 and check if all bytes are printable ASCII
 *   4. If block 0 passes, decrypt blocks 1–9 and require ≥50% all-ASCII
 *   5. If both filters pass, atomically store the key as a candidate
 *
 * KEY INDEX ENCODING
 * ------------------
 * The 47-bit key_idx encodes the 47 unknown effective key bits:
 *
 *   key_idx bits [46:42] → byte[1] effective bits [4:0]  (top 2 are fixed 11)
 *   key_idx bits [41:35] → byte[2] effective bits [6:0]
 *   key_idx bits [34:28] → byte[3] effective bits [6:0]
 *   key_idx bits [27:21] → byte[4] effective bits [6:0]
 *   key_idx bits [20:14] → byte[5] effective bits [6:0]
 *   key_idx bits [13:7]  → byte[6] effective bits [6:0]
 *   key_idx bits [6:0]   → byte[7] effective bits [6:0]
 *
 * For each byte, the 7 effective bits are placed in bits [7:1] and the
 * parity bit (bit 0) is computed by set_parity().
 *
 * Parameters:
 *   key_offset    — starting key index for this batch
 *   num_ct_blocks — number of ciphertext blocks available (up to 10 used)
 *   result_count  — device pointer to atomic counter of found candidates
 *   result_keys   — device buffer to store up to 64 candidate keys (8 bytes each)
 */
__global__ void des_crack_kernel(
    uint64_t key_offset,
    int num_ct_blocks,
    uint32_t *result_count,
    uchar *result_keys
) {
    /* ---- Load SP tables from global memory into shared memory ----
     *
     * Shared memory is per-block (49152 bytes on RTX 3090).  I use 2048
     * bytes (512 × 4) for the SP tables.  Each thread loads a subset of
     * entries in a strided pattern so all 256 threads cooperate.
     *
     * __syncthreads() ensures all entries are loaded before any thread
     * starts using them.
     */
    __shared__ uint32_t s_SP[512];
    for (int i = threadIdx.x; i < 512; i += blockDim.x) {
        s_SP[i] = d_SP[i];
    }
    __syncthreads();

    /* ---- Compute this thread's key index ---- */
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t key_idx = key_offset + gid;

    /* ---- Construct the 8-byte DES key from key_idx ----
     *
     * byte[0] = 0xD3: all 7 effective bits known from Stage 8.
     *   0xD3 = 11010011, effective bits (7–1) = 1101001, parity bit = 1 ✓
     *
     * byte[1]: top 2 effective bits = 11 (known from Stage 8).
     *   The 7 effective bits are: 1 1 [key_idx bit 46] ... [key_idx bit 42]
     *   k1_full7 = 0x60 | lower5 = 110_xxxxx
     *   Shift left by 1 to place in bits [7:1], then set parity for bit 0.
     *
     * bytes[2–7]: all 7 effective bits come from key_idx.
     *   Extract 7 bits, shift left by 1 into [7:1], set parity for bit 0.
     */
    uchar key[8];
    uint32_t k1_lower5 = (uint32_t)((key_idx >> 42) & 0x1F);  /* 5 unknown bits */
    uint32_t k1_full7 = 0x60 | k1_lower5;                      /* prepend fixed 11 */

    key[0] = 0xD3;                                                          /* fully known */
    key[1] = set_parity((uchar)(k1_full7 << 1));                            /* 2 fixed + 5 variable */
    key[2] = set_parity((uchar)(((key_idx >> 35) & 0x7F) << 1));            /* 7 variable bits */
    key[3] = set_parity((uchar)(((key_idx >> 28) & 0x7F) << 1));
    key[4] = set_parity((uchar)(((key_idx >> 21) & 0x7F) << 1));
    key[5] = set_parity((uchar)(((key_idx >> 14) & 0x7F) << 1));
    key[6] = set_parity((uchar)(((key_idx >>  7) & 0x7F) << 1));
    key[7] = set_parity((uchar)(( key_idx        & 0x7F) << 1));

    /* ---- Compute PC1: extract the key schedule starting state ---- */
    uint32_t c0, d0;
    compute_pc1(key, &c0, &d0);

    /* ---- FILTER 1: decrypt block 0 and check for all-ASCII ----
     *
     * This is the primary filter.  A random 8-byte block has a ~0.02%
     * chance of being all printable ASCII, so this eliminates 99.98%
     * of candidate keys immediately.  Early return keeps the common
     * path very short.
     */
    if (!des_decrypt_check_ascii(d_ct, c0, d0, s_SP)) return;

    /* ---- FILTER 2: check blocks 1–9 (≥50% must be all-ASCII) ----
     *
     * The few keys that pass filter 1 (~0.02%) are checked more
     * thoroughly.  I decrypt up to 9 more blocks and require at least
     * half of all 10 blocks to be all-ASCII.  This reduces false
     * positives to essentially zero.
     */
    int ascii_count = 1;                                   /* block 0 already passed */
    int check = num_ct_blocks < 10 ? num_ct_blocks : 10;   /* cap at 10 blocks */
    for (int blk = 1; blk < check; blk++) {
        if (des_decrypt_check_ascii(d_ct + blk * 8, c0, d0, s_SP))
            ascii_count++;
    }

    if (ascii_count < (check + 1) / 2) return;             /* need ≥50% all-ASCII */

    /* ---- CANDIDATE FOUND: store the key ----
     *
     * atomicAdd returns the old value, giving each candidate a unique slot.
     * I store up to 64 candidates (more than enough — in practice exactly
     * one key passes both filters).
     */
    uint32_t idx = atomicAdd(result_count, 1);
    if (idx < 64) {
        for (int i = 0; i < 8; i++) result_keys[idx * 8 + i] = key[i];
    }
}


/* ============================================================================
 *                           HOST CODE
 * ============================================================================
 *
 * The host is responsible for:
 *   1. Self-testing the DES implementation before launching the GPU
 *   2. Loading the ciphertext file (text.d) and uploading it to the GPU
 *   3. Dispatching the brute-force kernel in batches and checking for results
 *   4. Decrypting the full ciphertext on the CPU once a key is found
 *
 * The host has its own DES implementation (identical algorithm, not optimised
 * for throughput) used for the self-test and the final full-file decryption.
 */


/* ---- Host DES constants (same as device, duplicated for the host) ---- */
static const int h_KEY_SHIFTS[16] = {1,1,2,2,2,2,2,2,1,2,2,2,2,2,2,1};
static const int h_pc1_c[28] = {57,49,41,33,25,17,9,1,58,50,42,34,26,18,10,2,59,51,43,35,27,19,11,3,60,52,44,36};
static const int h_pc1_d[28] = {63,55,47,39,31,23,15,7,62,54,46,38,30,22,14,6,61,53,45,37,29,21,13,5,28,20,12,4};

/*
 * PC2 table (standard, 1-indexed).  48 entries select bits from the 56-bit
 * {C,D} concatenation.  Positions 1–28 come from C; 29–56 from D.
 *
 * The table-driven approach is only used on the host for clarity.  The GPU
 * uses the bitwise compute_pc2() for speed.
 */
static const int h_pc2[48] = {14,17,11,24,1,5,3,28,15,6,21,10,23,19,12,4,26,8,16,7,27,20,13,2,41,52,31,37,47,55,30,40,51,45,33,48,44,49,39,56,34,53,46,42,50,36,29,32};


/*
 * host_des_key_schedule — generate all 32 subkey words from an 8-byte key.
 *
 * This is the standard DES key schedule:
 *   1. Apply PC1 to extract 28-bit C and D halves
 *   2. For each of 16 rounds:
 *      a. Left-rotate C and D by KEY_SHIFTS[round] bits
 *      b. Apply PC2 to select 48 bits and pack into sk[round*2] and sk[round*2+1]
 *
 * The PC2 packing places each S-box's 6 bits at the correct position:
 *   sbox 0,2,4,6 (even) → sk[round*2]   (S-boxes 1,3,5,7 in DES numbering)
 *   sbox 1,3,5,7 (odd)  → sk[round*2+1] (S-boxes 2,4,6,8 in DES numbering)
 *   Each group of 6 bits occupies positions [base+5 .. base] where
 *   base = 24 - (sbox/2)*8.
 */
void host_des_key_schedule(const uchar *key, uint32_t *sk) {
    uint32_t c = 0, d = 0;

    /* PC1: extract 56 bits into C (28 bits) and D (28 bits) */
    for (int i = 0; i < 28; i++) {
        int bc = h_pc1_c[i]-1; if (key[bc/8] & (0x80 >> (bc%8))) c |= (1u<<(27-i));
        int bd = h_pc1_d[i]-1; if (key[bd/8] & (0x80 >> (bd%8))) d |= (1u<<(27-i));
    }

    /* 16 rounds of rotation + PC2 */
    for (int r = 0; r < 16; r++) {
        /* Left-rotate C and D within 28 bits */
        for (int s = 0; s < h_KEY_SHIFTS[r]; s++) {
            c = ((c<<1)|(c>>27)) & 0x0FFFFFFF;
            d = ((d<<1)|(d>>27)) & 0x0FFFFFFF;
        }

        /* PC2: select 48 bits from {C,D} and pack into two 32-bit words */
        sk[r*2]=0; sk[r*2+1]=0;
        for (int i = 0; i < 48; i++) {
            int src = h_pc2[i];
            /* src 1–28 → C bit (28-src), src 29–56 → D bit (56-src) */
            int val = (src<=28) ? ((c>>(28-src))&1) : ((d>>(56-src))&1);
            if (val) {
                int sbox=i/6, bip=i%6, w=sbox&1, base=24-(sbox>>1)*8, tgt=base+5-bip;
                if (w==0) sk[r*2] |= (1u<<tgt); else sk[r*2+1] |= (1u<<tgt);
            }
        }
    }
}


/*
 * host_des_crypt — encrypt or decrypt one 8-byte block.
 *
 * Uses the same IP / Feistel / FP algorithm as the GPU kernel, but with
 * pre-computed subkeys (sk[32]) and no shared-memory tricks.  Used for:
 *   - Self-test (encrypt + decrypt known vectors)
 *   - Full-file decryption once the key is found
 *
 * Parameters:
 *   in      — 8-byte input block
 *   out     — 8-byte output block
 *   sk      — 32-word subkey array from host_des_key_schedule
 *   decrypt — 0 for encrypt (subkeys in forward order),
 *             1 for decrypt (subkeys in reverse order)
 */
void host_des_crypt(const uchar *in, uchar *out, const uint32_t *sk, int decrypt) {
    uint32_t l, r, t;

    /* Load input block */
    l = ((uint32_t)in[0]<<24) | ((uint32_t)in[1]<<16) | ((uint32_t)in[2]<<8) | in[3];
    r = ((uint32_t)in[4]<<24) | ((uint32_t)in[5]<<16) | ((uint32_t)in[6]<<8) | in[7];

    /* IP + pre-rotation (same steps as GPU) */
    t=((l>>4)^r)&0x0F0F0F0F; r^=t; l^=t<<4;
    t=((l>>16)^r)&0x0000FFFF; r^=t; l^=t<<16;
    t=((r>>2)^l)&0x33333333; l^=t; r^=t<<2;
    t=((r>>8)^l)&0x00FF00FF; l^=t; r^=t<<8;
    r=(r<<1)|(r>>31); t=(l^r)&0xAAAAAAAA; r^=t; l^=t; l=(l<<1)|(l>>31);

    /* 16 Feistel rounds */
    for (int rnd = 0; rnd < 16; rnd++) {
        uint32_t w, f;
        int idx = decrypt ? (15-rnd) : rnd;             /* reverse subkey order for decrypt */

        /* Feistel function: odd S-boxes via sk[idx*2], even via sk[idx*2+1] */
        w=(r<<28)|(r>>4); w^=sk[idx*2];
        f =SP_ALL[384+(w&0x3F)]; w>>=8; f|=SP_ALL[256+(w&0x3F)]; w>>=8;
        f|=SP_ALL[128+(w&0x3F)]; w>>=8; f|=SP_ALL[  0+(w&0x3F)];

        w=r^sk[idx*2+1];
        f|=SP_ALL[448+(w&0x3F)]; w>>=8; f|=SP_ALL[320+(w&0x3F)]; w>>=8;
        f|=SP_ALL[192+(w&0x3F)]; w>>=8; f|=SP_ALL[ 64+(w&0x3F)];

        l^=f; t=l; l=r; r=t;                            /* XOR + swap */
    }

    /* Undo last swap */
    t=l; l=r; r=t;

    /* FP (same corrected version as GPU) */
    l=(l<<31)|(l>>1); t=(l^r)&0xAAAAAAAA; r^=t; l^=t;
    r=(r<<31)|(r>>1);
    t=((r>>8)^l)&0x00FF00FF; l^=t; r^=t<<8;
    t=((r>>2)^l)&0x33333333; l^=t; r^=t<<2;
    t=((l>>16)^r)&0x0000FFFF; r^=t; l^=t<<16;
    t=((l>>4)^r)&0x0F0F0F0F; r^=t; l^=t<<4;

    /* Store output block (big-endian) */
    out[0]=(uchar)(l>>24); out[1]=(uchar)(l>>16); out[2]=(uchar)(l>>8); out[3]=(uchar)l;
    out[4]=(uchar)(r>>24); out[5]=(uchar)(r>>16); out[6]=(uchar)(r>>8); out[7]=(uchar)r;
}


/*
 * host_set_parity — same as the device version, for host-side key construction.
 */
static uint8_t host_set_parity(uint8_t b) {
    uint8_t v = b; v ^= v>>4; v ^= v>>2; v ^= v>>1;
    return (b & 0xFE) | ((v&1)^1);
}


/*
 * index_to_key — convert a 47-bit key index to an 8-byte DES key (host version).
 *
 * Same encoding as the GPU kernel: byte[0] fixed, byte[1] partially fixed,
 * bytes[2–7] fully variable.  Used by the host for verification if needed.
 */
static void index_to_key(uint64_t key_idx, uint8_t *key) {
    uint32_t k1_lower5 = (uint32_t)((key_idx >> 42) & 0x1F);
    uint32_t k1_full7 = 0x60 | k1_lower5;
    key[0] = 0xD3;
    key[1] = host_set_parity((uint8_t)(k1_full7 << 1));
    key[2] = host_set_parity((uint8_t)(((key_idx >> 35) & 0x7F) << 1));
    key[3] = host_set_parity((uint8_t)(((key_idx >> 28) & 0x7F) << 1));
    key[4] = host_set_parity((uint8_t)(((key_idx >> 21) & 0x7F) << 1));
    key[5] = host_set_parity((uint8_t)(((key_idx >> 14) & 0x7F) << 1));
    key[6] = host_set_parity((uint8_t)(((key_idx >>  7) & 0x7F) << 1));
    key[7] = host_set_parity((uint8_t)(( key_idx        & 0x7F) << 1));
}


/*
 * CUDA_CHECK — macro to check CUDA API return codes.
 *
 * Every CUDA call returns a cudaError_t.  This macro prints a human-readable
 * error message with file and line number if the call fails.  Essential for
 * diagnosing driver version mismatches, out-of-memory, invalid launch
 * configurations, etc.
 */
#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        printf("CUDA ERROR at %s:%d: %s (code %d)\n", \
               __FILE__, __LINE__, cudaGetErrorString(_e), (int)_e); \
        return 1; \
    } \
} while(0)


/* ============================================================================
 * MAIN — orchestrate the brute-force search
 * ============================================================================
 */
int main(void) {
    uint8_t ct_data[1120];		/* full ciphertext (text.d, max 1112 bytes) */
    int ct_len;				/* actual file length */
    uint32_t result_count;		/* number of candidate keys found (host copy) */
    uint8_t result_keys[64 * 8];	/* up to 64 candidate keys (host copy) */
    uint32_t *d_result_count;		/* device pointer: atomic candidate counter */
    uchar *d_result_keys;		/* device pointer: candidate key buffer */
    int num_check_blocks = 10;		/* number of blocks to check in filter 2 */

    /*
     * Search space: 2^47 ≈ 140.7 trillion keys.
     *
     * I process the search space in batches of 2^26 = 67,108,864 keys.
     * Each batch launches one CUDA grid where each thread handles one key.
     * The batch size is a trade-off:
     *   - Too small: kernel launch overhead dominates
     *   - Too large: long gaps between progress updates and result checks
     * 2^26 gives ~4.5 seconds per batch at ~3.8 Gkeys/s, a good balance.
     */
    uint64_t total_keys = (uint64_t)1 << 47;
    uint64_t batch_size = (uint64_t)1 << 26;                  /* 64M keys per batch */
    uint64_t num_batches = (total_keys + batch_size - 1) / batch_size;

    printf("GPU DES Brute-Force Cracker - Stage 9 (CUDA) - OPTIMIZED\n");
    printf("=========================================================\n");
    printf("Search space: 2^47 = %llu keys\n", (unsigned long long)total_keys);
    printf("byte[0]=0xD3, byte[1] top 2 bits=11\n\n");

    /* ---- Print GPU information ---- */
    {
        int dev;
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        printf("GPU: %s (SM %d.%d, %d SMs, %.0f MB)\n",
               prop.name, prop.major, prop.minor,
               prop.multiProcessorCount,
               prop.totalGlobalMem / (1024.0 * 1024.0));
        printf("Max threads/block: %d, Shared mem/block: %zu bytes\n\n",
               prop.maxThreadsPerBlock, prop.sharedMemPerBlock);
    }

    /* ---- Self-test: verify the host DES implementation ----
     *
     * Two tests ensure the DES code is correct before I spend hours
     * searching.  If either fails, the program aborts immediately.
     *
     * Test 1 (NIST): encrypt "Now is t" with key 0123456789ABCDEF.
     *   Expected ciphertext: 3FA40E8A984D4815  (from FIPS PUB 81)
     *
     * Test 2 (Stage 9): decrypt block 0 of text.d with key D3CD1694CBA126FE.
     *   Expected plaintext: "O! curse"  (hex: 4F21206375727365)
     *   This verifies the known-correct key against the actual ciphertext.
     *   I added this test AFTER I had decrypted the plaintext.  I optimized
     *   the code some more, and I didn't want to wait to go through
     *   two hours of computation to find out that I made a mistake.  :-)
     */
    {
        uchar key1[8] = {0x01,0x23,0x45,0x67,0x89,0xAB,0xCD,0xEF};
        uchar pt1[8]  = {0x4E,0x6F,0x77,0x20,0x69,0x73,0x20,0x74};  /* "Now is t" */
        uchar exp1[8] = {0x3F,0xA4,0x0E,0x8A,0x98,0x4D,0x48,0x15};
        uchar ct1[8];
        uint32_t sk[32];
        host_des_key_schedule(key1, sk);
        host_des_crypt(pt1, ct1, sk, 0);    /* encrypt */
        if (memcmp(ct1, exp1, 8) != 0) {
            printf("SELF-TEST FAILED: NIST encrypt\n"); return 1;
        }

        uchar key2[8] = {0xD3,0xCD,0x16,0x94,0xCB,0xA1,0x26,0xFE};
        uchar ct2[8]  = {0xb5,0x5c,0x17,0x27,0xf6,0x26,0xfc,0xca};  /* block 0 of text.d */
        uchar exp2[8] = {0x4f,0x21,0x20,0x63,0x75,0x72,0x73,0x65};   /* "O! curse" */
        uchar pt2[8];
        host_des_key_schedule(key2, sk);
        host_des_crypt(ct2, pt2, sk, 1);    /* decrypt */
        if (memcmp(pt2, exp2, 8) != 0) {
            printf("SELF-TEST FAILED: Stage 9 decrypt\n"); return 1;
        }
        printf("Host DES self-test PASSED\n");
    }

    /* ---- Load ciphertext file (text.d) ---- */
    {
        FILE *f = fopen("text.d", "rb");
        if (!f) { printf("ERROR: Cannot open text.d\n"); return 1; }
        fseek(f, 0, SEEK_END); ct_len = (int)ftell(f); fseek(f, 0, SEEK_SET);
        fread(ct_data, 1, ct_len, f); fclose(f);
    }
    printf("text.d: %d bytes (%d blocks)\n", ct_len, ct_len / 8);
    printf("Block 0: ");
    for (int i = 0; i < 8; i++) printf("%02x", ct_data[i]);
    printf("\n\n");

    /* ---- Upload data to GPU ----
     *
     * Ciphertext (first 80 bytes = 10 blocks) → __constant__ memory d_ct.
     * SP tables (2048 bytes) → __device__ global memory d_SP.
     * Result buffers → device global memory via cudaMalloc.
     */
    int ct_upload = ct_len < 80 ? ct_len : 80;
    CUDA_CHECK(cudaMemcpyToSymbol(d_ct, ct_data, ct_upload));
    CUDA_CHECK(cudaMemcpyToSymbol(d_SP, SP_ALL, sizeof(SP_ALL)));
    CUDA_CHECK(cudaMalloc(&d_result_count, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_result_keys, 64 * 8));
    printf("Device memory setup OK\n");

    /* ---- Smoke-test the kernel with a tiny launch ----
     *
     * Launch just 1 block × 256 threads to verify the kernel compiles,
     * loads, and runs without crashing.  Catches architecture mismatches,
     * register overflows, and other launch-time errors early.
     */
    {
        printf("Testing kernel with 256 threads...\n");
        result_count = 0;
        CUDA_CHECK(cudaMemcpy(d_result_count, &result_count, sizeof(uint32_t), cudaMemcpyHostToDevice));
        des_crack_kernel<<<1, 256>>>(0, num_check_blocks, d_result_count, d_result_keys);
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("Test launch OK!\n\n");
    }

    /* ---- Main brute-force loop ----
     *
     * For each batch:
     *   1. Reset the result counter on the device
     *   2. Launch a grid of (batch_size / 256) blocks × 256 threads
     *   3. Synchronise and check for CUDA errors
     *   4. Read back the result counter; if > 0, read back the candidate keys
     *   5. Every 256 batches (~17 billion keys), print a progress line
     */
    int threads_per_block = 256;
    clock_t t_start = clock();

    printf("Starting brute-force (%llu batches of %llu)...\n\n",
           (unsigned long long)num_batches, (unsigned long long)batch_size);
    fflush(stdout);

    int found = 0;
    for (uint64_t batch = 0; batch < num_batches && !found; batch++) {
        uint64_t key_offset = batch * batch_size;
        uint64_t this_batch = batch_size;
        if (key_offset + this_batch > total_keys)
            this_batch = total_keys - key_offset;          /* truncate last batch */

        /* Reset the candidate counter on the device */
        result_count = 0;
        cudaMemcpy(d_result_count, &result_count, sizeof(uint32_t), cudaMemcpyHostToDevice);

        /* Launch the kernel: one thread per key, 256 threads per block */
        int num_blocks_gpu = (int)((this_batch + threads_per_block - 1) / threads_per_block);
        des_crack_kernel<<<num_blocks_gpu, threads_per_block>>>(
            key_offset, num_check_blocks, d_result_count, d_result_keys);

        /* Wait for completion and check for errors */
        cudaError_t sync_err = cudaDeviceSynchronize();
        if (sync_err != cudaSuccess) {
            printf("CUDA ERROR: %s (code %d, batch %llu, grid %d)\n",
                   cudaGetErrorString(sync_err), (int)sync_err,
                   (unsigned long long)batch, num_blocks_gpu);
            break;
        }

        /* Check if any candidates were found in this batch */
        cudaMemcpy(&result_count, d_result_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        if (result_count > 0) {
            uint32_t to_read = result_count > 64 ? 64 : result_count;
            cudaMemcpy(result_keys, d_result_keys, to_read * 8, cudaMemcpyDeviceToHost);

            for (uint32_t i = 0; i < to_read; i++) {
                printf("\n*** CANDIDATE KEY FOUND! ***\n");
                printf("Key: ");
                for (int j = 0; j < 8; j++) printf("%02X", result_keys[i*8+j]);
                printf("\n");

                /* Quick preview: decrypt first 80 bytes on the host */
                uint32_t sk[32];
                host_des_key_schedule(result_keys + i*8, sk);
                printf("First 80 bytes: ");
                int check = (ct_len < 80) ? ct_len : 80;
                for (int blk = 0; blk + 8 <= check; blk += 8) {
                    uchar pt[8];
                    host_des_crypt(ct_data + blk, pt, sk, 1);
                    for (int j = 0; j < 8; j++) {
                        if (pt[j] >= 0x20 && pt[j] <= 0x7E) putchar(pt[j]);
                        else if (pt[j] == 0x0A) putchar('\n');
                        else printf("[%02X]", pt[j]);
                    }
                }
                printf("\n");
                found = 1;
            }
            fflush(stdout);
        }

        /* ---- Progress report (every 256 batches ≈ 17.2 billion keys) ---- */
        if ((batch & 255) == 0 || found) {
            uint64_t keys_done = (batch + 1) * batch_size;
            if (keys_done > total_keys) keys_done = total_keys;
            double elapsed = (double)(clock() - t_start) / CLOCKS_PER_SEC;
            double pct = 100.0 * keys_done / total_keys;
            double rate = elapsed > 0 ? keys_done / elapsed : 0;
            double eta = rate > 0 ? (total_keys - keys_done) / rate : 0;
            printf("[%7.0fs] %14llu / %llu keys (%5.2f%%) %.2f Gkeys/s ETA %.0fs\n",
                   elapsed, (unsigned long long)keys_done, (unsigned long long)total_keys,
                   pct, rate / 1e9, eta);
            fflush(stdout);
        }
    }

    double total_time = (double)(clock() - t_start) / CLOCKS_PER_SEC;

    /* ---- Full decryption with the found key ----
     *
     * Once a candidate passes both GPU filters, I decrypt the ENTIRE
     * ciphertext file on the host and print the plaintext.  This lets us
     * visually confirm the result is meaningful English text.
     */
    if (found) {
        printf("\n\n=== FULL DECRYPTION ===\n");
        uint32_t sk[32];
        host_des_key_schedule(result_keys, sk);
        for (int blk = 0; blk + 8 <= ct_len; blk += 8) {
            uchar pt[8];
            host_des_crypt(ct_data + blk, pt, sk, 1);
            for (int j = 0; j < 8; j++) {
                if (pt[j] >= 0x20 && pt[j] <= 0x7E) putchar(pt[j]);
                else if (pt[j] == 0x0A) putchar('\n');
                else printf("[%02X]", pt[j]);
            }
        }
        printf("\n=== END ===\n");
        printf("\nKey found in %.1f seconds (%.1f minutes)\n", total_time, total_time / 60);
    } else {
        printf("\nSearch exhausted in %.1f seconds. No key found.\n", total_time);
    }

    /* ---- Clean up device memory ---- */
    cudaFree(d_result_count);
    cudaFree(d_result_keys);
    return 0;
}
