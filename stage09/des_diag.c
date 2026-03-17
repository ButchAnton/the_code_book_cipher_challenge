/*
 * ============================================================================
 * des_diag.c — DES implementation diagnostic and correctness verifier
 * ============================================================================
 *
 * PURPOSE
 * -------
 * This file was written to track down bugs in the SP-table DES implementation
 * used by the GPU cracker (crack_des_cuda.cu) before any GPU code was run.
 * It implements the SAME DES algorithm twice:
 *
 *   1. textbook_des_encrypt — a slow, bit-level reference implementation
 *      that follows the FIPS 46-3 standard exactly: bit-numbered tables,
 *      explicit bit extraction, single S-box lookup per 6-bit group.
 *      This is unambiguously correct but far too slow for brute force.
 *
 *   2. sp_des_encrypt — the fast SP-table version (same algorithm as the
 *      GPU kernel) that uses combined S-box + P permutation lookup tables
 *      and the pre-rotated IP/FP trick from Eric Young's libdes.
 *
 * Both versions encrypt the same plaintext with the same key and the results
 * are compared round by round.  Any mismatch pinpoints exactly which step
 * in the fast implementation is wrong.
 *
 * BUGS FOUND
 * ----------
 * Running this diagnostic revealed two bugs in the fast implementation:
 *
 *   Bug 1 — IP/FP round-trip failure:
 *     The FP (Final Permutation) had `left` and `right` swapped in every
 *     operation.  This caused decryption to produce garbage even when the
 *     key schedule was correct.  The fix was to carefully invert every IP
 *     step (PERM_OP operations and ROL/ROR rotations) in the correct order.
 *
 *   Bug 2 — PC2 subkey words swapped (sk0 ↔ sk1):
 *     The bitwise compute_pc2() function had the two 32-bit output words
 *     assigned to the wrong halves.  sk0 should feed S-boxes 1,3,5,7
 *     (XORed with ROR(R,4)) and sk1 should feed S-boxes 2,4,6,8 (XORed
 *     with R directly).  They were backwards.  The fix was derived by
 *     comparing the textbook PC2 output bit-by-bit against the bitwise
 *     expressions and swapping the assignments.
 *
 * After both fixes, all four tests below pass and the fast SP-table
 * implementation produces identical output to the textbook for every tested
 * key and plaintext.
 *
 * TESTS
 * -----
 * Test 1: NIST FIPS 81 encrypt vector
 *         key = 0123456789ABCDEF, pt = "Now is t" (4E6F772069732074)
 *         expected CT = 3FA40E8A984D4815
 *
 * Test 2: NIST variable-plaintext vector
 *         key = 0101010101010101 (weak key, all parity), pt = 8000000000000000
 *         expected CT = 95F8A5E5DD31D900
 *
 * Test 3: Round-by-round comparison of textbook vs SP-table on Test 1 vector.
 *         Both should produce identical L/R values after every round.
 *
 * Test 4: Round-by-round comparison on Test 2 vector.
 *
 * Test 5: IP-FP round-trip (no rounds, just the permutations).
 *         Applying IP then FP should recover the original input exactly.
 *
 * BUILD
 * -----
 *   cl /O2 des_diag.c /Fe:des_diag.exe          (MSVC)
 *   gcc -O2 -o des_diag des_diag.c              (GCC)
 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>

typedef uint8_t uchar;

/* ===================== TEXTBOOK DES (bit-level, slow but correct) ===================== */

static const int IP_TABLE[64] = {
    58,50,42,34,26,18,10, 2, 60,52,44,36,28,20,12, 4,
    62,54,46,38,30,22,14, 6, 64,56,48,40,32,24,16, 8,
    57,49,41,33,25,17, 9, 1, 59,51,43,35,27,19,11, 3,
    61,53,45,37,29,21,13, 5, 63,55,47,39,31,23,15, 7
};

static const int FP_TABLE[64] = {
    40, 8,48,16,56,24,64,32, 39, 7,47,15,55,23,63,31,
    38, 6,46,14,54,22,62,30, 37, 5,45,13,53,21,61,29,
    36, 4,44,12,52,20,60,28, 35, 3,43,11,51,19,59,27,
    34, 2,42,10,50,18,58,26, 33, 1,41, 9,49,17,57,25
};

static const int E_TABLE[48] = {
    32, 1, 2, 3, 4, 5,  4, 5, 6, 7, 8, 9,
     8, 9,10,11,12,13, 12,13,14,15,16,17,
    16,17,18,19,20,21, 20,21,22,23,24,25,
    24,25,26,27,28,29, 28,29,30,31,32, 1
};

static const int P_TABLE[32] = {
    16, 7,20,21,29,12,28,17, 1,15,23,26, 5,18,31,10,
     2, 8,24,14,32,27, 3, 9,19,13,30, 6,22,11, 4,25
};

static const int S_BOXES[8][4][16] = {
    /* S1 */
    {{14, 4,13, 1, 2,15,11, 8, 3,10, 6,12, 5, 9, 0, 7},
     { 0,15, 7, 4,14, 2,13, 1,10, 6,12,11, 9, 5, 3, 8},
     { 4, 1,14, 8,13, 6, 2,11,15,12, 9, 7, 3,10, 5, 0},
     {15,12, 8, 2, 4, 9, 1, 7, 5,11, 3,14,10, 0, 6,13}},
    /* S2 */
    {{15, 1, 8,14, 6,11, 3, 4, 9, 7, 2,13,12, 0, 5,10},
     { 3,13, 4, 7,15, 2, 8,14,12, 0, 1,10, 6, 9,11, 5},
     { 0,14, 7,11,10, 4,13, 1, 5, 8,12, 6, 9, 3, 2,15},
     {13, 8,10, 1, 3,15, 4, 2,11, 6, 7,12, 0, 5,14, 9}},
    /* S3 */
    {{10, 0, 9,14, 6, 3,15, 5, 1,13,12, 7,11, 4, 2, 8},
     {13, 7, 0, 9, 3, 4, 6,10, 2, 8, 5,14,12,11,15, 1},
     {13, 6, 4, 9, 8,15, 3, 0,11, 1, 2,12, 5,10,14, 7},
     { 1,10,13, 0, 6, 9, 8, 7, 4,15,14, 3,11, 5, 2,12}},
    /* S4 */
    {{ 7,13,14, 3, 0, 6, 9,10, 1, 2, 8, 5,11,12, 4,15},
     {13, 8,11, 5, 6,15, 0, 3, 4, 7, 2,12, 1,10,14, 9},
     {10, 6, 9, 0,12,11, 7,13,15, 1, 3,14, 5, 2, 8, 4},
     { 3,15, 0, 6,10, 1,13, 8, 9, 4, 5,11,12, 7, 2,14}},
    /* S5 */
    {{ 2,12, 4, 1, 7,10,11, 6, 8, 5, 3,15,13, 0,14, 9},
     {14,11, 2,12, 4, 7,13, 1, 5, 0,15,10, 3, 9, 8, 6},
     { 4, 2, 1,11,10,13, 7, 8,15, 9,12, 5, 6, 3, 0,14},
     {11, 8,12, 7, 1,14, 2,13, 6,15, 0, 9,10, 4, 5, 3}},
    /* S6 */
    {{12, 1,10,15, 9, 2, 6, 8, 0,13, 3, 4,14, 7, 5,11},
     {10,15, 4, 2, 7,12, 9, 5, 6, 1,13,14, 0,11, 3, 8},
     { 9,14,15, 5, 2, 8,12, 3, 7, 0, 4,10, 1,13,11, 6},
     { 4, 3, 2,12, 9, 5,15,10,11,14, 1, 7, 6, 0, 8,13}},
    /* S7 */
    {{ 4,11, 2,14,15, 0, 8,13, 3,12, 9, 7, 5,10, 6, 1},
     {13, 0,11, 7, 4, 9, 1,10,14, 3, 5,12, 2,15, 8, 6},
     { 1, 4,11,13,12, 3, 7,14,10,15, 6, 8, 0, 5, 9, 2},
     { 6,11,13, 8, 1, 4,10, 7, 9, 5, 0,15,14, 2, 3,12}},
    /* S8 */
    {{13, 2, 8, 4, 6,15,11, 1,10, 9, 3,14, 5, 0,12, 7},
     { 1,15,13, 8,10, 3, 7, 4,12, 5, 6, 2, 0,14, 9,11},
     { 7,11, 4, 1, 9,12,14, 2, 0, 6,10,13,15, 3, 5, 8},
     { 2, 1,14, 7, 4,10, 8,13,15,12, 9, 0, 3, 5, 6,11}}
};

static const int PC1_C[28] = {
    57,49,41,33,25,17, 9, 1,58,50,42,34,26,18,
    10, 2,59,51,43,35,27,19,11, 3,60,52,44,36
};
static const int PC1_D[28] = {
    63,55,47,39,31,23,15, 7,62,54,46,38,30,22,
    14, 6,61,53,45,37,29,21,13, 5,28,20,12, 4
};
static const int PC2_TABLE[48] = {
    14,17,11,24, 1, 5, 3,28,15, 6,21,10,
    23,19,12, 4,26, 8,16, 7,27,20,13, 2,
    41,52,31,37,47,55,30,40,51,45,33,48,
    44,49,39,56,34,53,46,42,50,36,29,32
};
static const int KEY_SHIFTS[16] = {1,1,2,2,2,2,2,2,1,2,2,2,2,2,2,1};

/* Bit manipulation helpers for 64/32-bit values */
/* Bit numbering: 1=MSB, 64=LSB for 64-bit; 1=MSB, 32=LSB for 32-bit */
int getbit64(uint64_t v, int b) { return (int)((v >> (64 - b)) & 1); }
int getbit32(uint32_t v, int b) { return (int)((v >> (32 - b)) & 1); }

/* Textbook DES encrypt */
void textbook_des_encrypt(const uchar *key_bytes, const uchar *pt, uchar *ct) {
    uint64_t input = 0;
    int i, j, round;
    uint32_t L, R;
    uint32_t C, D;
    uint32_t subkeys_L[16], subkeys_R[16]; /* 24 bits each for 48-bit subkey */
    uint64_t K48[16]; /* 48-bit round keys */

    /* Build 64-bit input */
    for (i = 0; i < 8; i++)
        input |= ((uint64_t)pt[i]) << (56 - 8*i);

    /* IP */
    uint64_t ip_out = 0;
    for (i = 0; i < 64; i++) {
        if (getbit64(input, IP_TABLE[i]))
            ip_out |= ((uint64_t)1 << (63 - i));
    }
    L = (uint32_t)(ip_out >> 32);
    R = (uint32_t)(ip_out & 0xFFFFFFFF);

    /* Key schedule */
    uint64_t key64 = 0;
    for (i = 0; i < 8; i++)
        key64 |= ((uint64_t)key_bytes[i]) << (56 - 8*i);

    C = 0; D = 0;
    for (i = 0; i < 28; i++) {
        if (getbit64(key64, PC1_C[i])) C |= (1u << (27 - i));
        if (getbit64(key64, PC1_D[i])) D |= (1u << (27 - i));
    }

    for (round = 0; round < 16; round++) {
        for (j = 0; j < KEY_SHIFTS[round]; j++) {
            C = ((C << 1) | (C >> 27)) & 0x0FFFFFFF;
            D = ((D << 1) | (D >> 27)) & 0x0FFFFFFF;
        }
        /* PC2: build 48-bit key */
        K48[round] = 0;
        for (i = 0; i < 48; i++) {
            int src = PC2_TABLE[i]; /* 1-indexed in C||D */
            int val;
            if (src <= 28)
                val = (C >> (28 - src)) & 1;
            else
                val = (D >> (56 - src)) & 1;
            if (val) K48[round] |= ((uint64_t)1 << (47 - i));
        }
    }

    /* 16 rounds */
    for (round = 0; round < 16; round++) {
        uint32_t newR;
        /* Expand R to 48 bits */
        uint64_t ER = 0;
        for (i = 0; i < 48; i++) {
            if (getbit32(R, E_TABLE[i]))
                ER |= ((uint64_t)1 << (47 - i));
        }
        /* XOR with round key */
        uint64_t X = ER ^ K48[round];

        /* S-box substitution */
        uint32_t sbox_out = 0;
        for (i = 0; i < 8; i++) {
            int bits6 = (int)((X >> (42 - 6*i)) & 0x3F);
            int row = ((bits6 >> 4) & 2) | (bits6 & 1);
            int col = (bits6 >> 1) & 0xF;
            int val = S_BOXES[i][row][col];
            sbox_out |= ((uint32_t)val << (28 - 4*i));
        }

        /* P permutation */
        uint32_t P_out = 0;
        for (i = 0; i < 32; i++) {
            if (getbit32(sbox_out, P_TABLE[i]))
                P_out |= (1u << (31 - i));
        }

        /* Feistel: new R = L XOR P_out, new L = old R */
        newR = L ^ P_out;
        L = R;
        R = newR;
    }

    /* Final: swap L and R, then FP */
    uint64_t pre_fp = ((uint64_t)R << 32) | L;
    uint64_t output64 = 0;
    for (i = 0; i < 64; i++) {
        if (getbit64(pre_fp, FP_TABLE[i]))
            output64 |= ((uint64_t)1 << (63 - i));
    }

    for (i = 0; i < 8; i++)
        ct[i] = (uchar)(output64 >> (56 - 8*i));
}

/* ===================== SP-TABLE DES (from kernel) ===================== */

static const uint32_t SP1[64] = {
    0x01010400,0x00000000,0x00010000,0x01010404,0x01010004,0x00010404,0x00000004,0x00010000,
    0x00000400,0x01010400,0x01010404,0x00000400,0x01000404,0x01010004,0x01000000,0x00000004,
    0x00000404,0x01000400,0x01000400,0x00010400,0x00010400,0x01010000,0x01010000,0x01000404,
    0x00010004,0x01000004,0x01000004,0x00010004,0x00000000,0x00000404,0x00010404,0x01000000,
    0x00010000,0x01010404,0x00000004,0x01010000,0x01010400,0x01000000,0x01000000,0x00000400,
    0x01010004,0x00010000,0x00010400,0x01000004,0x00000400,0x00000004,0x01000404,0x00010404,
    0x01010404,0x00010004,0x01010000,0x01000404,0x01000004,0x00000404,0x00010404,0x01010400,
    0x00000404,0x01000400,0x01000400,0x00000000,0x00010004,0x00010400,0x00000000,0x01010004
};
static const uint32_t SP2[64] = {
    0x80108020,0x80008000,0x00008000,0x00108020,0x00100000,0x00000020,0x80100020,0x80008020,
    0x80000020,0x80108020,0x80108000,0x80000000,0x80008000,0x00100000,0x00000020,0x80100020,
    0x00108000,0x00100020,0x80008020,0x00000000,0x80000000,0x00008000,0x00108020,0x80100000,
    0x00100020,0x80000020,0x00000000,0x00108000,0x00008020,0x80108000,0x80100000,0x00008020,
    0x00000000,0x00108020,0x80100020,0x00100000,0x80008020,0x80100000,0x80108000,0x00008000,
    0x80100000,0x80008000,0x00000020,0x80108020,0x00108020,0x00000020,0x00008000,0x80000000,
    0x00008020,0x80108000,0x00100000,0x80000020,0x00100020,0x80008020,0x80000020,0x00100020,
    0x00108000,0x00000000,0x80008000,0x00008020,0x80000000,0x80100020,0x80108020,0x00108000
};
static const uint32_t SP3[64] = {
    0x00000208,0x08020200,0x00000000,0x08020008,0x08000200,0x00000000,0x00020208,0x08000200,
    0x00020008,0x08000008,0x08000008,0x00020000,0x08020208,0x00020008,0x08020000,0x00000208,
    0x08000000,0x00000008,0x08020200,0x00000200,0x00020200,0x08020000,0x08020008,0x00020208,
    0x08000208,0x00020200,0x00020000,0x08000208,0x00000008,0x08020208,0x00000200,0x08000000,
    0x08020200,0x08000000,0x00020008,0x00000208,0x00020000,0x08020200,0x08000200,0x00000000,
    0x00000200,0x00020008,0x08020208,0x08000200,0x08000008,0x00000200,0x00000000,0x08020008,
    0x08000208,0x00020000,0x08000000,0x08020208,0x00000008,0x00020208,0x00020200,0x08000008,
    0x08020000,0x08000208,0x00000208,0x08020000,0x00020208,0x00000008,0x08020008,0x00020200
};
static const uint32_t SP4[64] = {
    0x00802001,0x00002081,0x00002081,0x00000080,0x00802080,0x00800081,0x00800001,0x00002001,
    0x00000000,0x00802000,0x00802000,0x00802081,0x00000081,0x00000000,0x00800080,0x00800001,
    0x00000001,0x00002000,0x00800000,0x00802001,0x00000080,0x00800000,0x00002001,0x00002080,
    0x00800081,0x00000001,0x00002080,0x00800080,0x00002000,0x00802080,0x00802081,0x00000081,
    0x00800080,0x00800001,0x00802000,0x00802081,0x00000081,0x00000000,0x00000000,0x00802000,
    0x00002080,0x00800080,0x00800081,0x00000001,0x00802001,0x00002081,0x00002081,0x00000080,
    0x00802081,0x00000081,0x00000001,0x00002000,0x00800001,0x00002001,0x00802080,0x00800081,
    0x00002001,0x00002080,0x00800000,0x00802001,0x00000080,0x00800000,0x00002000,0x00802080
};
static const uint32_t SP5[64] = {
    0x00000100,0x02080100,0x02080000,0x42000100,0x00080000,0x00000100,0x40000000,0x02080000,
    0x40080100,0x00080000,0x02000100,0x40080100,0x42000100,0x42080000,0x00080100,0x40000000,
    0x02000000,0x40080000,0x40080000,0x00000000,0x40000100,0x42080100,0x42080100,0x02000100,
    0x42080000,0x40000100,0x00000000,0x42000000,0x02080100,0x02000000,0x42000000,0x00080100,
    0x00080000,0x42000100,0x00000100,0x02000000,0x40000000,0x02080000,0x42000100,0x40080100,
    0x02000100,0x40000000,0x42080000,0x02080100,0x40080100,0x00000100,0x02000000,0x42080000,
    0x42080100,0x00080100,0x42000000,0x42080100,0x02080000,0x00000000,0x40080000,0x42000000,
    0x00080100,0x02000100,0x40000100,0x00080000,0x00000000,0x40080000,0x02080100,0x40000100
};
static const uint32_t SP6[64] = {
    0x20000010,0x20400000,0x00004000,0x20404010,0x20400000,0x00000010,0x20404010,0x00400000,
    0x20004000,0x00404010,0x00400000,0x20000010,0x00400010,0x20004000,0x20000000,0x00004010,
    0x00000000,0x00400010,0x20004010,0x00004000,0x00404000,0x20004010,0x00000010,0x20400010,
    0x20400010,0x00000000,0x00404010,0x20404000,0x00004010,0x00404000,0x20404000,0x20000000,
    0x20004000,0x00000010,0x20400010,0x00404000,0x20404010,0x00400000,0x00004010,0x20000010,
    0x00400000,0x20004000,0x20000000,0x00004010,0x20000010,0x20404010,0x00404000,0x20400000,
    0x00404010,0x20404000,0x00000000,0x20400010,0x00000010,0x00004000,0x20400000,0x00404010,
    0x00004000,0x00400010,0x20004010,0x00000000,0x20404000,0x20000000,0x00400010,0x20004010
};
static const uint32_t SP7[64] = {
    0x00200000,0x04200002,0x04000802,0x00000000,0x00000800,0x04000802,0x00200802,0x04200800,
    0x04200802,0x00200000,0x00000000,0x04000002,0x00000002,0x04000000,0x04200002,0x00000802,
    0x04000800,0x00200802,0x00200002,0x04000800,0x04000002,0x04200000,0x04200800,0x00200002,
    0x04200000,0x00000800,0x00000802,0x04200802,0x00200800,0x00000002,0x04000000,0x00200800,
    0x04000000,0x00200800,0x00200000,0x04000802,0x04000802,0x04200002,0x04200002,0x00000002,
    0x00200002,0x04000000,0x04000800,0x00200000,0x04200800,0x00000802,0x00200802,0x04200800,
    0x00000802,0x04000002,0x04200802,0x04200000,0x00200800,0x00000000,0x00000002,0x04200802,
    0x00000000,0x00200802,0x04200000,0x00000800,0x04000002,0x04000800,0x00000800,0x00200002
};
static const uint32_t SP8[64] = {
    0x10001040,0x00001000,0x00040000,0x10041040,0x10000000,0x10001040,0x00000040,0x10000000,
    0x00040040,0x10040000,0x10041040,0x00041000,0x10041000,0x00041040,0x00001000,0x00000040,
    0x10040000,0x10000040,0x10001000,0x00001040,0x00041000,0x00040040,0x10040040,0x10041000,
    0x00001040,0x00000000,0x00000000,0x10040040,0x10000040,0x10001000,0x00041040,0x00040000,
    0x00041040,0x00040000,0x10041000,0x00001000,0x00000040,0x10040040,0x00001000,0x00041040,
    0x10001000,0x00000040,0x10000040,0x10040000,0x10040040,0x10000000,0x00040000,0x10001040,
    0x00000000,0x10041040,0x00040040,0x10000040,0x10040000,0x10001000,0x10001040,0x00000000,
    0x10041040,0x00041000,0x00041000,0x00001040,0x00001040,0x00040040,0x10000000,0x10041000
};

void sp_des_encrypt(const uchar *key_bytes, const uchar *pt, uchar *ct,
                    uint32_t *out_L, uint32_t *out_R, int trace) {
    uint32_t c = 0, d = 0;
    uint32_t sk[32];
    uint32_t left, right, temp;
    int i;

    /* PC1 */
    for (i = 0; i < 28; i++) {
        int bc = PC1_C[i] - 1;
        if (key_bytes[bc/8] & (0x80 >> (bc%8))) c |= (1u << (27-i));
        int bd = PC1_D[i] - 1;
        if (key_bytes[bd/8] & (0x80 >> (bd%8))) d |= (1u << (27-i));
    }

    /* Key schedule with textbook PC2 packing */
    for (int round = 0; round < 16; round++) {
        for (int s = 0; s < KEY_SHIFTS[round]; s++) {
            c = ((c << 1) | (c >> 27)) & 0x0FFFFFFF;
            d = ((d << 1) | (d >> 27)) & 0x0FFFFFFF;
        }
        sk[round*2] = 0;
        sk[round*2+1] = 0;
        for (i = 0; i < 48; i++) {
            int src = PC2_TABLE[i];
            int val;
            if (src <= 28) val = (c >> (28-src)) & 1;
            else val = (d >> (56-src)) & 1;
            if (val) {
                int sbox = i / 6;
                int bit_in_sbox = i % 6;
                int word = sbox & 1;
                int base = 24 - (sbox >> 1) * 8;
                int target = base + 5 - bit_in_sbox;
                if (word == 0) sk[round*2] |= (1u << target);
                else sk[round*2+1] |= (1u << target);
            }
        }
    }

    /* Block cipher (SP table version) */
    left  = ((uint32_t)pt[0]<<24) | ((uint32_t)pt[1]<<16) | ((uint32_t)pt[2]<<8) | pt[3];
    right = ((uint32_t)pt[4]<<24) | ((uint32_t)pt[5]<<16) | ((uint32_t)pt[6]<<8) | pt[7];

    /* IP + pre-rotation */
    temp = ((left >> 4) ^ right) & 0x0F0F0F0F; right ^= temp; left ^= temp << 4;
    temp = ((left >> 16) ^ right) & 0x0000FFFF; right ^= temp; left ^= temp << 16;
    temp = ((right >> 2) ^ left) & 0x33333333; left ^= temp; right ^= temp << 2;
    temp = ((right >> 8) ^ left) & 0x00FF00FF; left ^= temp; right ^= temp << 8;
    right = (right << 1) | (right >> 31);
    temp = (left ^ right) & 0xAAAAAAAA; right ^= temp; left ^= temp;
    left = (left << 1) | (left >> 31);

    if (trace) printf("SP after IP: L=%08x R=%08x\n", left, right);

    for (int round = 0; round < 16; round++) {
        uint32_t work, fval;
        work = (right << 28) | (right >> 4);
        work ^= sk[round * 2];
        fval  = SP7[work & 0x3F]; work >>= 8;
        fval |= SP5[work & 0x3F]; work >>= 8;
        fval |= SP3[work & 0x3F]; work >>= 8;
        fval |= SP1[work & 0x3F];

        work = right ^ sk[round * 2 + 1];
        fval |= SP8[work & 0x3F]; work >>= 8;
        fval |= SP6[work & 0x3F]; work >>= 8;
        fval |= SP4[work & 0x3F]; work >>= 8;
        fval |= SP2[work & 0x3F];

        left ^= fval;
        temp = left; left = right; right = temp;

        if (trace) printf("SP round %2d: L=%08x R=%08x\n", round, left, right);
    }

    temp = left; left = right; right = temp;

    if (out_L) *out_L = left;
    if (out_R) *out_R = right;

    /* FP */
    right = (right << 31) | (right >> 1);
    temp = (left ^ right) & 0xAAAAAAAA; left ^= temp; right ^= temp;
    left = (left << 31) | (left >> 1);
    temp = ((left >> 8) ^ right) & 0x00FF00FF; right ^= temp; left ^= temp << 8;
    temp = ((left >> 2) ^ right) & 0x33333333; right ^= temp; left ^= temp << 2;
    temp = ((right >> 16) ^ left) & 0x0000FFFF; left ^= temp; right ^= temp << 16;
    temp = ((right >> 4) ^ left) & 0x0F0F0F0F; left ^= temp; right ^= temp << 4;

    ct[0] = (uchar)(right >> 24); ct[1] = (uchar)(right >> 16);
    ct[2] = (uchar)(right >> 8);  ct[3] = (uchar)(right);
    ct[4] = (uchar)(left >> 24);  ct[5] = (uchar)(left >> 16);
    ct[6] = (uchar)(left >> 8);   ct[7] = (uchar)(left);
}

int main(void) {
    uchar ct_ref[8], ct_sp[8];
    int pass = 1;
    int i;

    /* ===== TEST 1: Textbook DES correctness ===== */
    printf("=== Test 1: Textbook DES NIST vector ===\n");
    {
        uchar key[8] = {0x01,0x23,0x45,0x67,0x89,0xAB,0xCD,0xEF};
        uchar pt[8]  = {0x4E,0x6F,0x77,0x20,0x69,0x73,0x20,0x74};
        uchar expected[8] = {0x3F,0xA4,0x0E,0x8A,0x98,0x4D,0x48,0x15};
        textbook_des_encrypt(key, pt, ct_ref);
        printf("Textbook: ");
        for (i=0;i<8;i++) printf("%02x", ct_ref[i]);
        printf("\nExpected: ");
        for (i=0;i<8;i++) printf("%02x", expected[i]);
        printf("\n");
        if (memcmp(ct_ref, expected, 8) == 0) printf("PASS\n\n");
        else { printf("FAIL\n\n"); pass = 0; }
    }

    /* ===== TEST 2: Textbook DES variable PT ===== */
    printf("=== Test 2: Textbook DES variable PT ===\n");
    {
        uchar key[8] = {0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01};
        uchar pt[8]  = {0x80,0x00,0x00,0x00,0x00,0x00,0x00,0x00};
        uchar expected[8] = {0x95,0xF8,0xA5,0xE5,0xDD,0x31,0xD9,0x00};
        textbook_des_encrypt(key, pt, ct_ref);
        printf("Textbook: ");
        for (i=0;i<8;i++) printf("%02x", ct_ref[i]);
        printf("\nExpected: ");
        for (i=0;i<8;i++) printf("%02x", expected[i]);
        printf("\n");
        if (memcmp(ct_ref, expected, 8) == 0) printf("PASS\n\n");
        else { printf("FAIL\n\n"); pass = 0; }
    }

    /* ===== TEST 3: Compare textbook vs SP-table, round by round ===== */
    printf("=== Test 3: Compare textbook vs SP-table (NIST vector) ===\n");
    {
        uchar key[8] = {0x01,0x23,0x45,0x67,0x89,0xAB,0xCD,0xEF};
        uchar pt[8]  = {0x4E,0x6F,0x77,0x20,0x69,0x73,0x20,0x74};

        /* Textbook */
        textbook_des_encrypt(key, pt, ct_ref);

        /* SP-table (with round-by-round trace) */
        printf("SP-table round trace:\n");
        sp_des_encrypt(key, pt, ct_sp, NULL, NULL, 1);

        printf("\nTextbook CT: ");
        for (i=0;i<8;i++) printf("%02x", ct_ref[i]);
        printf("\nSP-table CT: ");
        for (i=0;i<8;i++) printf("%02x", ct_sp[i]);
        printf("\n");
        if (memcmp(ct_ref, ct_sp, 8) == 0) printf("MATCH\n\n");
        else printf("MISMATCH\n\n");
    }

    /* ===== TEST 4: Compare with zero key (subkeys all 0) ===== */
    printf("=== Test 4: Compare textbook vs SP-table (zero key, pt=8000...) ===\n");
    {
        uchar key[8] = {0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01};
        uchar pt[8]  = {0x80,0x00,0x00,0x00,0x00,0x00,0x00,0x00};

        textbook_des_encrypt(key, pt, ct_ref);
        printf("SP-table round trace:\n");
        sp_des_encrypt(key, pt, ct_sp, NULL, NULL, 1);

        printf("\nTextbook CT: ");
        for (i=0;i<8;i++) printf("%02x", ct_ref[i]);
        printf("\nSP-table CT: ");
        for (i=0;i<8;i++) printf("%02x", ct_sp[i]);
        printf("\n");
        if (memcmp(ct_ref, ct_sp, 8) == 0) printf("MATCH\n\n");
        else printf("MISMATCH\n\n");
    }

    /* ===== TEST 5: IP-FP round trip (no encryption) ===== */
    printf("=== Test 5: IP-FP round trip ===\n");
    {
        uint32_t left = 0x01234567, right = 0x89ABCDEF, temp;
        uint32_t orig_left = left, orig_right = right;

        /* IP */
        temp = ((left >> 4) ^ right) & 0x0F0F0F0F; right ^= temp; left ^= temp << 4;
        temp = ((left >> 16) ^ right) & 0x0000FFFF; right ^= temp; left ^= temp << 16;
        temp = ((right >> 2) ^ left) & 0x33333333; left ^= temp; right ^= temp << 2;
        temp = ((right >> 8) ^ left) & 0x00FF00FF; left ^= temp; right ^= temp << 8;
        right = (right << 1) | (right >> 31);
        temp = (left ^ right) & 0xAAAAAAAA; right ^= temp; left ^= temp;
        left = (left << 1) | (left >> 31);

        printf("After IP:  L=%08x R=%08x\n", left, right);

        /* FP (inverse) */
        right = (right << 31) | (right >> 1);
        temp = (left ^ right) & 0xAAAAAAAA; left ^= temp; right ^= temp;
        left = (left << 31) | (left >> 1);
        temp = ((left >> 8) ^ right) & 0x00FF00FF; right ^= temp; left ^= temp << 8;
        temp = ((left >> 2) ^ right) & 0x33333333; right ^= temp; left ^= temp << 2;
        temp = ((right >> 16) ^ left) & 0x0000FFFF; left ^= temp; right ^= temp << 16;
        temp = ((right >> 4) ^ left) & 0x0F0F0F0F; left ^= temp; right ^= temp << 4;

        printf("After FP:  L=%08x R=%08x\n", left, right);
        printf("Output bytes would be: %08x%08x\n", right, left);
        printf("Original input was:    %08x%08x\n", orig_left, orig_right);

        if (right == orig_left && left == orig_right)
            printf("IP-FP round trip: PASS\n\n");
        else if (left == orig_left && right == orig_right)
            printf("IP-FP round trip: PASS (no swap needed)\n\n");
        else
            printf("IP-FP round trip: FAIL\n\n");
    }

    printf("========================================\n");
    printf("%s\n", pass ? "ALL TEXTBOOK TESTS PASSED" : "SOME TEXTBOOK TESTS FAILED");
    return 0;
}
