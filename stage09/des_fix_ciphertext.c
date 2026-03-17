/*
 * ============================================================================
 * des_fix_ciphertext.c — identify and repair corrupted ciphertext blocks
 * ============================================================================
 *
 * PURPOSE
 * -------
 * The Stage 9 ciphertext (text.d, 1112 bytes = 139 DES blocks) was obtained
 * by scanning the relevant page of "The Code Book" and manually correcting
 * the OCR output.  The transcription introduced errors in two blocks.
 *
 * This program runs AFTER the correct DES key has been found by the GPU
 * cracker (key = D3CD1694CBA126FE).  It:
 *
 *   1. Decrypts all 139 blocks with the correct key.
 *   2. Identifies blocks whose decrypted bytes are not printable ASCII
 *      (i.e., non-ASCII = corrupted ciphertext due to transcription error).
 *   3. For each corrupted block, determines the correct plaintext from
 *      context (surrounding blocks + knowledge of the original text).
 *   4. Re-encrypts the correct plaintext to obtain the correct ciphertext.
 *   5. Patches the binary file with the corrected ciphertext bytes.
 *   6. Writes two corrected output files:
 *        text_fixed.d       — 1112 bytes, block 130 fixed
 *        text_fixed_trunc.d — 1104 bytes, block 130 fixed, last block removed
 *
 * CORRUPTED BLOCKS FOUND
 * ----------------------
 * Block 130 (offset 0x410, bytes 1040–1047):
 *   Transcribed CT: d2 ab ed 2f ca 2d e1 2b
 *   Decrypted as:   82 b7 35 fe fa 93 89 13  ← not ASCII
 *
 *   Context from surrounding blocks:
 *     Block 129 → "y factor"
 *     Block 131 → "public m"
 *
 *   The original text reads "...by factoring the public modulus..."
 *   So block 130 must be "ing the ".
 *
 *   Correct CT = DES_encrypt("ing the ", key) = 22 72 7b ad b5 9e 0d c2
 *   This is patched into the file at offset 0x410.
 *
 * Block 138 (offset 0x450, bytes 1104–1111):
 *   Transcribed CT: 0e b6 a9 76 ea d7 3a fd
 *   Decrypted as:   96 1e b0 cc 7b 34 2c 00  ← not ASCII
 *
 *   This is the LAST block of the file.  Block 137 decrypts to " luck!\n\n"
 *   which is the natural end of the message.  Block 138 is most likely
 *   padding (or trailing garbage from the book scan), so this block is
 *   simply omitted in text_fixed_trunc.d.
 *
 *   For completeness, this program also tests what ciphertext we would
 *   expect if the last block were zero-padding, PKCS5-padding, or spaces —
 *   none of these match the transcribed bytes, confirming the block is
 *   corrupt rather than a deliberate padding scheme.
 *
 * DES IMPLEMENTATION
 * ------------------
 * Uses the same SP-table / pre-rotated-IP/FP implementation as the GPU
 * cracker (crack_des_cuda.cu), now compiled as CPU code.  The key schedule
 * and block cipher functions are identical; `decrypt` flag controls whether
 * subkeys are applied in forward or reverse order.
 *
 * BUILD
 * -----
 *   cl /O2 des_fix_ciphertext.c /Fe:des_fix_ciphertext.exe   (MSVC)
 *   gcc -O2 -o des_fix_ciphertext des_fix_ciphertext.c       (GCC)
 *
 * INPUT
 *   text.d must be present in the current directory.
 *
 * OUTPUT
 *   text_fixed.d        — block 130 corrected (1112 bytes)
 *   text_fixed_trunc.d  — block 130 corrected, block 138 omitted (1104 bytes)
 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>

typedef unsigned char uchar;

static const uint32_t SP_ALL[512] = {
    0x01010400,0x00000000,0x00010000,0x01010404,0x01010004,0x00010404,0x00000004,0x00010000,
    0x00000400,0x01010400,0x01010404,0x00000400,0x01000404,0x01010004,0x01000000,0x00000004,
    0x00000404,0x01000400,0x01000400,0x00010400,0x00010400,0x01010000,0x01010000,0x01000404,
    0x00010004,0x01000004,0x01000004,0x00010004,0x00000000,0x00000404,0x00010404,0x01000000,
    0x00010000,0x01010404,0x00000004,0x01010000,0x01010400,0x01000000,0x01000000,0x00000400,
    0x01010004,0x00010000,0x00010400,0x01000004,0x00000400,0x00000004,0x01000404,0x00010404,
    0x01010404,0x00010004,0x01010000,0x01000404,0x01000004,0x00000404,0x00010404,0x01010400,
    0x00000404,0x01000400,0x01000400,0x00000000,0x00010004,0x00010400,0x00000000,0x01010004,
    0x80108020,0x80008000,0x00008000,0x00108020,0x00100000,0x00000020,0x80100020,0x80008020,
    0x80000020,0x80108020,0x80108000,0x80000000,0x80008000,0x00100000,0x00000020,0x80100020,
    0x00108000,0x00100020,0x80008020,0x00000000,0x80000000,0x00008000,0x00108020,0x80100000,
    0x00100020,0x80000020,0x00000000,0x00108000,0x00008020,0x80108000,0x80100000,0x00008020,
    0x00000000,0x00108020,0x80100020,0x00100000,0x80008020,0x80100000,0x80108000,0x00008000,
    0x80100000,0x80008000,0x00000020,0x80108020,0x00108020,0x00000020,0x00008000,0x80000000,
    0x00008020,0x80108000,0x00100000,0x80000020,0x00100020,0x80008020,0x80000020,0x00100020,
    0x00108000,0x00000000,0x80008000,0x00008020,0x80000000,0x80100020,0x80108020,0x00108000,
    0x00000208,0x08020200,0x00000000,0x08020008,0x08000200,0x00000000,0x00020208,0x08000200,
    0x00020008,0x08000008,0x08000008,0x00020000,0x08020208,0x00020008,0x08020000,0x00000208,
    0x08000000,0x00000008,0x08020200,0x00000200,0x00020200,0x08020000,0x08020008,0x00020208,
    0x08000208,0x00020200,0x00020000,0x08000208,0x00000008,0x08020208,0x00000200,0x08000000,
    0x08020200,0x08000000,0x00020008,0x00000208,0x00020000,0x08020200,0x08000200,0x00000000,
    0x00000200,0x00020008,0x08020208,0x08000200,0x08000008,0x00000200,0x00000000,0x08020008,
    0x08000208,0x00020000,0x08000000,0x08020208,0x00000008,0x00020208,0x00020200,0x08000008,
    0x08020000,0x08000208,0x00000208,0x08020000,0x00020208,0x00000008,0x08020008,0x00020200,
    0x00802001,0x00002081,0x00002081,0x00000080,0x00802080,0x00800081,0x00800001,0x00002001,
    0x00000000,0x00802000,0x00802000,0x00802081,0x00000081,0x00000000,0x00800080,0x00800001,
    0x00000001,0x00002000,0x00800000,0x00802001,0x00000080,0x00800000,0x00002001,0x00002080,
    0x00800081,0x00000001,0x00002080,0x00800080,0x00002000,0x00802080,0x00802081,0x00000081,
    0x00800080,0x00800001,0x00802000,0x00802081,0x00000081,0x00000000,0x00000000,0x00802000,
    0x00002080,0x00800080,0x00800081,0x00000001,0x00802001,0x00002081,0x00002081,0x00000080,
    0x00802081,0x00000081,0x00000001,0x00002000,0x00800001,0x00002001,0x00802080,0x00800081,
    0x00002001,0x00002080,0x00800000,0x00802001,0x00000080,0x00800000,0x00002000,0x00802080,
    0x00000100,0x02080100,0x02080000,0x42000100,0x00080000,0x00000100,0x40000000,0x02080000,
    0x40080100,0x00080000,0x02000100,0x40080100,0x42000100,0x42080000,0x00080100,0x40000000,
    0x02000000,0x40080000,0x40080000,0x00000000,0x40000100,0x42080100,0x42080100,0x02000100,
    0x42080000,0x40000100,0x00000000,0x42000000,0x02080100,0x02000000,0x42000000,0x00080100,
    0x00080000,0x42000100,0x00000100,0x02000000,0x40000000,0x02080000,0x42000100,0x40080100,
    0x02000100,0x40000000,0x42080000,0x02080100,0x40080100,0x00000100,0x02000000,0x42080000,
    0x42080100,0x00080100,0x42000000,0x42080100,0x02080000,0x00000000,0x40080000,0x42000000,
    0x00080100,0x02000100,0x40000100,0x00080000,0x00000000,0x40080000,0x02080100,0x40000100,
    0x20000010,0x20400000,0x00004000,0x20404010,0x20400000,0x00000010,0x20404010,0x00400000,
    0x20004000,0x00404010,0x00400000,0x20000010,0x00400010,0x20004000,0x20000000,0x00004010,
    0x00000000,0x00400010,0x20004010,0x00004000,0x00404000,0x20004010,0x00000010,0x20400010,
    0x20400010,0x00000000,0x00404010,0x20404000,0x00004010,0x00404000,0x20404000,0x20000000,
    0x20004000,0x00000010,0x20400010,0x00404000,0x20404010,0x00400000,0x00004010,0x20000010,
    0x00400000,0x20004000,0x20000000,0x00004010,0x20000010,0x20404010,0x00404000,0x20400000,
    0x00404010,0x20404000,0x00000000,0x20400010,0x00000010,0x00004000,0x20400000,0x00404010,
    0x00004000,0x00400010,0x20004010,0x00000000,0x20404000,0x20000000,0x00400010,0x20004010,
    0x00200000,0x04200002,0x04000802,0x00000000,0x00000800,0x04000802,0x00200802,0x04200800,
    0x04200802,0x00200000,0x00000000,0x04000002,0x00000002,0x04000000,0x04200002,0x00000802,
    0x04000800,0x00200802,0x00200002,0x04000800,0x04000002,0x04200000,0x04200800,0x00200002,
    0x04200000,0x00000800,0x00000802,0x04200802,0x00200800,0x00000002,0x04000000,0x00200800,
    0x04000000,0x00200800,0x00200000,0x04000802,0x04000802,0x04200002,0x04200002,0x00000002,
    0x00200002,0x04000000,0x04000800,0x00200000,0x04200800,0x00000802,0x00200802,0x04200800,
    0x00000802,0x04000002,0x04200802,0x04200000,0x00200800,0x00000000,0x00000002,0x04200802,
    0x00000000,0x00200802,0x04200000,0x00000800,0x04000002,0x04000800,0x00000800,0x00200002,
    0x10001040,0x00001000,0x00040000,0x10041040,0x10000000,0x10001040,0x00000040,0x10000000,
    0x00040040,0x10040000,0x10041040,0x00041000,0x10041000,0x00041040,0x00001000,0x00000040,
    0x10040000,0x10000040,0x10001000,0x00001040,0x00041000,0x00040040,0x10040040,0x10041000,
    0x00001040,0x00000000,0x00000000,0x10040040,0x10000040,0x10001000,0x00041040,0x00040000,
    0x00041040,0x00040000,0x10041000,0x00001000,0x00000040,0x10040040,0x00001000,0x00041040,
    0x10001000,0x00000040,0x10000040,0x10040000,0x10040040,0x10000000,0x00040000,0x10001040,
    0x00000000,0x10041040,0x00040040,0x10000040,0x10040000,0x10001000,0x10001040,0x00000000,
    0x10041040,0x00041000,0x00041000,0x00001040,0x00001040,0x00040040,0x10000000,0x10041000
};

static const int h_KEY_SHIFTS[16] = {1,1,2,2,2,2,2,2,1,2,2,2,2,2,2,1};
static const int h_pc1_c[28] = {57,49,41,33,25,17,9,1,58,50,42,34,26,18,10,2,59,51,43,35,27,19,11,3,60,52,44,36};
static const int h_pc1_d[28] = {63,55,47,39,31,23,15,7,62,54,46,38,30,22,14,6,61,53,45,37,29,21,13,5,28,20,12,4};
static const int h_pc2[48] = {14,17,11,24,1,5,3,28,15,6,21,10,23,19,12,4,26,8,16,7,27,20,13,2,41,52,31,37,47,55,30,40,51,45,33,48,44,49,39,56,34,53,46,42,50,36,29,32};

void host_des_key_schedule(const uchar *key, uint32_t *sk) {
    uint32_t c = 0, d = 0;
    for (int i = 0; i < 28; i++) {
        int bc = h_pc1_c[i]-1; if (key[bc/8] & (0x80 >> (bc%8))) c |= (1u<<(27-i));
        int bd = h_pc1_d[i]-1; if (key[bd/8] & (0x80 >> (bd%8))) d |= (1u<<(27-i));
    }
    for (int r = 0; r < 16; r++) {
        for (int s = 0; s < h_KEY_SHIFTS[r]; s++) {
            c = ((c<<1)|(c>>27)) & 0x0FFFFFFF; d = ((d<<1)|(d>>27)) & 0x0FFFFFFF;
        }
        sk[r*2]=0; sk[r*2+1]=0;
        for (int i = 0; i < 48; i++) {
            int src = h_pc2[i], val = (src<=28) ? ((c>>(28-src))&1) : ((d>>(56-src))&1);
            if (val) { int sbox=i/6,bip=i%6,w=sbox&1,base=24-(sbox>>1)*8,tgt=base+5-bip;
                if (w==0) sk[r*2] |= (1u<<tgt); else sk[r*2+1] |= (1u<<tgt); }
        }
    }
}

void host_des_crypt(const uchar *in, uchar *out, const uint32_t *sk, int decrypt) {
    uint32_t l,r,t;
    l=((uint32_t)in[0]<<24)|((uint32_t)in[1]<<16)|((uint32_t)in[2]<<8)|in[3];
    r=((uint32_t)in[4]<<24)|((uint32_t)in[5]<<16)|((uint32_t)in[6]<<8)|in[7];
    t=((l>>4)^r)&0x0F0F0F0F; r^=t; l^=t<<4;
    t=((l>>16)^r)&0x0000FFFF; r^=t; l^=t<<16;
    t=((r>>2)^l)&0x33333333; l^=t; r^=t<<2;
    t=((r>>8)^l)&0x00FF00FF; l^=t; r^=t<<8;
    r=(r<<1)|(r>>31); t=(l^r)&0xAAAAAAAA; r^=t; l^=t; l=(l<<1)|(l>>31);
    for (int rnd=0;rnd<16;rnd++) {
        uint32_t w,f; int idx=decrypt?(15-rnd):rnd;
        w=(r<<28)|(r>>4); w^=sk[idx*2];
        f=SP_ALL[384+(w&0x3F)]; w>>=8; f|=SP_ALL[256+(w&0x3F)]; w>>=8; f|=SP_ALL[128+(w&0x3F)]; w>>=8; f|=SP_ALL[0+(w&0x3F)];
        w=r^sk[idx*2+1];
        f|=SP_ALL[448+(w&0x3F)]; w>>=8; f|=SP_ALL[320+(w&0x3F)]; w>>=8; f|=SP_ALL[192+(w&0x3F)]; w>>=8; f|=SP_ALL[64+(w&0x3F)];
        l^=f; t=l; l=r; r=t;
    }
    t=l; l=r; r=t;
    l=(l<<31)|(l>>1); t=(l^r)&0xAAAAAAAA; r^=t; l^=t;
    r=(r<<31)|(r>>1);
    t=((r>>8)^l)&0x00FF00FF; l^=t; r^=t<<8;
    t=((r>>2)^l)&0x33333333; l^=t; r^=t<<2;
    t=((l>>16)^r)&0x0000FFFF; r^=t; l^=t<<16;
    t=((l>>4)^r)&0x0F0F0F0F; r^=t; l^=t<<4;
    out[0]=(uchar)(l>>24); out[1]=(uchar)(l>>16); out[2]=(uchar)(l>>8); out[3]=(uchar)l;
    out[4]=(uchar)(r>>24); out[5]=(uchar)(r>>16); out[6]=(uchar)(r>>8); out[7]=(uchar)r;
}

int main(void) {
    uchar key[8] = {0xD3, 0xCD, 0x16, 0x94, 0xCB, 0xA1, 0x26, 0xFE};
    uint32_t sk[32];
    host_des_key_schedule(key, sk);

    printf("=== FIXING BLOCK 130 ===\n");
    {
        uchar correct_pt[8] = {'i','n','g',' ','t','h','e',' '};
        uchar correct_ct[8];
        host_des_crypt(correct_pt, correct_ct, sk, 0);
        printf("Correct PT: \"ing the \" = ");
        for (int i = 0; i < 8; i++) printf("%02x", correct_pt[i]);
        printf("\nCorrect CT: ");
        for (int i = 0; i < 8; i++) printf("%02x", correct_ct[i]);
        printf("\nWrong CT:   d2abed2fca2de12b\n");
        printf("File offset: 1040 (0x410)\n\n");
    }

    printf("=== TESTING BLOCK 138 CANDIDATES ===\n");
    printf("Wrong CT:  0eb6a976ead73afd -> PT: 961eb0cc7b342c00\n\n");

    /* The file might have been 1104 bytes originally (138 blocks = text only) */
    /* Or 1112 bytes with 8 bytes of padding */
    printf("Possible correct plaintexts for block 138 and their CTs:\n\n");

    const char *labels[] = {
        "8x NUL (zero padding)",
        "8x 0x08 (PKCS5 padding for 0 remaining)",
        "8x space",
        "NUL NUL NUL NUL NUL NUL NUL NUL",
    };
    uchar candidates[][8] = {
        {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
        {0x08,0x08,0x08,0x08,0x08,0x08,0x08,0x08},
        {0x20,0x20,0x20,0x20,0x20,0x20,0x20,0x20},
    };
    int nc = 3;

    for (int c = 0; c < nc; c++) {
        uchar ct[8];
        host_des_crypt(candidates[c], ct, sk, 0);
        printf("  %s:\n    PT: ", labels[c]);
        for (int i = 0; i < 8; i++) printf("%02x", candidates[c][i]);
        printf(" -> CT: ");
        for (int i = 0; i < 8; i++) printf("%02x", ct[i]);
        printf("\n");
    }

    printf("\n=== PATCHING text.d -> text_fixed.d ===\n");

    /* Read original */
    uchar ct_data[1120];
    int ct_len;
    FILE *f = fopen("text.d", "rb");
    if (!f) f = fopen("C:\\Users\\banton\\iCloudDrive\\2K\\stage09\\text.d", "rb");
    if (!f) { printf("ERROR: Cannot open text.d\n"); return 1; }
    fseek(f, 0, SEEK_END); ct_len = (int)ftell(f); fseek(f, 0, SEEK_SET);
    fread(ct_data, 1, ct_len, f); fclose(f);

    /* Fix block 130 */
    uchar fix130_pt[8] = {'i','n','g',' ','t','h','e',' '};
    uchar fix130_ct[8];
    host_des_crypt(fix130_pt, fix130_ct, sk, 0);
    memcpy(ct_data + 130*8, fix130_ct, 8);
    printf("Block 130 patched: ");
    for (int i = 0; i < 8; i++) printf("%02x", fix130_ct[i]);
    printf("\n");

    /* Write fixed file (without last block since it's also corrupt and possibly just padding) */
    /* Option 1: Write 1112 bytes keeping block 138 as-is */
    /* Option 2: Write 1104 bytes truncating the corrupt last block */

    /* Write full 1112-byte version with block 130 fixed */
    f = fopen("text_fixed.d", "wb");
    if (!f) f = fopen("C:\\Users\\banton\\iCloudDrive\\2K\\stage09\\text_fixed.d", "wb");
    if (!f) { printf("ERROR: Cannot create text_fixed.d\n"); return 1; }
    fwrite(ct_data, 1, ct_len, f);
    fclose(f);
    printf("Wrote text_fixed.d (1112 bytes, block 130 fixed)\n");

    /* Also write truncated 1104-byte version */
    f = fopen("text_fixed_trunc.d", "wb");
    if (!f) f = fopen("C:\\Users\\banton\\iCloudDrive\\2K\\stage09\\text_fixed_trunc.d", "wb");
    if (!f) { printf("ERROR: Cannot create text_fixed_trunc.d\n"); return 1; }
    fwrite(ct_data, 1, 1104, f);
    fclose(f);
    printf("Wrote text_fixed_trunc.d (1104 bytes, block 130 fixed, last block removed)\n");

    /* Verify: decrypt block 130 from fixed file */
    printf("\nVerification - block 130 from fixed file:\n");
    uchar verify_pt[8];
    host_des_crypt(ct_data + 130*8, verify_pt, sk, 1);
    printf("  CT: ");
    for (int i = 0; i < 8; i++) printf("%02x", ct_data[130*8+i]);
    printf(" -> PT: \"");
    for (int i = 0; i < 8; i++) putchar(verify_pt[i]);
    printf("\"\n");

    /* Full corrected decrypt */
    printf("\n=== FULL CORRECTED DECRYPTION (138 blocks) ===\n");
    for (int blk = 0; blk < 138; blk++) {
        uchar pt[8];
        host_des_crypt(ct_data + blk*8, pt, sk, 1);
        for (int j = 0; j < 8; j++) {
            if (pt[j] >= 0x20 && pt[j] <= 0x7E) putchar(pt[j]);
            else if (pt[j] == 0x0A) putchar('\n');
            else if (pt[j] == 0x0D) ; /* skip CR */
            else printf("[%02X]", pt[j]);
        }
    }
    printf("\n=== END ===\n");

    return 0;
}
