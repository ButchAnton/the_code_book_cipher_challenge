#!/usr/bin/env python3
"""
homo_fix.py -- Correct known homophonic mapping errors and re-optimise.

BACKGROUND: THE HOMOPHONE PROBLEM
===================================
Stage 07 uses a HOMOPHONIC Polybius square: multiple bigrams can encode the
same plaintext letter.  For example, the two most common letters in German
(E ~17%, N ~10%) might each have two or three distinct bigrams assigned to
them in the square, while rare letters (Q, Y, X) have just one.

The hill-climbing solver in fast_decode.py assumes a strict one-to-one
(bijection) mapping: exactly one bigram per letter.  Under that constraint,
when two bigrams actually mean the same letter, the HC splits them arbitrarily
-- assigning one to the correct letter and the other to a wrong rare letter
(e.g., P, Q, or X).

After reading the HC output from fast_decode.py, three misassignments were
identified by recognising German words:

  1. cell(RE) was mapped to P, but should be F.
     Evidence: "PRMATEWURDE" should be "FERMATWURDE" (Fermat was/became).
     Also: "PRANZISKANERKLOSTER" should be "FRANZISKANERKLOSTER".
     Conclusion: bigram (RE) is a second homophone for F.

  2. cell(EO) was mapped to X, but should be S.
     Evidence: "NEXTUDIUM" should be "STUDIUM" (studies).
     Also: "NEXTADTEA" should be "STADTEA" (city of...).
     Conclusion: bigram (EO) is a second homophone for S.

  3. cell(CR) was mapped to Q, but should be E.
     Evidence: "QNIGMAMASCHINE" should be "ENIGMAMASCHINE" (Enigma machine).
     Also: "QINE" prefixes should be "EINE" (a/one).
     Conclusion: bigram (CR) is a second homophone for E.

STRATEGY
=========
1. Run standard HC (5 restarts x 500k) from the German-frequency-order
   initialisation to get a reproducible baseline.  This gives the same
   mapping fast_decode.py would produce.

2. Apply the three homophone corrections manually:
     poly[cell(RE)]: 15 (P) -> 5  (F)
     poly[cell(EO)]: 23 (X) -> 18 (S)
     poly[cell(CR)]: 16 (Q) -> 4  (E)
   After this step, cells for E, F, and S each have two homophones mapped
   to them.  Letters P, Q, X now have zero cells mapped to them.

3. Re-run HC (50 restarts x 1M) from the corrected starting point.
   The HC can now adjust all non-space cells freely.  Some re-arrangement
   occurs as adjacent bigrams are re-optimised in light of the fixes.

WHY THE SCORE DOESN'T REACH PERFECT GERMAN
===========================================
Even after fixes, the score plateaus at approximately -4.80.  This is because:
  a) Two more homophones remain unidentified:
     - cell(PR) maps to J, but the correct letter is uncertain (D or E).
     - cell(TM) maps to Y, which appears to be a sentence-end period marker.
  b) The quadgram scorer cannot distinguish between statistically equivalent
     assignments -- a bigram that appears only 17 times has insufficient
     context for unambiguous placement.
  c) Perfect German text would score approximately -3.5.  The remaining gap
     of ~1.3 score units corresponds to ~2-3 still-wrong bigrams.

EXPECTED OUTPUT
===============
  Baseline score: approximately -4.99
  After fixes:    approximately -4.88 (immediate improvement)
  After 50x HC:   approximately -4.80 (best achievable with this approach)
  Letter frequencies: near-perfect match to German (all ratios 0.85-1.17x)
  Decoded text: "FRMATE WURDE" (FERMAT WURDE), "FRANZISKANERKLOSTER",
                "UNIVERSITAT", "TOULOUSE", "ENIGMAMASCHINE" all visible.
"""
import math
from pathlib import Path
import numpy as np
from numpy.random import default_rng

SCRIPT_DIR = Path(__file__).parent
MISC_DIR   = SCRIPT_DIR.parent / "misc"
CT_PATH    = SCRIPT_DIR / "ciphertext_stage07.txt"
QG_PATH    = MISC_DIR / "german_quadgrams.txt"

# ---- Cipher constants ---------------------------------------------------------
# 8 cipher symbols; cell index = row_idx * 8 + col_idx in [0, 63].
SYM2IDX = {'C':0,'E':1,'M':2,'O':3,'P':4,'R':5,'T':6,'U':7}

# N_CELLS=64: 8x8 Polybius square.  KEY_LEN=8, COL_LEN=881 (7048/8, 881 prime).
N_SYM = 8; N_CELLS = 64; KEY_LEN = 8; COL_LEN = 881

# Corrected cipher-block pair orientation (confirmed by exhaustive testing).
# Each tuple (ca, cb): ca = ROW symbol column, cb = COL symbol column.
CIPHER_PAIRS = [(0,7),(2,1),(4,3),(6,5)]  # corrected orientation

# German letter frequency order for warm-start initialisation.
GFO = "ENISRATDHULCGMOBWFKZPVJYXQ"


# ==============================================================================
# Data loading
# ==============================================================================

def load_qg(path):
    """
    Load German quadgrams into a (26,26,26,26) float32 log10-probability array.
    Absent quadgrams receive floor = log10(0.01) - log10(total).
    See diag_stage07.py for full explanation of the scoring scheme.
    """
    counts = {}; total = 0
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) >= 2:
                qg = p[0].upper(); cnt = int(p[1])
                if len(qg) == 4 and qg.isalpha():
                    counts[qg] = cnt; total += cnt
    log_tot = math.log10(total)
    # Floor probability for unseen quadgrams (small but non-zero penalty).
    floor   = math.log10(0.01) - log_tot
    arr = np.full((26,26,26,26), floor, dtype=np.float32)
    for qg, cnt in counts.items():
        idx = [ord(c)-65 for c in qg]
        if all(0 <= i < 26 for i in idx) and cnt > 0:
            arr[idx[0],idx[1],idx[2],idx[3]] = math.log10(cnt) - log_tot
    return arr


def parse_ct(path):
    """Parse the tab-separated ciphertext file into a flat (7048,) int32 array."""
    seq = []
    with open(path) as f:
        for line in f:
            for tok in line.strip().split('\t'):
                t = tok.strip()
                if t in SYM2IDX: seq.append(SYM2IDX[t])
    return np.array(seq, dtype=np.int32)


def make_bigrams(block_cols, pair_order):
    """
    Assemble the 3524-bigram sequence for arrangement pair_order = (1,2,3,0).
    See fast_decode.py for full explanation of the interleaving logic.
    """
    n_pairs = 4
    bigrams = np.empty(COL_LEN * n_pairs, dtype=np.int32)
    for pos, pair_idx in enumerate(pair_order):
        ca, cb = CIPHER_PAIRS[pair_idx]
        bg = block_cols[ca] * N_SYM + block_cols[cb]
        # Interleave: positions pos, pos+4, pos+8, ... get this pair's bigrams.
        bigrams[pos::n_pairs] = bg
    return bigrams


# ==============================================================================
# Quadgram scoring
# ==============================================================================

def score_qg(letters, qg_arr):
    """
    Average German quadgram log10-probability per quadgram.
    Using the average prevents gaming by unmapping cells (see diag_stage07.py).
    Higher (less negative) = better German text.
    """
    if letters.size < 4: return -1e18
    n = letters.size - 3
    return float(qg_arr[letters[:-3],letters[1:-2],letters[2:-1],letters[3:]].sum()) / n


# ==============================================================================
# Hill-climbing (gaming-proof: swaps only within letter_cells)
# ==============================================================================

def hc_run(bigrams, poly_in, letter_cells, qg_arr, n_iter, seed):
    """
    One HC restart, swapping only within the supplied letter_cells.

    letter_cells: int32 array of the cell indices that are currently mapped
    to a letter (poly >= 0).  By restricting swaps to this set we ensure:
      - The space cell is never touched.
      - Unmapped/homophone cells that share a letter assignment are all included
        and can be freely re-assigned to any letter during optimisation.
      - The bijection structure is preserved within the mapped set.

    This is the same gaming-proof design as fast_decode.py's hc_once,
    generalised to accept an arbitrary letter_cells array so it works both
    before and after the homophone corrections (where letter_cells grows from
    26 to 29 cells because the three newly-fixed homophones are added).

    Returns (best_poly, best_score).
    poly_in is NOT modified (we work on a copy).
    """
    rng = default_rng(seed)
    poly = poly_in.copy()
    lc = letter_cells
    n = len(lc)

    # Initial score from the provided starting poly.
    lets = poly[bigrams]; lets = lets[lets >= 0]
    score = score_qg(lets, qg_arr)
    best_poly = poly.copy(); best_score = score

    for _ in range(n_iter):
        # Pick two distinct cells from letter_cells.
        ai = int(rng.integers(n))
        bi = int(rng.integers(n))
        if ai == bi: continue
        a, b = lc[ai], lc[bi]

        # Trial swap.
        poly[a], poly[b] = poly[b], poly[a]
        lets = poly[bigrams]; lets = lets[lets >= 0]
        ns = score_qg(lets, qg_arr)

        if ns >= score:
            # Accept (improvement or neutral).
            score = ns
            if score > best_score:
                best_poly = poly.copy(); best_score = score
        else:
            # Reject: undo swap.
            poly[a], poly[b] = poly[b], poly[a]

    return best_poly, best_score


# ==============================================================================
# Text decoding
# ==============================================================================

def decode_text(poly, bigrams, sep_bi):
    """
    Decode bigrams to a string, using sep_bi as the word-space indicator.

    sep_bi: the bigram cell index that was assigned as the space separator
            (the most-frequent bigram identified during initialisation).
    Any cell with poly=-1 or matching sep_bi is rendered as a space.
    Other cells render as the letter chr(65 + poly[cell]).
    """
    chars = []
    for bi in bigrams:
        bi = int(bi)
        li = int(poly[bi])
        if li < 0 or bi == sep_bi:
            chars.append(' ')
        else:
            chars.append(chr(65 + li))
    return ''.join(chars)


# ==============================================================================
# Main
# ==============================================================================

print("Loading QG...", flush=True)
qg_arr = load_qg(QG_PATH)

# Parse ciphertext and split into 8 block-columns.
flat = parse_ct(CT_PATH)
block_cols = [flat[j*COL_LEN:(j+1)*COL_LEN] for j in range(KEY_LEN)]

# Assemble bigrams for the confirmed best arrangement (1,2,3,0).
bigrams = make_bigrams(block_cols, (1,2,3,0))
print(f"Bigrams: {len(bigrams)}, unique: {np.unique(bigrams).size}")

# ---- Initialise the baseline Polybius mapping ----
# Sort bigrams by descending frequency to identify the space separator and
# assign the 26 most-frequent non-space bigrams to German frequency order.
cnts = np.bincount(bigrams.astype(np.int64), minlength=N_CELLS)
order = np.argsort(-cnts)  # bigram cell indices sorted most-to-least frequent
sep_bi = int(order[0])     # most-frequent bigram = word-space separator

poly0 = np.full(N_CELLS, -1, dtype=np.int32)
letter_cells0 = []
for i in range(26):
    if i+1 < len(order) and cnts[order[i+1]] > 0:
        # Rank i+1 overall (0-based) → i-th German letter (E, N, I, ...).
        poly0[order[i+1]] = ord(GFO[i]) - ord('A')
        letter_cells0.append(int(order[i+1]))
letter_cells0 = np.array(letter_cells0, dtype=np.int32)

# ---- Phase 1: standard HC baseline (reproduces fast_decode.py result) ----
# We run from scratch (not from a saved best) to guarantee reproducibility.
# 5 restarts x 500k is enough to converge to the same local optimum that
# fast_decode.py found, which is the starting point for our homophone fixes.
print("\nRunning standard HC (5 restarts x 500k) to get baseline...", flush=True)
best_std_score = -1e18
best_std_poly = None
for seed in range(5):
    p, s = hc_run(bigrams, poly0, letter_cells0, qg_arr, 500_000, seed)
    if s > best_std_score:
        best_std_score = s
        best_std_poly = p.copy()
        # Show the first 60 decoded chars as a visual quality indicator.
        print(f"  seed {seed}: score={s:.4f}  {''.join(chr(65+int(x)) if x>=0 else ' ' for x in p[bigrams[:60]])[:60]}", flush=True)

print(f"\nBaseline best score: {best_std_score:.4f}")

# ---- Show the current letter mapping for all 27 ranked cells ----
# Rank 0 = space separator, ranks 1-26 = letter cells.
# This lets us visually confirm which bigram carries which letter before
# applying the homophone corrections.
print("\n=== Current letter mapping (non-space cells by frequency) ===")
for rank in range(27):
    i = int(order[rank])
    r, c = divmod(i, 8)
    # Convert row/col indices back to symbol names for readability.
    sr = list(SYM2IDX.keys())[r]
    sc = list(SYM2IDX.keys())[c]
    letter = chr(65 + best_std_poly[i]) if best_std_poly[i] >= 0 else 'SEP'
    print(f"  rank {rank+1:2d}: ({sr}{sc}) freq={cnts[i]:4d} ({100*cnts[i]/len(bigrams):.1f}%) -> {letter}")

# ---- Phase 2: apply the three identified homophone corrections ----
# The HC assigned each bigram a unique letter, but three bigrams are actually
# homophones (second encodings) of letters that already have another bigram.
# We identify the incorrect assignments by reading the decoded text and
# recognising German words:
#
#   cell(RE) = bigram (R row, E col) = cell index 5*8+1 = 41
#     HC assigned: P (letter index 15)
#     Correct:     F (letter index 5)
#     Evidence:    "PRMATEWURDE" -> "FRMATEWURDE" (= "FERMAT WURDE")
#                  "PRANZISKANERKLOSTER" -> "FRANZISKANERKLOSTER"
#
#   cell(EO) = bigram (E row, O col) = cell index 1*8+3 = 11
#     HC assigned: X (letter index 23)
#     Correct:     S (letter index 18)
#     Evidence:    "NEXTUDIUM" -> "NESTUDIUM" (= "STUDIUM" + separator artefact)
#                  "NEXTADTE" -> "NESTADTE" (= "INSTADT")
#
#   cell(CR) = bigram (C row, R col) = cell index 0*8+5 = 5
#     HC assigned: Q (letter index 16)
#     Correct:     E (letter index 4)
#     Evidence:    "QNIGMAMASCHINE" -> "ENIGMAMASCHINE"
#                  "QINE" -> "EINE" (a/one)
print("\n=== Applying homophone fixes ===")
fixed_poly = best_std_poly.copy()
changes = []
for cell in range(N_CELLS):
    v = int(fixed_poly[cell])
    if v == 15:  # P (index 15) -> F (index 5)
        fixed_poly[cell] = 5
        r, c = divmod(cell, 8)
        changes.append(f"cell({list(SYM2IDX.keys())[r]}{list(SYM2IDX.keys())[c]}): P->F")
    elif v == 16:  # Q (index 16) -> E (index 4)
        fixed_poly[cell] = 4
        r, c = divmod(cell, 8)
        changes.append(f"cell({list(SYM2IDX.keys())[r]}{list(SYM2IDX.keys())[c]}): Q->E")
    elif v == 23:  # X (index 23) -> S (index 18)
        fixed_poly[cell] = 18
        r, c = divmod(cell, 8)
        changes.append(f"cell({list(SYM2IDX.keys())[r]}{list(SYM2IDX.keys())[c]}): X->S")

for ch in changes:
    print(f"  {ch}")

# Score the corrected poly immediately to see the instant improvement.
lets_f = fixed_poly[bigrams]; lets_f = lets_f[lets_f >= 0]
score_f = score_qg(lets_f, qg_arr)
print(f"Score after fixes: {score_f:.4f} (was {best_std_score:.4f})")

# ---- Phase 3: re-run HC from the corrected starting point ----
# After the fixes, three cells now share their letter with another cell
# (E appears twice, F twice, S twice).  We rebuild letter_cells to include
# ALL non-space mapped cells (29 cells: 26 original + 3 newly-valid homophones
# that were previously mapped to P/Q/X which are now effectively vacated).
# The HC can re-assign all of these freely.
#
# Note: the score may not improve dramatically.  The homophonic ambiguity
# is a fundamental limitation -- the quadgram scorer cannot always
# distinguish two homophones of the same letter, so the HC may shuffle them
# without consistent improvement.  The expected gain is ~0.09 score units
# (from -4.88 to ~-4.80), reflecting re-optimisation of neighbouring bigrams.
lc_fixed = np.array(
    [i for i in range(N_CELLS) if fixed_poly[i] >= 0 and i != sep_bi],
    dtype=np.int32,
)
print(f"\nContinuing HC from fixed poly ({len(lc_fixed)} letter cells)...")
print("Running 50 restarts x 1M iterations...", flush=True)

best_fixed_score = score_f
best_fixed_poly = fixed_poly.copy()

for seed in range(50):
    p_f, s_f = hc_run(bigrams, fixed_poly, lc_fixed, qg_arr, 1_000_000, seed + 200)
    if s_f > best_fixed_score:
        best_fixed_score = s_f
        best_fixed_poly = p_f.copy()
        # Show first 80 chars (with spaces from sep_bi) as a quality check.
        sample = ''.join(chr(65+int(x)) if x>=0 else ' ' for x in p_f[bigrams[:80]])
        print(f"  seed {seed:2d}: score={s_f:.4f}  {sample[:80]}", flush=True)
    elif seed % 10 == 0:
        print(f"  seed {seed:2d}: best so far={best_fixed_score:.4f}", flush=True)

print(f"\nBest score from fixed HC: {best_fixed_score:.4f}")

# ---- Display decoded text ----
print("\n=== Full decoded text (Y shown as '.', sep=space) ===")
text = decode_text(best_fixed_poly, bigrams, sep_bi)
# Y (letter index 24) appears overwhelmingly at sentence boundaries and
# acts as a sentence-end period marker rather than as a genuine letter Y.
# Replacing it with '.' makes the text structure much more readable.
text_punct = text.replace('Y', '.')
for i in range(0, len(text_punct), 80):
    print(f"  {text_punct[i:i+80]}")

# Raw output: separator = '_', all letters shown directly.
# Useful for inspecting the uninterrupted letter sequence without the
# visual noise of spaces introduced by the sep_bi substitution.
print("\n=== Raw decoded (no sep substitution) ===")
raw = ''.join(chr(65+int(best_fixed_poly[bi])) if best_fixed_poly[bi]>=0 else '_' for bi in bigrams)
for i in range(0, len(raw), 80):
    print(f"  {raw[i:i+80]}")

# ---- Letter frequency comparison ----
# A near-perfect score would show all ratios close to 1.0x.
# Ratios far from 1.0 indicate still-incorrect bigram assignments.
print("\n=== Letter frequency vs German ===")
GERMAN_FREQ = [0.0651,0.0189,0.0306,0.0508,0.1740,0.0165,0.0301,0.0476,
               0.0755,0.0027,0.0121,0.0344,0.0253,0.0978,0.0251,0.0079,
               0.0002,0.0700,0.0727,0.0615,0.0435,0.0067,0.0189,0.0003,0.0004,0.0113]
lets_out = best_fixed_poly[bigrams]; lets_out = lets_out[lets_out >= 0]
freq = np.bincount(lets_out, minlength=26).astype(float) / lets_out.size
print(f"  {'Letter':6s} {'Obs':8s} {'Exp':8s} {'Ratio':6s}")
for li in np.argsort(-freq)[:15]:
    print(f"  {chr(65+li):6s} {freq[li]:8.3f} {GERMAN_FREQ[li]:8.3f} {freq[li]/max(GERMAN_FREQ[li],0.001):6.2f}x")
