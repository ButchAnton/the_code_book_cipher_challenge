#!/usr/bin/env python3
"""
fast_decode.py -- Focused HC solver for Stage 07, using the confirmed best
                  arrangement (1,2,3,0) and the corrected pair orientation.

ROLE IN THE PIPELINE
====================
After diag_stage07.py identified:
  - The 4 correct cipher-block pairs: (0,7), (1,2), (3,4), (5,6)
  - The best arrangement: (1,2,3,0)
  - The corrected row/col orientation (see CIPHER_PAIRS below)

this script runs 50 hill-climbing restarts × 500k iterations each on that
single arrangement to get the best possible Polybius mapping.

KEY IMPROVEMENTS OVER diag_stage07.py
======================================
1. Corrected orientation: see CIPHER_PAIRS comment below.
2. HC gaming fix: swaps are restricted to the 26 letter cells only.
   In diag_stage07.py, the HC could swap unmapped (-1) cells with letter
   cells, effectively shrinking the decoded sequence length and producing
   an artificially higher average score without improving text quality.
   Here we precompute letter_cells and only pick from those 26 cells,
   maintaining a strict 26-letter bijection at all times.
3. Multiple restarts with different seeds: each seed explores a different
   random walk.  Because HC is deterministic from a given start, different
   seeds escape different local optima.  50 restarts gives good coverage.

EXPECTED OUTPUT
===============
  Best score: approximately -4.97 (per quadgram, average log10-probability).
  Decoded text: recognisable German biography of Pierre de Fermat.
  Key words visible: FERMAT (as PRMATE before homophone fix), TOULOUSE,
  UNIVERSITAT, FRANZISKANERKLOSTER, ENIGMAMASCHINE.
  (Run homo_fix.py next to correct the remaining homophone errors.)
"""
import math, sys
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).parent
MISC_DIR   = SCRIPT_DIR.parent / "misc"
CT_PATH    = SCRIPT_DIR / "ciphertext_stage07.txt"
QG_PATH    = MISC_DIR / "german_quadgrams.txt"

# ---- Cipher constants ---------------------------------------------------------

# Maps each of the 8 cipher symbols to an integer index 0-7.
# Used to encode ciphertext tokens and to label Polybius rows/columns.
SYM2IDX = {'C':0,'E':1,'M':2,'O':3,'P':4,'R':5,'T':6,'U':7}

# N_SYM=8:    8 distinct ciphertext symbols → 8×8 = 64 possible bigrams.
# N_CELLS=64: total cells in the Polybius square (each bigram is one cell).
# KEY_LEN=8:  transposition key length (7048 = 8×881, 881 is prime).
# COL_LEN=881: number of rows in the transposition grid.
N_SYM = 8; N_CELLS = 64; KEY_LEN = 8; COL_LEN = 881

# CORRECTED CIPHER PAIRS — orientation fix.
#
# In the original ADFGVX fractionation:
#   - Each plaintext letter is encoded as (ROW_symbol, COL_symbol).
#   - The symbol stream is written into the transposition grid row-by-row,
#     so column 0 of the grid gets all the ROW symbols from odd positions,
#     column 1 all the COL symbols, etc.
#   - After columnar transposition with key length 8, the 8 output block-
#     columns are a permutation of the original 8 grid columns.
#
# The naive initial guess was that the lower-indexed block-column in each
# pair is always the ROW symbol.  Empirical testing showed this is WRONG
# for some pairs.  The corrected orientation is:
#   - (0, 7): block-col 0 = ROW,  block-col 7 = COL
#   - (2, 1): block-col 2 = ROW,  block-col 1 = COL  ← reversed vs. naive
#   - (4, 3): block-col 4 = ROW,  block-col 3 = COL  ← reversed
#   - (6, 5): block-col 6 = ROW,  block-col 5 = COL  ← reversed
# This orientation was confirmed by comparing all 384 combinations
# (24 arrangements × 16 orientations) and finding this set dominant
# at -4.97 vs. the best wrong orientation at -5.74.
CIPHER_PAIRS = [(0,7),(2,1),(4,3),(6,5)]

# German letter frequency order (most to least frequent, letters only).
# Used to initialise the Polybius mapping: most-frequent bigram → E, etc.
GERMAN_FREQ_ORDER = "ENISRATDHULCGMOBWFKZPVJYXQ"


# ==============================================================================
# Data loading
# ==============================================================================

def load_qg(path):
    """
    Load German quadgrams and return a (26,26,26,26) float32 log10-probability
    array.  Unseen quadgrams get a small floor value (log10(0.01/total)).
    See diag_stage07.py for detailed explanation of the scoring scheme.
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
    # Floor log-probability for quadgrams absent from the training corpus.
    floor   = math.log10(0.01) - log_tot
    arr = np.full((26,26,26,26), floor, dtype=np.float32)
    for qg, cnt in counts.items():
        idx = [ord(c)-65 for c in qg]
        if all(0 <= i < 26 for i in idx) and cnt > 0:
            arr[idx[0],idx[1],idx[2],idx[3]] = math.log10(cnt) - log_tot
    return arr


# ==============================================================================
# Ciphertext parsing
# ==============================================================================

def parse_ct(path):
    """
    Parse the tab-separated ciphertext file into a flat (7048,) int32 array.
    The display format (26 tokens per line) is purely cosmetic; all symbols
    are read in sequential order.
    """
    seq = []
    with open(path) as f:
        for line in f:
            for tok in line.strip().split('\t'):
                t = tok.strip()
                if t in SYM2IDX: seq.append(SYM2IDX[t])
    return np.array(seq, dtype=np.int32)


# ==============================================================================
# Bigram construction
# ==============================================================================

def make_bigrams(block_cols, pair_order):
    """
    Assemble the bigram sequence for a given arrangement of the 4 cipher pairs.

    pair_order is a 4-tuple permutation of [0,1,2,3], specifying which
    cipher-block pair occupies each of the 4 fractionated-text positions per
    row.  For the confirmed best arrangement, pair_order = (1, 2, 3, 0).

    How the interleaving works:
      Each of the 4 pairs contributes one bigram per plaintext character row.
      With 881 rows and 4 bigrams/row, the total sequence is 3524 bigrams.
      The assignment bigrams[pos::n_pairs] = bg interleaves 4 streams:
        pos=0 → positions  0,  4,  8, ... (bigram slot 0 of each row)
        pos=1 → positions  1,  5,  9, ... (bigram slot 1 of each row)
        pos=2 → positions  2,  6, 10, ... (bigram slot 2 of each row)
        pos=3 → positions  3,  7, 11, ... (bigram slot 3 of each row)
      This reconstructs the original left-to-right order of the 4 plaintext
      characters encoded per row.

    Returns (3524,) int32 array of cell indices in [0, 63].
    """
    n_pairs = 4
    bigrams = np.empty(COL_LEN * n_pairs, dtype=np.int32)
    for pos, pair_idx in enumerate(pair_order):
        ca, cb = CIPHER_PAIRS[pair_idx]
        # Encode (ROW symbol, COL symbol) as a single cell index 0-63.
        bg = block_cols[ca] * N_SYM + block_cols[cb]
        # Interleave into the output array at stride n_pairs.
        bigrams[pos::n_pairs] = bg
    return bigrams


# ==============================================================================
# Quadgram scoring
# ==============================================================================

def score_qg(letters, qg_arr):
    """
    Average German quadgram log10-probability per quadgram.

    Using the average (not the sum) prevents the optimiser from gaming the
    score by unmapping cells — shrinking the letter sequence would reduce the
    denominator and inflate the average if the sum were used instead.

    letters: int32 array of letter indices (0-25), all >= 0 (no spaces).
    Returns a float; higher (less negative) = better German text.
    """
    if letters.size < 4: return -1e18
    n = letters.size - 3
    return float(qg_arr[letters[:-3],letters[1:-2],letters[2:-1],letters[3:]].sum()) / n


# ==============================================================================
# Hill-climbing Polybius optimisation (gaming-proof version)
# ==============================================================================

def hc_once(bigrams, qg_arr, n_iter=500_000, seed=0):
    """
    One hill-climbing restart over the Polybius mapping.

    GAMING-PROOF DESIGN
    -------------------
    Unlike diag_stage07.py's hc_once, this version restricts swaps to the
    26 letter cells (letter_cells array).  This means:
      - The space cell is always excluded (it's not in letter_cells).
      - Unmapped (-1) cells are never swapped with letter cells.
      - The 26 letter assignments always form a complete bijection A-Z.
    Without this restriction, the HC can falsely improve its score by
    collapsing letters to -1, reducing the decoded sequence length and
    making the average per-quadgram score look better.

    INITIALISATION
    --------------
    1. Sort bigrams by descending frequency.
    2. The most-frequent bigram → space (poly = -1, never swapped).
    3. The next 26 bigrams → letters E, N, I, S, R, A, T, D, H, U, L, C, G,
       M, O, B, W, F, K, Z, P, V, J, Y, X, Q  (German frequency order).
    These 26 cell indices are stored in letter_cells and are the only
    candidates for swapping throughout the search.

    GLOBAL BEST TRACKING
    --------------------
    The current poly can change on neutral moves (equal score), so we
    separately track best_poly (the highest score ever seen) alongside the
    working poly (which may have drifted).

    Parameters
    ----------
    bigrams : (3524,) int32 — bigram cell indices for the chosen arrangement.
    qg_arr  : (26,26,26,26) float32 — German quadgram log-probabilities.
    n_iter  : number of swap attempts.
    seed    : RNG seed; different seeds produce different random walks,
              helping escape different local optima.

    Returns (best_poly, best_score).
    """
    rng = np.random.default_rng(seed)

    # ---- Frequency-rank initialisation ----
    cnts = np.bincount(bigrams.astype(np.int64), minlength=N_CELLS)
    order = np.argsort(-cnts)  # bigram indices sorted most-to-least frequent

    # Build initial poly: space → -1 (rank 0), letters → ranks 1-26.
    poly = np.full(N_CELLS, -1, dtype=np.int32)
    space_bi = int(order[0])   # most frequent bigram = space separator
    letter_cells = []
    for i in range(26):
        if i+1 < len(order) and cnts[order[i+1]] > 0:
            poly[order[i+1]] = ord(GERMAN_FREQ_ORDER[i]) - ord('A')
            letter_cells.append(int(order[i+1]))
    # Convert to numpy array for fast random index selection.
    letter_cells = np.array(letter_cells, dtype=np.int32)

    # Initial score.
    lets = poly[bigrams]; lets = lets[lets >= 0]
    score = score_qg(lets, qg_arr)
    best_poly = poly.copy(); best_score = score

    for _ in range(n_iter):
        # Pick two distinct letter cells to swap (both drawn from letter_cells).
        # This guarantees we never touch the space cell or any unmapped cell.
        ai = int(rng.integers(len(letter_cells)))
        bi = int(rng.integers(len(letter_cells)))
        if ai == bi: continue
        a, b = letter_cells[ai], letter_cells[bi]

        # Trial swap.
        poly[a], poly[b] = poly[b], poly[a]
        lets = poly[bigrams]; lets = lets[lets >= 0]
        ns = score_qg(lets, qg_arr)

        if ns >= score:
            # Accept improvement or neutral move.
            score = ns
            if score > best_score:
                best_poly = poly.copy(); best_score = score
        else:
            # Reject: undo swap.
            poly[a], poly[b] = poly[b], poly[a]

    return best_poly, best_score


# ==============================================================================
# Text decoding helper
# ==============================================================================

def decode_text(poly, bigrams, sep_as_space=True):
    """
    Decode bigrams using the given poly mapping and return a plain string.

    poly[cell] = -1  → space (if sep_as_space=True) or skip.
    poly[cell] ∈ 0-25 → letter A-Z.

    sep_as_space=True  : unmapped cells appear as ' ' in the output string.
    sep_as_space=False : unmapped cells are skipped (raw letter sequence only).
    """
    lets = poly[bigrams]
    chars = []
    for li in lets:
        li = int(li)
        if li < 0:
            # This bigram is the space separator (or an unmapped homophone).
            if sep_as_space: chars.append(' ')
        else:
            chars.append(chr(65 + li))
    return ''.join(chars)


# ==============================================================================
# Main: load data, run 50 HC restarts, display best decoding
# ==============================================================================

print("Loading QG...", flush=True)
qg_arr = load_qg(QG_PATH)

# Parse ciphertext and split into 8 block-columns of 881 symbols each.
flat = parse_ct(CT_PATH)
block_cols = [flat[j*COL_LEN:(j+1)*COL_LEN] for j in range(KEY_LEN)]

# Assemble bigrams for the confirmed best arrangement (1,2,3,0).
# This arrangement was identified by diag_stage07.py and exhaustive testing
# of all 384 combinations (24 arrangements × 16 orientations).
BEST_ARR = (1, 2, 3, 0)
bigrams = make_bigrams(block_cols, BEST_ARR)
print(f"Bigrams: {len(bigrams)}, unique: {np.unique(bigrams).size}", flush=True)

# ---- Multi-restart hill-climbing ----
# 50 restarts × 500k iterations each.  Each restart uses a different RNG seed,
# which changes the sequence of proposed swaps and therefore explores a
# different region of the 26! permutation space.  The global best across all
# restarts is kept.  In practice the best score is found within the first few
# restarts; the remaining restarts confirm there is no significantly better
# local optimum.
N_RESTARTS = 50
best_score = -1e18
best_poly = None

print(f"\nRunning {N_RESTARTS} HC restarts x 500k...", flush=True)
for restart in range(N_RESTARTS):
    poly, score = hc_once(bigrams, qg_arr, n_iter=500_000, seed=restart)
    if score > best_score:
        best_score = score
        best_poly = poly.copy()
        # Show a 60-char sample of the raw (no-space) decoded text so we can
        # visually track whether quality is improving across restarts.
        sample = decode_text(best_poly, bigrams[:200], sep_as_space=False)
        print(f"  restart {restart:2d}: score={score:.4f}  {sample[:60]}", flush=True)
    elif restart % 10 == 0:
        print(f"  restart {restart:2d}: score={score:.4f} (no improvement)", flush=True)

# ---- Display results ----
print(f"\nBest score: {best_score:.4f}")

# Full decoded text with spaces shown (separator bigram replaced by ' ').
print("\n=== Full decoded text (separator as space) ===")
text = decode_text(best_poly, bigrams, sep_as_space=True)
for i in range(0, len(text), 80):
    print(f"  {text[i:i+80]}")

# Raw letter sequence (no space substitution) — useful for inspecting the
# uninterrupted bigram-to-letter mapping quality.
print("\n=== Raw (no separator conversion) ===")
raw = decode_text(best_poly, bigrams, sep_as_space=False)
for i in range(0, min(1200, len(raw)), 80):
    print(f"  {raw[i:i+80]}")
