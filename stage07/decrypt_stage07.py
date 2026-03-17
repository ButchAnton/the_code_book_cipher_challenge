#!/usr/bin/env python3
"""
decrypt_stage07.py -- Stage 07: 8-symbol ADFGVX-variant cipher solver.

CIPHER STRUCTURE
================
This cipher is an ADFGVX variant that uses 8 symbols (C, E, M, O, P, R, T, U)
instead of the standard six.  Encryption steps:

  1. SUBSTITUTION -- each plaintext character is replaced by a 2-symbol pair
     using an 8x8 Polybius square (64 cells).  With 8 symbols the bigrams
     span {C,E,M,O,P,R,T,U}^2 giving 64 possible pairs -- enough to encode
     the full German alphabet and additional characters (spaces, digits, etc.).

  2. TRANSPOSITION -- the resulting fractionated symbol stream is written
     row-by-row into an 8-column grid (key length 8).  The columns are then
     permuted by a keyword and concatenated to form the ciphertext.

The ciphertext file shows the transposed data in a 26-char-per-row display
format (just for readability); the actual ciphertext is the flat symbol
sequence, which splits cleanly into 8 equal blocks of 881 symbols each.

ATTACK STRATEGY
===============
Key insight: 7048 = 8 x 881 (exact), confirming key length = 8.

Step 1 -- PAIR DISCOVERY
  For each pair of 8 ciphertext blocks (c0..c7), compute the pairwise bigram
  index-of-coincidence (IOC).  Correct Polybius pairs give IOC ~= 0.074 (German)
  while random pairs give IOC ~= 0.016.
  Result: pairs {0,7}, {1,2}, {3,4}, {5,6} all have IOC ~= 0.067-0.070.

Step 2 -- ARRANGEMENT SEARCH (4! = 24 orderings)
  We must assign each of the 4 cipher-block pairs to one of the 4 fractionated-
  text row positions.  We try all 24 orderings and identify the best.

Step 3 -- POLYBIUS MAPPING RECOVERY
  For a fixed arrangement, the 3524 bigrams form a substitution cipher.
  Strategy: identify the 26 most-frequent non-space bigrams as the 26 letter
  bigrams, then run Simulated Annealing (SA) to find the best letter assignment.

  WHY SA OVER HC: Hill-climbing gets stuck in local optima at ~-4.31 per
  quadgram.  SA with temperature T0=0.5 -> Tmin=0.001 accepts temporarily-worse
  moves, allowing escape from these optima and convergence toward -3.0 to -3.5
  (correct German text).

Step 4 -- SEPARATOR IDENTIFICATION
  The most frequency-overrepresented decoded letter (relative to expected German
  frequency) likely corresponds to the space character in the original plaintext.

USAGE
=====
  py -3 decrypt_stage07.py

OUTPUT
======
  decrypted_stage07.txt -- full plaintext + diagnostic metadata
"""

from __future__ import annotations

import math
import sys
import time
from itertools import permutations
from pathlib import Path

import numpy as np

# ---- File paths ---------------------------------------------------------------
SCRIPT_DIR      = Path(__file__).parent
MISC_DIR        = SCRIPT_DIR.parent / "misc"
CIPHERTEXT_PATH = SCRIPT_DIR / "ciphertext_stage07.txt"
QUADGRAMS_PATH  = MISC_DIR   / "german_quadgrams.txt"

# ---- Cipher constants ---------------------------------------------------------
SYM2IDX   = {'C': 0, 'E': 1, 'M': 2, 'O': 3, 'P': 4, 'R': 5, 'T': 6, 'U': 7}
SYM_NAMES = ['C', 'E', 'M', 'O', 'P', 'R', 'T', 'U']
N_SYM     = 8                # number of distinct ciphertext symbols
N_CELLS   = N_SYM * N_SYM   # = 64 Polybius square cells
KEY_LEN   = 8                # transposition key length (7048 = 8 x 881)
COL_LEN   = 7048 // KEY_LEN  # = 881 chars per block column

# Identified cipher-block pairs: each pair has IOC ~= 0.067-0.070 (German ~= 0.074)
# These 4 pairs partition {0,...,7} perfectly.
CIPHER_PAIRS = [(0, 7), (1, 2), (3, 4), (5, 6)]

# German letter frequency order (most to least frequent)
GERMAN_FREQ_ORDER = "ENISRATDHULCGMOBWFKZPVJYXQ"

# German letter frequencies indexed A=0 .. Z=25 (letters-only basis)
GERMAN_FREQ = np.array([
    0.0651,  # A
    0.0189,  # B
    0.0306,  # C
    0.0508,  # D
    0.1740,  # E
    0.0165,  # F
    0.0301,  # G
    0.0476,  # H
    0.0755,  # I
    0.0027,  # J
    0.0121,  # K
    0.0344,  # L
    0.0253,  # M
    0.0978,  # N
    0.0251,  # O
    0.0079,  # P
    0.0002,  # Q
    0.0700,  # R
    0.0727,  # S
    0.0615,  # T
    0.0435,  # U
    0.0067,  # V
    0.0189,  # W
    0.0003,  # X
    0.0004,  # Y
    0.0113,  # Z
], dtype=np.float64)

# ---- Search parameters --------------------------------------------------------
# Phase 1: quick HC on all 24 arrangements to identify candidates
N_HC_RESTARTS_P1 = 2
N_HC_ITER_P1     = 300_000

# Phase 2: SA on top N_TOP_P2 arrangements for thorough optimization
N_SA_RESTARTS_P2 = 5
N_SA_ITER_P2     = 2_000_000
SA_T0            = 0.5      # initial temperature (accept ~0.5/qg worse moves at ~37%)
SA_TMIN          = 0.001    # final temperature (effectively pure HC)
N_TOP_P2         = 8        # number of top arrangements to refine


# ==============================================================================
# Data loading
# ==============================================================================

def load_quadgrams(path: Path) -> tuple[np.ndarray, float]:
    """
    Load German quadgrams from '../misc/german_quadgrams.txt'.
    Format: '<QUADGRAM> <COUNT>' per line.

    Returns
    -------
    qg_arr : float32 ndarray, shape (26, 26, 26, 26)
        log10-probability for every 4-letter German sequence.
    floor  : float
        log10-probability for quadgrams absent from the file.
    """
    counts: dict[str, int] = {}
    total = 0
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 2:
                qg  = parts[0].upper()
                cnt = int(parts[1])
                if len(qg) == 4 and qg.isalpha():
                    counts[qg] = cnt
                    total += cnt

    log_total = math.log10(total)

    # floor: log10-probability assigned to any quadgram not in the corpus.
    # Represents "this quadgram occurs at rate 0.01 out of 'total' total
    # observations" -- a small but non-zero penalty that avoids -infinity
    # scores while still strongly discouraging unlikely sequences.
    floor     = math.log10(0.01) - log_total

    # 4-D array: qg_arr[a][b][c][d] = log10 P(abcd) for letters a,b,c,d in A-Z.
    # Indexing with four numpy arrays simultaneously gives vectorised O(1) lookups
    # over an entire sequence -- much faster than a Python dict per quadgram.
    # Initialise every cell to 'floor'; overwrite with observed log-probs below.
    qg_arr = np.full((26, 26, 26, 26), floor, dtype=np.float32)
    for qg, cnt in counts.items():
        idx = [ord(c) - 65 for c in qg]
        if all(0 <= i < 26 for i in idx) and cnt > 0:
            # Store log10(count / total) = log10(count) - log10(total).
            qg_arr[idx[0], idx[1], idx[2], idx[3]] = math.log10(cnt) - log_total

    return qg_arr, floor


def parse_ciphertext(path: Path) -> np.ndarray:
    """
    Parse the ciphertext file and return a flat 1-D int32 array of
    symbol indices (0-7).  The file format (26 tab-separated columns
    per line) is treated as cosmetic; all symbols are read in order.
    """
    seq: list[int] = []
    with open(path) as fh:
        for line in fh:
            for tok in line.strip().split('\t'):
                t = tok.strip()
                if t in SYM2IDX:
                    seq.append(SYM2IDX[t])
    return np.array(seq, dtype=np.int32)


# ==============================================================================
# Pair-IOC verification
# ==============================================================================

def pairwise_ioc(ca: np.ndarray, cb: np.ndarray) -> float:
    """
    Bigram IOC between two ciphertext block columns.
    bigram[r] = ca[r] * 8 + cb[r].  Returns IOC over all rows.
    """
    n = min(len(ca), len(cb))
    bg = ca[:n] * N_SYM + cb[:n]
    cnts = np.bincount(bg, minlength=N_CELLS)
    return float(cnts.dot(cnts - 1)) / (n * (n - 1))


# ==============================================================================
# Polybius mapping utilities
# ==============================================================================

def freq_init_polybius(bigrams: np.ndarray, skip_top: int = 1) -> np.ndarray:
    """
    Frequency-rank initialisation.

    skip_top : number of most-frequent bigrams to leave unmapped (-1).
               Default 1 skips the single most-frequent bigram, which in
               German text with spaces is almost always the space character.

    The next 26 bigrams (by descending count) are mapped to German letter
    frequency order (E N I S R A T D H U L C G M O B W F K Z P V J Y X Q).

    Returns an int32 (N_CELLS,) Polybius mapping; -1 = unmapped.
    """
    cnts  = np.bincount(bigrams.astype(np.int64), minlength=N_CELLS)
    order = np.argsort(-cnts)   # bigram cell indices sorted most-to-least frequent

    poly = np.full(N_CELLS, -1, dtype=np.int32)   # -1 = unmapped
    letter_rank = 0
    for i in range(N_CELLS):
        bi = int(order[i])
        if cnts[bi] == 0:
            break                     # remaining bigrams have zero count; stop
        if i < skip_top:
            continue                  # skip the top bigrams (space / sentence-end)
        if letter_rank < 26:
            # Map this bigram to the letter_rank-th most-frequent German letter.
            # E.g. letter_rank=0 -> E (most frequent in German, ~17%),
            #      letter_rank=1 -> N (~10%), letter_rank=2 -> I (~7.5%), ...
            # This warm-start lets the HC reach good text in far fewer iterations
            # than a random initialisation, because the frequency ranking of the
            # ciphertext bigrams closely mirrors the frequency ranking of German
            # letters (both reflect the same underlying plaintext distribution).
            poly[bi] = ord(GERMAN_FREQ_ORDER[letter_rank]) - ord('A')
            letter_rank += 1
    return poly


def decode(bigrams: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Map bigram indices to letter indices; drop unmapped (-1) entries."""
    letters = poly[bigrams]
    return letters[letters >= 0]


def score_qg(letters: np.ndarray, qg_arr: np.ndarray) -> float:
    """
    Average German quadgram log-probability per quadgram.

    Using the AVERAGE (not sum) prevents the optimizer from artificially
    improving its score by unmapping high-frequency bigrams.
    """
    if letters.size < 4:
        return -1e18
    n_qg = letters.size - 3
    return float(
        qg_arr[letters[:-3], letters[1:-2], letters[2:-1], letters[3:]].sum()
    ) / n_qg


def precompute_letter_structure(
        bigrams: np.ndarray,
        skip_top: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Identify the 26 letter bigram cells (ranks skip_top..skip_top+25 by
    frequency) and return their cell values and the subset of bigrams array
    positions that are those cells.

    This precomputation allows the SA inner loop to:
      1. Only pick from the 26 letter cells when choosing swap candidates
         (avoids ever-unmapping the space or any non-letter cell).
      2. Score only the letter positions (fast, no filtering needed).

    Returns
    -------
    letter_cells   : int32 array, shape (26,)   -- Polybius cell indices
    letter_bigrams : int32 array, shape (N_let,) -- subset of bigrams that are letter cells
    """
    cnts  = np.bincount(bigrams.astype(np.int64), minlength=N_CELLS)
    order = np.argsort(-cnts)   # bigram cell indices, most-to-least frequent

    # The 26 letter cells are ranks skip_top .. skip_top+25.
    # (Rank 0 .. skip_top-1 are the space/separator cells, left unmapped.)
    letter_cells = np.array(
        [int(order[i]) for i in range(skip_top, skip_top + 26)],
        dtype=np.int32,
    )

    # letter_bigrams: the subset of the bigram stream whose cell value is one
    # of the 26 letter cells.  The SA inner loop scores only this subsequence,
    # which is dramatically faster than scoring the full 3524-bigram stream:
    #   - Space bigrams (~15% of 3524 = ~529) are excluded.
    #   - The quadgram score is insensitive to them anyway (spaces are dropped).
    #   - Scoring ~2730 letters instead of 3524 total gives ~23% speedup per step.
    letter_mask    = np.isin(bigrams, letter_cells)
    letter_bigrams = bigrams[letter_mask]

    return letter_cells, letter_bigrams


def init_letter_poly(
        letter_cells: np.ndarray,
        letter_bigrams: np.ndarray,
) -> np.ndarray:
    """
    Initialize the Polybius poly array so that letter_cells are mapped to
    A-Z in German frequency order.

    Returns full (N_CELLS,) poly with -1 for non-letter cells and
    distinct A-Z assignments for the 26 letter cells.
    """
    cnts = np.bincount(letter_bigrams.astype(np.int64), minlength=N_CELLS)

    # Sort letter_cells by their frequency (descending) and assign German freq order
    cell_counts   = cnts[letter_cells]
    sorted_idx    = np.argsort(-cell_counts)   # indices into letter_cells, freq-desc

    poly = np.full(N_CELLS, -1, dtype=np.int32)
    for rank, lc_idx in enumerate(sorted_idx):
        cell      = letter_cells[lc_idx]
        poly[cell] = ord(GERMAN_FREQ_ORDER[rank]) - ord('A')

    return poly


# ==============================================================================
# Hill-climbing (quick pass)
# ==============================================================================

def hc_polybius(
        bigrams:   np.ndarray,
        qg_arr:    np.ndarray,
        n_iter:    int = 300_000,
        seed:      int = 0,
        poly_init: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """
    Hill-climbing on the Polybius mapping (skip_top=1, letter bigrams only).

    Move: swap any two non-space cells (both letter and non-letter cells
    are eligible, but the space cell is never swapped).  The average-per-
    quadgram score prevents the HC from collapsing to fewer letters.

    Used for Phase 1 (quick pass on all 24 arrangements).
    Returns (best_poly, best_score).
    """
    rng  = np.random.default_rng(seed)
    poly = (freq_init_polybius(bigrams, skip_top=1) if poly_init is None
            else poly_init.copy())

    # Fix: identify space bigram and never swap it
    cnts     = np.bincount(bigrams.astype(np.int64), minlength=N_CELLS)
    space_bi = int(np.argmax(cnts))
    poly[space_bi] = -1   # ensure space is excluded

    lets  = decode(bigrams, poly)
    score = score_qg(lets, qg_arr)
    best_poly  = poly.copy()
    best_score = score

    for _ in range(n_iter):
        a = int(rng.integers(N_CELLS))
        b = int(rng.integers(N_CELLS))
        if a == b or a == space_bi or b == space_bi:
            continue

        poly[a], poly[b] = poly[b], poly[a]

        lets      = decode(bigrams, poly)
        new_score = score_qg(lets, qg_arr)

        if new_score >= score:
            score = new_score
            if score > best_score:
                best_poly  = poly.copy()
                best_score = score
        else:
            poly[a], poly[b] = poly[b], poly[a]

    return best_poly, best_score


# ==============================================================================
# Simulated Annealing (thorough pass)
# ==============================================================================

def sa_polybius(
        letter_cells:   np.ndarray,
        letter_bigrams: np.ndarray,
        qg_arr:         np.ndarray,
        n_iter:         int   = 2_000_000,
        seed:           int   = 0,
        T0:             float = 0.5,
        Tmin:           float = 0.001,
        poly_init:      np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """
    Simulated Annealing for the Polybius mapping.

    Move: swap two of the 26 letter cells only.  This constrains the search
    to the space of assignments of the 26 most-frequent non-space bigrams
    to the 26 German letters, giving a clean 26! permutation problem where
    German quadgrams provide unambiguous signal.

    WHY SA BEATS HC: HC gets stuck at local optima (~-4.31 per quadgram).
    SA accepts temporarily-worse moves with probability exp(delta/T), allowing
    it to escape these traps and converge toward the global optimum (~-3.0 to
    -3.5 for correct German text).

    Temperature decays exponentially from T0 to Tmin over n_iter steps.

    Returns (best_poly, best_score) where best_poly is a full (N_CELLS,) array
    with -1 for non-letter cells and letter indices for the 26 letter cells.
    """
    rng  = np.random.default_rng(seed)
    n_lc = len(letter_cells)       # = 26

    # Initialize
    if poly_init is None:
        poly = init_letter_poly(letter_cells, letter_bigrams)
    else:
        poly = poly_init.copy()

    letters    = poly[letter_bigrams]   # (N_let,) -- all >= 0, no spaces
    score      = score_qg(letters, qg_arr)
    best_poly  = poly.copy()
    best_score = score

    # Precompute the per-step temperature decay factor.
    # Temperature decays exponentially: T(step) = T0 * exp(step * log_decay).
    # With T0=0.5 and Tmin=0.001 over n_iter steps, temperature halves roughly
    # every n_iter * ln(2) / ln(T0/Tmin) steps -- giving controlled annealing.
    log_decay = math.log(Tmin / T0) / n_iter

    for step in range(n_iter):
        # Pick two distinct letter cells to swap.
        # SA restricts moves to the 26 letter cells (same as HC), so the space
        # cell is never perturbed and all 26 letter assignments stay valid.
        i = int(rng.integers(n_lc))
        j = int(rng.integers(n_lc))
        if i == j:
            continue

        a = letter_cells[i]
        b = letter_cells[j]

        poly[a], poly[b] = poly[b], poly[a]

        letters   = poly[letter_bigrams]
        new_score = score_qg(letters, qg_arr)

        delta = new_score - score          # positive = improvement
        T     = T0 * math.exp(step * log_decay)   # current temperature

        # Metropolis acceptance criterion:
        #   - Always accept improvements (delta >= 0).
        #   - Accept worsening moves with probability exp(delta/T).
        #     At high T (early), exp(delta/T) ~ 1 for small |delta|: almost any
        #     move is accepted, allowing broad exploration of the landscape.
        #     At low T (late), exp(delta/T) -> 0: only near-neutral or improving
        #     moves are accepted, equivalent to hill-climbing.
        if delta >= 0 or rng.random() < math.exp(delta / T):
            score = new_score
            if score > best_score:
                best_poly  = poly.copy()
                best_score = score
        else:
            # Reject: undo swap.
            poly[a], poly[b] = poly[b], poly[a]

    return best_poly, best_score


# ==============================================================================
# Arrangement search
# ==============================================================================

def make_bigrams_for_arrangement(
        block_cols: list[np.ndarray],
        pair_order: tuple[int, ...],
) -> np.ndarray:
    """
    Build the 3524-bigram sequence for a given arrangement.

    pair_order: a permutation of range(4), specifying which cipher-block pair
    goes to fractionated-text positions (0,1), (2,3), (4,5), (6,7).

    Within-pair orientation is fixed as (lower_index, higher_index); any
    orientation ambiguity is absorbed by the Polybius mapping optimization.

    Returns bigrams as a (3524,) int32 array.
    """
    n_rows  = COL_LEN   # 881
    n_pairs = 4         # bigrams per fractionated-text row
    bigrams = np.empty(n_rows * n_pairs, dtype=np.int32)

    for pos, pair_idx in enumerate(pair_order):
        ca, cb = CIPHER_PAIRS[pair_idx]
        col_a  = block_cols[ca]   # (881,) -- ROW symbols for all 881 rows
        col_b  = block_cols[cb]   # (881,) -- COL symbols for all 881 rows
        # Encode (row_sym, col_sym) as a single integer in [0, 63].
        bg     = col_a * N_SYM + col_b    # (881,)

        # Interleave into the output array at stride n_pairs.
        # bigrams[pos::n_pairs] sets positions pos, pos+4, pos+8, ...
        # so that all four bigrams for row r land at positions 4r, 4r+1, 4r+2, 4r+3.
        # This reconstructs the original left-to-right order of the 4 plaintext
        # characters that were encoded into each row of the transposition grid.
        bigrams[pos::n_pairs] = bg

    return bigrams


# ==============================================================================
# Separator identification
# ==============================================================================

def identify_separator(letters: np.ndarray) -> int:
    """
    Identify which decoded letter most likely represents the word separator
    (space, or most frequent non-alphabetic character).

    In German text with ~15% spaces, the separator's decoded letter will appear
    well above its expected German letter frequency.  Returns the letter index
    (0-25) of the most over-represented letter relative to German expectations,
    or -1 if no letter exceeds its expectation by more than 5%.
    """
    n   = max(len(letters), 1)
    obs = np.bincount(letters, minlength=26).astype(np.float64) / n

    # For each letter, compute how much its observed frequency exceeds the
    # expected German frequency.  The space/separator character will appear at
    # approximately 15% of all decoded positions, but its "German letter"
    # frequency is at most ~17% (for E).  Any letter appearing ~15% above its
    # baseline is almost certainly the space character.
    excess = obs - GERMAN_FREQ
    best = int(np.argmax(excess))
    # Require at least 5% excess to confidently call it the separator;
    # below that threshold the identification would be ambiguous.
    return best if float(excess[best]) > 0.05 else -1


# ==============================================================================
# Output helpers
# ==============================================================================

def letters_to_str(letters: np.ndarray, sep_letter: int = -1) -> str:
    """
    Convert letter index array to a printable string.
    If sep_letter >= 0, that letter is rendered as a space character.
    """
    chars = []
    for li in letters:
        li = int(li)
        if li == sep_letter:
            chars.append(' ')
        else:
            chars.append(chr(65 + li))
    return ''.join(chars)


def print_polybius_square(poly: np.ndarray) -> None:
    """Pretty-print the recovered 8x8 Polybius square."""
    print("     " + "   ".join(SYM_NAMES))
    print("    +" + "----" * N_SYM)
    for row in range(N_SYM):
        cells = []
        for col in range(N_SYM):
            idx = row * N_SYM + col
            li  = int(poly[idx])
            cells.append(chr(65 + li) if 0 <= li <= 25 else '.')
        print(f"  {SYM_NAMES[row]} | " + "  ".join(cells))


# ==============================================================================
# Main
# ==============================================================================

def main() -> None:
    t0 = time.time()

    # ---- Load resources -------------------------------------------------------
    print("Loading German quadgrams ...", flush=True)
    qg_arr, floor = load_quadgrams(QUADGRAMS_PATH)
    n_loaded = int(np.sum(qg_arr > floor))
    print(f"  {n_loaded:,} quadgrams  ({time.time()-t0:.1f}s)", flush=True)

    print("\nParsing ciphertext ...", flush=True)
    flat = parse_ciphertext(CIPHERTEXT_PATH)
    print(f"  {len(flat):,} symbols total", flush=True)
    print(f"  Key length 8: {len(flat)} = 8 x {len(flat)//8} (exact: {len(flat)%8==0})",
          flush=True)

    # Split the flat symbol sequence into 8 equal block-columns.
    # block_cols[j] = the j-th column of the transposition grid (881 symbols).
    # Each column is exclusively ROW symbols or exclusively COL symbols, because
    # the columnar transposition permutes whole columns at a time.
    block_cols = [flat[j * COL_LEN:(j + 1) * COL_LEN] for j in range(KEY_LEN)]
    print(f"  Block column length: {COL_LEN}", flush=True)

    # ---- Step 1: Verify cipher-block pairs ------------------------------------
    print("\n-- Step 1: Cipher-block pair verification --", flush=True)
    for ca, cb in CIPHER_PAIRS:
        ioc = pairwise_ioc(block_cols[ca], block_cols[cb])
        print(f"  Pair (block {ca}, block {cb}): IOC = {ioc:.5f}"
              f"  (German ~= 0.074, random ~= 0.016)", flush=True)

    # ---- Step 2: Arrangement search -------------------------------------------
    all_arrangements = list(permutations(range(4)))  # 4! = 24
    print(
        f"\n-- Step 2: Arrangement search ({len(all_arrangements)} orderings) --",
        flush=True,
    )

    # ---- Quick pass: frequency-initialised score for all 24 arrangements ----
    # For each arrangement, build the bigram sequence, apply the German-frequency
    # Polybius initialisation (no HC yet), and record the initial score.
    # This cheap pass identifies the most promising arrangements to HC-optimise.
    # Arrangements where the frequency-matched initialisation already produces a
    # good score are likely close to a real German text decoding.
    print(f"  Quick-scoring all {len(all_arrangements)} arrangements ...", flush=True)
    quick_results: list[tuple[float, tuple, np.ndarray]] = []
    for arr in all_arrangements:
        bigrams = make_bigrams_for_arrangement(block_cols, arr)
        poly    = freq_init_polybius(bigrams, skip_top=1)
        lets    = decode(bigrams, poly)
        score   = score_qg(lets, qg_arr)
        quick_results.append((score, arr, bigrams))

    quick_results.sort(key=lambda x: -x[0])
    print(f"  Top 5 arrangements by quick score:", flush=True)
    for score, arr, _ in quick_results[:5]:
        print(f"    arrangement {arr}: quick score = {score:.2f}", flush=True)

    # ---- Phase 1: HC quick pass on all 24 arrangements ----
    # Run a short HC (2 restarts x 300k) on every arrangement.
    # The purpose is to move each arrangement from its frequency-init starting
    # point to the nearest local optimum, which is enough to reliably rank
    # arrangements and identify the top candidates for the deeper SA phase.
    # Two restarts catch cases where the first seed lands in a poor local trap.
    print(
        f"\n  Phase 1 HC ({N_HC_RESTARTS_P1} restarts x {N_HC_ITER_P1:,})"
        f" on all {len(all_arrangements)} ...",
        flush=True,
    )
    phase1: list[tuple[float, tuple, np.ndarray, np.ndarray]] = []
    for score_q, arr, bigrams in quick_results:
        best_poly  = None
        best_score = -1e18
        for r in range(N_HC_RESTARTS_P1):
            poly, score = hc_polybius(bigrams, qg_arr, n_iter=N_HC_ITER_P1, seed=r)
            if score > best_score:
                best_score = score
                best_poly  = poly.copy()
        sample = letters_to_str(decode(bigrams[:80], best_poly))[:40]
        print(f"    arr {arr}: score={best_score:.2f}  |  {sample}", flush=True)
        phase1.append((best_score, arr, bigrams, best_poly))

    phase1.sort(key=lambda x: -x[0])

    # ---- Phase 2: SA on top N_TOP_P2 arrangements ----
    # SA can escape the local optima that trap Phase 1 HC, at the cost of
    # longer runtime.  We only run SA on the top 8 arrangements from Phase 1
    # (not all 24) to keep runtime manageable.
    #
    # For each arrangement:
    #   - Precompute letter_cells and letter_bigrams (the 26-cell subset).
    #     SA only swaps within letter_cells, maintaining the bijection constraint
    #     and scoring only the ~2730 letter-bigrams (not the full 3524).
    #   - Run N_SA_RESTARTS_P2 independent SA chains from different seeds.
    #     Different seeds start from different frequency-init points, exploring
    #     different valleys in the 26! assignment space.
    #   - Keep the overall best poly across all SA restarts for this arrangement.
    print(
        f"\n  Phase 2 SA ({N_SA_RESTARTS_P2} restarts x {N_SA_ITER_P2:,}"
        f", T: {SA_T0}->{SA_TMIN}) on top {N_TOP_P2} ...",
        flush=True,
    )

    final_results: list[tuple[float, tuple, np.ndarray, np.ndarray]] = []
    for _, arr, bigrams, poly_p1 in phase1[:N_TOP_P2]:
        # Precompute which cells are the 26 letter cells and the bigram subset
        # that hits only those cells.  Both are reused across all SA restarts.
        letter_cells, letter_bigrams = precompute_letter_structure(bigrams, skip_top=1)

        # Initialise from the Phase 1 HC result (a good warm start for SA).
        best_poly  = poly_p1.copy()
        best_score = score_qg(decode(bigrams, poly_p1), qg_arr)

        for r in range(N_SA_RESTARTS_P2):
            # Use widely-spaced seeds (r*17+200) to decorrelate the SA chains.
            poly, score = sa_polybius(
                letter_cells, letter_bigrams, qg_arr,
                n_iter=N_SA_ITER_P2, seed=r * 17 + 200,
                T0=SA_T0, Tmin=SA_TMIN,
            )
            if score > best_score:
                best_score = score
                best_poly  = poly.copy()

        sample = letters_to_str(decode(bigrams[:80], best_poly))[:40]
        print(f"    arr {arr}: score={best_score:.2f}  |  {sample}", flush=True)
        final_results.append((best_score, arr, bigrams, best_poly))

    final_results.sort(key=lambda x: -x[0])
    best_score, best_arr, best_bigrams, best_poly = final_results[0]

    # ---- Step 3: Decode and identify separator --------------------------------
    # Decode the full bigram stream to letters, then identify the separator.
    # decode() drops unmapped (-1) cells, so all_letters contains only the
    # 26-letter decoded positions (no spaces).
    all_letters = decode(best_bigrams, best_poly)
    # identify_separator() finds the letter most over-represented relative to
    # German baseline -- this is the word-space character masquerading as a letter.
    sep_letter  = identify_separator(all_letters)
    sep_char    = chr(65 + sep_letter) if sep_letter >= 0 else '?'

    plaintext_raw   = letters_to_str(all_letters)                            # no spaces
    plaintext_clean = letters_to_str(all_letters, sep_letter=sep_letter)     # spaces shown

    print(f"\n-- Result --", flush=True)
    print(f"  Best arrangement : {best_arr}", flush=True)
    print(f"  Quadgram score   : {best_score:.2f}", flush=True)
    print(f"  Decoded letters  : {len(plaintext_raw)}", flush=True)
    print(f"  Separator letter : {sep_char} (index {sep_letter})", flush=True)
    print(f"  Plaintext (first 600 chars, sep shown as space):", flush=True)
    for i in range(0, min(len(plaintext_clean), 600), 60):
        print(f"    {plaintext_clean[i:i+60]}", flush=True)

    print("\n  Polybius square (recovered):", flush=True)
    print_polybius_square(best_poly)

    # ---- Save -----------------------------------------------------------------
    out_path = SCRIPT_DIR / "decrypted_stage07.txt"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("# Stage 07 -- ADFGVX-variant decryption\n")
        fh.write(f"# Key length         : {KEY_LEN}\n")
        fh.write(f"# Cipher pairs       : {CIPHER_PAIRS}\n")
        fh.write(f"# Best arrangement   : {best_arr}\n")
        fh.write(f"# Quadgram score     : {best_score:.2f}\n")
        fh.write(f"# Separator letter   : {sep_char} (index {sep_letter})\n")
        fh.write(f"# Polybius mapping   : {best_poly.tolist()}\n\n")
        fh.write("PLAINTEXT (separator shown as space):\n")
        fh.write(plaintext_clean)
        fh.write("\n")

    print(f"\n  Saved -> {out_path}  ({len(plaintext_raw)} decoded letters)", flush=True)
    print(f"  Total time: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
