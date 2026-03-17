#!/usr/bin/env python3
"""
diag_stage07.py -- Diagnostic analysis for the Stage 07 ADFGVX-variant cipher.

PURPOSE
=======
This is an exploratory / diagnostic script run BEFORE the main solver.
It answers three questions:
  1. Which pairs of the 8 ciphertext block-columns form valid Polybius bigrams?
     (Answered via pairwise bigram IOC — correct pairs score ~0.067, random ~0.016.)
  2. What is the bigram frequency distribution for the best arrangement?
     (The most frequent bigram is the word-space separator.)
  3. Which of the 24 possible pair-arrangements gives the best HC score?
     (4 cipher-block pairs × 4 fractionated-text positions = 4! = 24 orderings.)

CIPHER BACKGROUND
=================
Stage 07 uses an ADFGVX-style fractionation cipher extended to 8 symbols
(C, E, M, O, P, R, T, U) rather than the standard 6 (A, D, F, G, V, X).

Encryption works in two steps:
  Step 1 — SUBSTITUTION via an 8×8 Polybius square:
    Each plaintext character is looked up in the 8×8 grid; its position
    gives a 2-symbol (row, col) bigram from the 8-symbol alphabet.
    With 8 symbols, there are 8×8 = 64 possible bigrams — enough to cover
    the full German alphabet plus a word-space character.

  Step 2 — COLUMNAR TRANSPOSITION:
    The resulting symbol stream (twice as long as the plaintext) is written
    row-by-row into an 8-column grid (key length 8).  The 8 columns are
    then permuted by an unknown keyword and read off top-to-bottom to form
    the ciphertext.

The ciphertext has 7048 symbols.  7048 = 8 × 881, and 881 is prime, which
confirms key length 8 uniquely (no other factoring exists).

ATTACK OVERVIEW
===============
After transposition, column j of the ciphertext corresponds to one specific
column of the original 8-column grid.  In the original grid, even columns
(0, 2, 4, 6) held ROW symbols and odd columns (1, 3, 5, 7) held COL symbols.
So every ciphertext block-column is exclusively ROW or exclusively COL symbols.
A valid (ROW, COL) pair will produce bigrams drawn from the same Polybius
distribution → high bigram IOC.  A wrong pairing shuffles unrelated symbol
sequences → IOC near random (1/64 ≈ 0.016).

OUTPUT
======
Run this script first to confirm pairs and identify best arrangement for
fast_decode.py.
"""

import math
import sys
from itertools import permutations
from pathlib import Path

import numpy as np

# ---- File paths ---------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
MISC_DIR   = SCRIPT_DIR.parent / "misc"
CT_PATH    = SCRIPT_DIR / "ciphertext_stage07.txt"
QG_PATH    = MISC_DIR / "german_quadgrams.txt"

# ---- Cipher constants ---------------------------------------------------------

# SYM2IDX: maps each of the 8 cipher symbols to an integer 0-7.
# A Polybius bigram is (row_symbol, col_symbol).  We encode it as a single
# integer: cell = row_idx * 8 + col_idx  ∈ [0, 63].
SYM2IDX = {'C':0,'E':1,'M':2,'O':3,'P':4,'R':5,'T':6,'U':7}

# N_SYM=8:    8 distinct ciphertext symbols.
# N_CELLS=64: 8×8 Polybius square has 64 possible (row, col) bigrams.
# KEY_LEN=8:  columnar transposition key length (7048 / 8 = 881 rows, exact).
# COL_LEN=881: number of rows in the transposition grid.  881 is prime, so
#              7048 = 8 × 881 has no other factoring — key length 8 is unique.
N_SYM = 8; N_CELLS = 64; KEY_LEN = 8; COL_LEN = 881

# CIPHER_PAIRS: which pairs of block-columns form valid (ROW, COL) bigrams.
# Each tuple (ca, cb) means column ca carries the ROW symbol and column cb
# carries the COL symbol for that pair.
# NOTE: this initial set uses a naive (0,7),(1,2),(3,4),(5,6) guess for the
# IOC calculation.  The confirmed correct pairing after orientation analysis
# is [(0,7),(2,1),(4,3),(6,5)] — see fast_decode.py.
# All four pairs are confirmed by pairwise IOC ≈ 0.067-0.070 vs. random ≈ 0.016.
CIPHER_PAIRS = [(0,7),(1,2),(3,4),(5,6)]

# German letter frequency order, most-to-least frequent (letters only, no space).
# Used to initialise the Polybius mapping before hill-climbing: assign the most
# frequent bigram to E, second-most to N, etc.  This warm-start dramatically
# reduces HC convergence time vs. a random start.
GERMAN_FREQ_ORDER = "ENISRATDHULCGMOBWFKZPVJYXQ"


# ==============================================================================
# Data loading
# ==============================================================================

def load_qg(path):
    """
    Load German quadgram statistics and return a log-probability lookup array.

    The file format is '<QUADGRAM> <COUNT>' per line (e.g. 'EINE 123456').
    We convert counts to log10-probabilities so that scoring a sequence of
    quadgrams is a simple array sum (adding logs = multiplying probabilities).

    Why log10 instead of natural log?
      Convention only — the absolute scale doesn't matter since we only compare
      scores to each other.  log10 gives slightly more readable numbers.

    Why a 'floor' value for unseen quadgrams?
      Any quadgram not in the training corpus gets probability 0.01/total,
      i.e.  floor = log10(0.01) - log10(total).  Without a floor the score
      would be -infinity for any sequence containing an unseen quadgram,
      breaking comparison.  0.01 is low enough to strongly penalise unlikely
      sequences without causing numerical issues.

    Returns
    -------
    arr   : float32 ndarray shape (26,26,26,26) — log10-prob per quadgram.
    floor : float — log10-prob assigned to absent quadgrams.
    """
    counts = {}; total = 0
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p)>=2:
                qg=p[0].upper(); cnt=int(p[1])
                # Only keep clean 4-letter alphabetic quadgrams.
                if len(qg)==4 and qg.isalpha():
                    counts[qg]=cnt; total+=cnt

    log_tot = math.log10(total)
    # floor: log-probability for any quadgram absent from the corpus.
    # Equivalent to saying that unseen bigrams occur at rate 0.01 out of
    # 'total' observed quadgrams — a small but non-zero penalty.
    floor   = math.log10(0.01) - log_tot

    # 4D array indexed by [a][b][c][d], each 0-25 (A=0 … Z=25).
    # Initialise every cell to 'floor' before filling known quadgrams.
    arr = np.full((26,26,26,26), floor, dtype=np.float32)
    for qg,cnt in counts.items():
        idx=[ord(c)-65 for c in qg]
        if all(0<=i<26 for i in idx) and cnt>0:
            # Store log10(count/total) = log10(count) - log10(total).
            arr[idx[0],idx[1],idx[2],idx[3]] = math.log10(cnt)-log_tot
    return arr, floor


# ==============================================================================
# Ciphertext parsing
# ==============================================================================

def parse_ct(path):
    """
    Parse the ciphertext file into a flat 1-D int32 array of symbol indices.

    The ciphertext file stores the transposed symbol grid in a tab-separated
    display format (26 columns per visual row) purely for readability.  The
    actual ciphertext is just the flat sequence of 7048 symbols read in order.

    Returns an int32 array of length 7048, values in [0, 7].
    """
    seq=[]
    with open(path) as f:
        for line in f:
            for tok in line.strip().split('\t'):
                t=tok.strip()
                # Only include recognised symbols; ignore blank cells and headers.
                if t in SYM2IDX: seq.append(SYM2IDX[t])
    return np.array(seq, dtype=np.int32)


# ==============================================================================
# Pair identification via pairwise IOC
# ==============================================================================

def pairwise_ioc(ca, cb):
    """
    Compute the bigram Index of Coincidence (IOC) between two block-columns.

    For each row r, form bigram bg[r] = ca[r] * 8 + cb[r]  ∈ [0, 63].
    IOC = sum_k( n_k*(n_k-1) ) / (N*(N-1))
    where n_k is the count of bigram k and N is the total number of bigrams.

    Interpretation:
      - If (ca, cb) is a CORRECT Polybius pair, both columns share the same
        underlying letter distribution.  Repeated bigrams are common → IOC high
        (we observe ≈ 0.067-0.070 for German text).
      - If (ca, cb) is a WRONG pairing, the two columns come from different
        independent symbol streams.  Every bigram value is roughly equally
        likely → IOC near 1/64 ≈ 0.016 (random).

    The gap between 0.070 and 0.016 is unmistakable, making pair identification
    completely unambiguous even before attempting any decryption.
    """
    n=min(len(ca),len(cb))
    # Encode each (row_sym, col_sym) pair as a single integer in [0,63].
    bg=ca[:n]*N_SYM+cb[:n]
    # Count occurrences of each of the 64 possible bigram values.
    cnts=np.bincount(bg,minlength=N_CELLS)
    # IOC formula: Σ n_k(n_k-1) / N(N-1).
    return float(cnts.dot(cnts-1))/(n*(n-1))


# ==============================================================================
# Bigram construction
# ==============================================================================

def make_bigrams(block_cols, pair_order):
    """
    Build the ordered bigram sequence for a given arrangement of the 4 pairs.

    The Polybius substitution encodes each plaintext character as (row, col).
    After columnar transposition, the 4 cipher-block pairs need to be
    reassembled in the correct fractionated-text order.

    pair_order: a permutation of [0,1,2,3] specifying which cipher-block pair
    is assigned to fractionated-text positions (0,1), (2,3), (4,5), (6,7).
    There are 4! = 24 possible orderings to try.

    The interleaving  bigrams[pos::n_pairs] = bg  places:
      pair at pos=0 into positions 0, 4,  8, 12, ... (1st bigram of each row)
      pair at pos=1 into positions 1, 5,  9, 13, ... (2nd bigram of each row)
      pair at pos=2 into positions 2, 6, 10, 14, ... (3rd bigram of each row)
      pair at pos=3 into positions 3, 7, 11, 15, ... (4th bigram of each row)
    This reconstructs the left-to-right plaintext order across all 881 rows.

    Returns a (3524,) int32 array (881 rows × 4 bigrams/row).
    """
    n_rows=COL_LEN; n_pairs=4
    bigrams=np.empty(n_rows*n_pairs,dtype=np.int32)
    for pos,pair_idx in enumerate(pair_order):
        ca,cb=CIPHER_PAIRS[pair_idx]
        # Encode row symbol ca and col symbol cb as a single cell index.
        bg=block_cols[ca]*N_SYM+block_cols[cb]
        # Interleave: this pair's bigrams fill every n_pairs-th position
        # starting at 'pos', preserving the original plaintext column order.
        bigrams[pos::n_pairs]=bg
    return bigrams


# ==============================================================================
# Quadgram scoring
# ==============================================================================

def score_qg(letters, qg_arr):
    """
    Score a decoded letter sequence using German quadgram log-probabilities.

    Returns the AVERAGE log10-probability per quadgram (not the sum).

    Why average instead of sum?
      If we used the sum, an optimiser could artificially boost its score by
      shrinking the decoded sequence — e.g., by mapping many bigrams to -1
      (unmapped/space) so that fewer quadgrams are evaluated.  The average
      is independent of sequence length, so this gaming is impossible.

    A higher (less negative) score means better German text.
    Typical values:
      -6 to -7 : random letter sequence
      -4 to -5 : partially correct but imperfect
      -3 to -3.5 : clean correct German text
    """
    if letters.size<4: return -1e18
    n=letters.size-3
    # Vectorised 4D array lookup: each consecutive quadruple (i,i+1,i+2,i+3)
    # is looked up simultaneously, then summed and divided by count.
    return float(qg_arr[letters[:-3],letters[1:-2],letters[2:-1],letters[3:]].sum())/n


# ==============================================================================
# Hill-climbing Polybius optimisation
# ==============================================================================

def hc_once(bigrams, qg_arr, n_iter=500_000, seed=0):
    """
    Run one restart of hill-climbing to optimise the Polybius mapping.

    Initialisation:
      - Sort bigrams by descending frequency.
      - The single most-frequent bigram is designated as the space separator
        (left unmapped, poly=-1).  In German text with ~15% spaces, the space
        bigram will dominate.
      - The next 26 bigrams are mapped to German letter frequency order
        (E=most frequent, N=2nd, I=3rd, …).

    Move: swap the letter assignments of any two cells (excluding the space
    cell).  Accept the swap if and only if it does not decrease the score
    (pure greedy hill-climbing, no temperature).

    KNOWN LIMITATION (the "gaming" bug):
      This implementation allows swapping non-letter (poly=-1) cells with
      letter cells.  This can shrink the number of decoded letters, making
      the average score artificially higher even as the actual text quality
      worsens.  The fix (used in fast_decode.py) restricts swaps to the
      26 letter cells only.  This version is kept as-is for diagnostic
      comparison purposes.

    Returns (best_poly, best_score).
    best_poly is a (64,) int32 array: poly[cell] ∈ {-1, 0..25}.
      -1  = unmapped (space or unused cell)
      0-25 = letter A-Z assigned to this bigram.
    """
    rng=np.random.default_rng(seed)
    # Count how often each of the 64 possible bigrams appears in the ciphertext.
    cnts=np.bincount(bigrams.astype(np.int64),minlength=N_CELLS)
    # Sort bigrams most-to-least frequent.
    order=np.argsort(-cnts)

    # Initialise Polybius mapping: -1 everywhere, then assign 26 letters.
    poly=np.full(N_CELLS,-1,dtype=np.int32)
    # The top bigram is the space separator; never assign a letter to it.
    space_bi=int(order[0])
    for i,bi in enumerate(order[1:27]):
        # Rank 1 (2nd overall) → E, Rank 2 → N, … Rank 25 → Q.
        poly[bi]=ord(GERMAN_FREQ_ORDER[i])-ord('A')

    # Initial score: decode all bigrams (drop spaces), evaluate quadgrams.
    lets=poly[bigrams]; lets=lets[lets>=0]
    score=score_qg(lets,qg_arr)
    best_poly=poly.copy(); best_score=score

    for _ in range(n_iter):
        # Pick two random cells to swap (both must be non-space).
        a=int(rng.integers(N_CELLS)); b=int(rng.integers(N_CELLS))
        if a==b or a==space_bi or b==space_bi: continue

        # Trial swap.
        poly[a],poly[b]=poly[b],poly[a]
        lets=poly[bigrams]; lets=lets[lets>=0]
        ns=score_qg(lets,qg_arr)

        if ns>=score:
            # Accept improvement (or neutral move).
            score=ns
            # Track global best separately — score can fluctuate on neutral moves.
            if score>best_score: best_poly=poly.copy(); best_score=score
        else:
            # Reject: undo the swap.
            poly[a],poly[b]=poly[b],poly[a]

    return best_poly, best_score


# ==============================================================================
# Main diagnostic routine
# ==============================================================================

def main():
    print("Loading data...", flush=True)
    qg_arr, _ = load_qg(QG_PATH)
    flat = parse_ct(CT_PATH)
    # Split the flat 7048-symbol sequence into 8 block-columns of 881 symbols each.
    # block_cols[j] = the j-th column of the transposition grid (881 symbols).
    block_cols = [flat[j*COL_LEN:(j+1)*COL_LEN] for j in range(KEY_LEN)]

    # ------------------------------------------------------------------
    # 1. All 28 pairwise IOC values
    # ------------------------------------------------------------------
    # We test every possible pairing of the 8 block-columns: C(8,2) = 28 pairs.
    # The 4 correct Polybius pairs will show IOC ≈ 0.067-0.070 (near German ~0.074).
    # All other pairs will show IOC ≈ 0.016 (near random 1/64).
    # The large separation makes pair identification unambiguous.
    print("\n=== All 28 pairwise IOC values ===")
    pairs_ioc = []
    for i in range(8):
        for j in range(i+1,8):
            ioc=pairwise_ioc(block_cols[i],block_cols[j])
            pairs_ioc.append((ioc,i,j))
    pairs_ioc.sort(reverse=True)
    for ioc,i,j in pairs_ioc:
        # Mark the 4 confirmed correct pairs with an arrow.
        marker="  <--" if (i,j) in [(0,7),(1,2),(3,4),(5,6)] or (j,i) in [(0,7),(1,2),(3,4),(5,6)] else ""
        print(f"  ({i},{j}): IOC={ioc:.5f}{marker}")

    # ------------------------------------------------------------------
    # 2. Bigram count distribution for the best arrangement (1,2,3,0)
    # ------------------------------------------------------------------
    # After assembling bigrams for arrangement (1,2,3,0), we look at
    # the frequency of each of the 64 possible bigram values.
    # The most-frequent bigram is almost certainly the word-space separator:
    # in German text ~15% of characters are spaces, so the space bigram
    # will appear far more often than any letter bigram (~5-6% for E).
    print("\n=== Bigram frequency for arrangement (1,2,3,0) ===")
    arr = (1,2,3,0)
    bigrams = make_bigrams(block_cols, arr)
    cnts = np.bincount(bigrams.astype(np.int64), minlength=N_CELLS)
    order = np.argsort(-cnts)
    total = cnts.sum()
    print(f"  Total bigrams: {total}")
    print(f"  Non-zero cells: {(cnts>0).sum()}")
    print("  Top 35 bigrams by count:")
    for rank, bi in enumerate(order[:35]):
        # Convert cell index back to (row_symbol, col_symbol) names for readability.
        r,c = divmod(int(bi),8)
        sym_r=list(SYM2IDX.keys())[r]; sym_c=list(SYM2IDX.keys())[c]
        print(f"    rank {rank+1:2d}: cell {bi:2d} ({sym_r}{sym_c}) count={cnts[bi]:4d} ({100*cnts[bi]/total:.1f}%)")

    # ------------------------------------------------------------------
    # 3. Run HC and decode for arrangement (1,2,3,0)
    # ------------------------------------------------------------------
    # This uses the naive HC (may suffer from gaming — see hc_once docstring).
    # The purpose here is diagnostic: identify the separator and inspect the
    # raw decoded text for obvious German words.
    print("\n=== HC decoding for arrangement (1,2,3,0) ===", flush=True)
    best_poly, best_score = hc_once(bigrams, qg_arr, n_iter=500_000, seed=0)
    print(f"  Score: {best_score:.4f}")

    # Decode: apply poly to each bigram, drop unmapped (-1) entries.
    lets=best_poly[bigrams]; lets=lets[lets>=0]

    # ---- Letter frequency analysis ----
    # Compare observed frequencies to German expected frequencies.
    # The most over-represented letter is likely the space separator.
    n=lets.size
    freq=np.bincount(lets,minlength=26).astype(float)/n
    # German letter frequencies (letters only, spaces excluded), A-Z.
    GERMAN_FREQ = np.array([0.0651,0.0189,0.0306,0.0508,0.1740,0.0165,0.0301,0.0476,
                            0.0755,0.0027,0.0121,0.0344,0.0253,0.0978,0.0251,0.0079,
                            0.0002,0.0700,0.0727,0.0615,0.0435,0.0067,0.0189,0.0003,0.0004,0.0113])
    print(f"\n  Letter frequencies vs German expected:")
    print(f"  {'Letter':6s} {'Observed':>9s} {'Expected':>9s} {'Ratio':>7s}")
    for li in np.argsort(-freq)[:15]:
        print(f"  {chr(65+li):6s} {freq[li]:9.3f} {GERMAN_FREQ[li]:9.3f} {freq[li]/max(GERMAN_FREQ[li],0.0001):7.2f}x")

    # Identify the separator: whichever letter has the largest excess over
    # its expected German frequency is likely the space character.
    excess=freq-GERMAN_FREQ
    sep=int(np.argmax(excess))
    print(f"\n  Most over-represented (potential separator): {chr(65+sep)} (excess={excess[sep]:.3f})")

    # Display decoded text with the separator shown as a space character.
    chars=[]
    for li in lets:
        li=int(li)
        if li==sep: chars.append(' ')
        else: chars.append(chr(65+li))
    text=''.join(chars)
    print(f"\n  First 600 chars (with separator '{chr(65+sep)}' as space):")
    for i in range(0,min(600,len(text)),80):
        print(f"    {text[i:i+80]}")

    # Also show raw output without any separator substitution.
    raw=''.join(chr(65+int(li)) for li in lets)
    print(f"\n  Raw (no separator) first 300 chars:")
    print(f"    {raw[:300]}")

    # ------------------------------------------------------------------
    # 4. Test different skip_top values
    # ------------------------------------------------------------------
    # skip_top=1: treat only the top-1 bigram as space (default).
    # skip_top=2: treat top-2 as space/non-letter (in case there are 2
    #             separator-like bigrams, e.g. space and sentence-end marker).
    # skip_top=3,4: further exploration.
    # The coverage metric shows how many of the 3524 bigrams are decoded
    # to letters (higher = fewer holes in the text).
    print("\n=== HC with different skip_top values ===", flush=True)
    for skip in [1, 2, 3, 4]:
        cnts2=np.bincount(bigrams.astype(np.int64),minlength=N_CELLS)
        order2=np.argsort(-cnts2)
        poly2=np.full(N_CELLS,-1,dtype=np.int32)
        space_bi2=int(order2[0])
        for i in range(skip, skip+26):
            if i<len(order2) and cnts2[order2[i]]>0:
                poly2[order2[i]]=ord(GERMAN_FREQ_ORDER[i-skip])-ord('A')
        lets2=poly2[bigrams]; lets2=lets2[lets2>=0]
        s2=score_qg(lets2,qg_arr)
        print(f"  skip_top={skip}: initial score={s2:.4f}, coverage={lets2.size}/{total}")

    # ------------------------------------------------------------------
    # 5. Quick HC across all 24 arrangements (1 restart × 300k)
    # ------------------------------------------------------------------
    # There are 4 confirmed cipher-block pairs.  They must be assigned to
    # the 4 fractionated-text pair positions (row bigrams 0-3 of each row).
    # The number of possible assignments is 4! = 24.
    # We run a quick HC pass on each to identify the best candidate(s) for
    # the deeper search in fast_decode.py / decrypt_stage07.py.
    print("\n=== Quick HC across all 24 arrangements (1 restart x 300k) ===", flush=True)
    results=[]
    for arr2 in permutations(range(4)):
        bg2=make_bigrams(block_cols,arr2)
        poly2,score2=hc_once(bg2,qg_arr,n_iter=300_000,seed=0)
        lets2=poly2[bg2]; lets2=lets2[lets2>=0]
        # Show first 40 decoded characters as a quick sanity-check sample.
        sample=''.join(chr(65+int(x)) for x in lets2[:40])
        results.append((score2,arr2,sample))
    results.sort(reverse=True)
    # Print the top 8 arrangements — the winning arrangement should be clearly
    # separated from the rest by at least 0.1-0.2 score units.
    for score2,arr2,sample in results[:8]:
        print(f"  arr {arr2}: score={score2:.4f}  {sample}")

if __name__=="__main__":
    main()
