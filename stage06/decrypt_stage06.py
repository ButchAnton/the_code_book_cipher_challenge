#!/usr/bin/env python3
"""
Stage 06 -- Playfair Cipher Solver
====================================

Methodology
-----------
Cracks the stage-06 Playfair ciphertext with a two-phase search:

  Phase 1 -- Simulated Annealing (SA)
    Starting from a randomly shuffled 5x5 key square, SA makes random
    perturbations (cell swaps, row/column swaps, reversals, transpositions)
    and accepts improvements unconditionally.  Worsening moves are accepted
    with probability exp(delta / T), where T cools exponentially from
    T_START to T_END over SA_STEPS iterations.  This lets the search escape
    local optima early on while converging toward good solutions later.

  Phase 2 -- Hill-Climbing (HC)
    Pure greedy ascent from the best key found by SA.  Stops when no
    improvement occurs for HC_PATIENCE consecutive steps (or HC_MAX total
    steps, whichever comes first).

Each restart runs both phases independently.  The globally best result
across all restarts and all language scorers is kept.

Performance
-----------
The inner loop is vectorized with NumPy: Playfair decryption of all 332
digraphs is computed in one pass using masked array operations, and the
quadgram table is stored as a flat NumPy float64 array indexed by a single
integer (a*25^3 + b*25^2 + c*25 + d), giving a 20-40x speedup over the
pure-Python dict-lookup approach and enabling ~100 restarts per language.

Fitness function
----------------
Every candidate plaintext is scored by summing the log10-probabilities of
all 4-letter windows (quadgrams) against a prebuilt language model.
For English, the model is derived from the KJV Bible + Frankenstein corpus
(both available in ../misc/).  For French / German / Italian / Spanish,
prebuilt quadgram frequency files from ../misc/ are used.

Playfair variant
----------------
  Standard 25-letter alphabet: A-Z excluding J (I and J share one cell).
  Decryption rules:
    Same row    -> each letter moves one step LEFT  (wrapping, mod 5)
    Same column -> each letter moves one step UP    (wrapping, mod 5)
    Rectangle   -> letters swap column indices      (self-inverse)

Usage
-----
    python decrypt_stage06.py
"""

import math
import random
import re
import sys
from pathlib import Path

import numpy as np

# ---- File locations ----------------------------------------------------------
MISC_DIR    = Path("../misc")
CIPHER_FILE = Path("ciphertext_stage06.txt")
OUTPUT_FILE = Path("decrypted_stage06.txt")

# ---- Playfair 25-letter alphabet (I = J) -------------------------------------
PF_ALPHA = "ABCDEFGHIKLMNOPQRSTUVWXYZ"   # 25 letters, no J
PF_SET   = frozenset(PF_ALPHA)
# Map each Playfair letter to its index 0..24
ALPHA_IDX : dict[str, int] = {ch: i for i, ch in enumerate(PF_ALPHA)}

# ---- Solver hyper-parameters -------------------------------------------------
N_RESTARTS  = 100     # independent SA+HC runs per language scorer
SA_STEPS    = 5_000   # annealing iterations per restart
HC_MAX      = 20_000  # maximum hill-climbing steps per restart
HC_PATIENCE = 3_000   # HC stops early after this many non-improving steps
T_START     = 15.0    # initial SA temperature
T_END       = 0.10    # final SA temperature

# Known codewords from previous stages -- tried as Playfair keys first
KNOWN_KEYS = ["OTHELLO", "NEUTRON", "EQUATOR", "TRAJAN"]


# =============================================================================
# QUADGRAM SCORER  (string-based, used for loading; converted to fast table)
# =============================================================================

class QuadgramScorer:
    """
    Stores quadgram log10-probabilities in a Python dict.
    Used during resource loading; converted to FastScorer for SA/HC.
    """

    def __init__(self, name: str = "?"):
        self.name  = name
        self._data : dict[str, float] = {}
        self.floor : float = 0.0

    @classmethod
    def from_corpus(cls, paths: list[Path], name: str = "English") -> "QuadgramScorer":
        """
        Build scorer from raw plain-text corpus files.

        J is merged into I before counting to match Playfair decryption output
        (which never produces J).
        """
        obj    = cls(name=name)
        counts : dict[str, int] = {}
        total  = 0

        for path in paths:
            if not path.exists():
                print(f"  WARNING: corpus file not found: {path}", flush=True)
                continue
            text = path.read_text(encoding="utf-8", errors="ignore").upper()
            text = re.sub(r"[^A-Z]", "", text).replace("J", "I")
            for i in range(len(text) - 3):
                q         = text[i : i + 4]
                counts[q] = counts.get(q, 0) + 1
                total    += 1

        if total == 0:
            raise ValueError(f"No usable text found for scorer '{name}'")

        log_total = math.log10(total)
        obj._data = {q: math.log10(c) - log_total for q, c in counts.items()}
        obj.floor = math.log10(0.01) - log_total
        return obj

    @classmethod
    def from_file(cls, path: Path, name: str | None = None) -> "QuadgramScorer":
        """
        Load from a pre-built quadgram frequency file.

        Auto-detects format:
          Raw counts  : "ABCD 12345"      (all values >= 0)
          Log-probs   : "ABCD -8.1234"    (all values < 0)

        J->I normalisation is applied to all quadgram keys.
        """
        name = name or path.stem.replace("_", " ").title()
        obj  = cls(name=name)
        raw  : dict[str, float] = {}

        with path.open(encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                try:
                    quad = parts[0].upper().replace("J", "I")
                    val  = float(parts[1])
                    # Accumulate so J->I merging sums counts correctly
                    raw[quad] = raw.get(quad, 0.0) + val
                except ValueError:
                    continue

        if not raw:
            raise ValueError(f"Empty or unreadable quadgram file: {path}")

        sample = next(iter(raw.values()))
        if sample >= 0:
            # Raw counts -> log10 probabilities
            total     = sum(raw.values())
            log_total = math.log10(total)
            obj._data = {
                q: math.log10(max(v, 1)) - log_total
                for q, v in raw.items()
            }
            obj.floor = math.log10(0.01) - log_total
        else:
            obj._data = raw
            obj.floor = min(raw.values()) - 1.0

        return obj

    def score(self, text: str) -> float:
        """Return sum of log10-probabilities for all quadgrams in text."""
        data  = self._data
        floor = self.floor
        total = 0.0
        for i in range(len(text) - 3):
            total += data.get(text[i : i + 4], floor)
        return total


# =============================================================================
# FAST NUMPY SCORER  (20-40x faster than dict-based scoring)
# =============================================================================

class FastScorer:
    """
    Numpy-backed quadgram scorer.

    Stores scores in a flat float64 array of size 25^4 = 390,625, indexed by
      idx = a * 15625 + b * 625 + c * 25 + d
    where a, b, c, d are the 0..24 ALPHA_IDX values of each letter.
    Array lookups are ~10x faster than dict lookups, and the scoring loop
    can be vectorized over all quadgrams at once via NumPy indexing.
    """

    TABLE_SIZE = 25 ** 4  # 390,625

    def __init__(self, name: str, table: np.ndarray, floor: float):
        self.name  = name
        self.table = table    # shape (390625,), dtype float64
        self.floor = floor

    @classmethod
    def from_quadgram_scorer(cls, qs: QuadgramScorer) -> "FastScorer":
        """Convert a QuadgramScorer to a FastScorer."""
        table = np.full(cls.TABLE_SIZE, qs.floor, dtype=np.float64)
        for quad, score in qs._data.items():
            if len(quad) != 4:
                continue
            if not all(c in ALPHA_IDX for c in quad):
                continue   # skip quadgrams with letters outside PF_ALPHA
            a, b, c, d = (ALPHA_IDX[quad[0]], ALPHA_IDX[quad[1]],
                          ALPHA_IDX[quad[2]], ALPHA_IDX[quad[3]])
            table[a * 15625 + b * 625 + c * 25 + d] = score
        return cls(name=qs.name, table=table, floor=qs.floor)

    def score_array(self, pt: np.ndarray) -> float:
        """
        Score an integer plaintext array.

        pt -- 1-D int32 array of letter indices (0..24).
        Returns the sum of log10-probabilities over all quadgrams.
        """
        # Compute 4-gram indices in one vectorized expression
        idx = (pt[:-3].astype(np.int64) * 15625
               + pt[1:-2] * 625
               + pt[2:-1] * 25
               + pt[3:])
        return float(self.table[idx].sum())


# =============================================================================
# VECTORIZED PLAYFAIR DECRYPTION
# =============================================================================

def build_square(keyword: str) -> tuple[list[str], dict[str, tuple[int, int]]]:
    """
    Construct a 5x5 Playfair key square from keyword.

    Fills unused positions with the remaining alphabet letters in A-Z order.
    Returns (square, pos) where pos maps each letter to its (row, col).
    """
    seen   : set[str]  = set()
    square : list[str] = []

    for ch in keyword.upper().replace("J", "I"):
        if ch in PF_SET and ch not in seen:
            square.append(ch)
            seen.add(ch)

    for ch in PF_ALPHA:
        if ch not in seen:
            square.append(ch)

    pos = {ch: (i // 5, i % 5) for i, ch in enumerate(square)}
    return square, pos


def playfair_decrypt_str(ciphertext: str,
                         sq: list[str],
                         pos: dict[str, tuple[int, int]]) -> str:
    """
    Decrypt a full ciphertext using the given key square (string-based).
    Used for displaying human-readable results.
    """
    buf : list[str] = []
    app = buf.append
    for i in range(0, len(ciphertext), 2):
        a, b = ciphertext[i], ciphertext[i + 1]
        ra, ca = pos[a]
        rb, cb = pos[b]
        if ra == rb:
            app(sq[ra*5 + (ca - 1) % 5])
            app(sq[rb*5 + (cb - 1) % 5])
        elif ca == cb:
            app(sq[((ra - 1) % 5) * 5 + ca])
            app(sq[((rb - 1) % 5) * 5 + cb])
        else:
            app(sq[ra*5 + cb])
            app(sq[rb*5 + ca])
    return "".join(buf)


def numpy_eval(sq_int: np.ndarray,
               ct_ai: np.ndarray,
               ct_bi: np.ndarray,
               fast_scorer: FastScorer) -> float:
    """
    Vectorized Playfair decryption + scoring using NumPy.

    sq_int -- int32 array of shape (25,); sq_int[pos] = ALPHA_IDX of the
              letter at that position in the key square.
    ct_ai  -- int32 array of shape (n_pairs,): ALPHA_IDX of first  cipher letter.
    ct_bi  -- int32 array of shape (n_pairs,): ALPHA_IDX of second cipher letter.

    Returns the language-fitness score (float).
    """
    # Build row/col lookup: row_of[letter_idx] = row in key square (0..4)
    positions = np.arange(25, dtype=np.int32)
    row_of = np.empty(25, dtype=np.int32)
    col_of = np.empty(25, dtype=np.int32)
    row_of[sq_int] = positions // 5   # fancy-indexed assignment
    col_of[sq_int] = positions % 5

    ra = row_of[ct_ai]
    ca = col_of[ct_ai]
    rb = row_of[ct_bi]
    cb = col_of[ct_bi]

    same_row = ra == rb
    same_col = (~same_row) & (ca == cb)
    rect     = (~same_row) & (ca != cb)

    n  = len(ct_ai)
    pa = np.empty(n, dtype=np.int32)
    pb = np.empty(n, dtype=np.int32)

    # Rule 1: same row -- shift left
    pa[same_row] = sq_int[ra[same_row] * 5 + (ca[same_row] - 1) % 5]
    pb[same_row] = sq_int[rb[same_row] * 5 + (cb[same_row] - 1) % 5]

    # Rule 2: same column -- shift up
    pa[same_col] = sq_int[((ra[same_col] - 1) % 5) * 5 + ca[same_col]]
    pb[same_col] = sq_int[((rb[same_col] - 1) % 5) * 5 + cb[same_col]]

    # Rule 3: rectangle -- swap column indices
    pa[rect] = sq_int[ra[rect] * 5 + cb[rect]]
    pb[rect] = sq_int[rb[rect] * 5 + ca[rect]]

    # Interleave pa, pb -> full plaintext integer array of length 2*n
    pt      = np.empty(2 * n, dtype=np.int32)
    pt[::2] = pa
    pt[1::2] = pb

    return fast_scorer.score_array(pt)


# =============================================================================
# SIMULATED ANNEALING + HILL-CLIMBING
# =============================================================================

def _np_perturb(sq: np.ndarray) -> np.ndarray:
    """
    Return a perturbed copy of key square sq (int32 array of 25 elements).

    Six strategies with bias toward fine-grained cell swaps:
      50 %  Swap two randomly chosen cells
      10 %  Swap two randomly chosen rows
      10 %  Swap two randomly chosen columns
      10 %  Reverse the order of one row
      10 %  Reverse the order of one column
      10 %  Transpose the entire 5x5 grid
    """
    s = sq.copy()
    r = random.random()

    if r < 0.50:
        # Fine-grained: swap two arbitrary cells
        i, j = random.sample(range(25), 2)
        s[i], s[j] = s[j], s[i]

    elif r < 0.60:
        # Swap two rows
        r1, r2 = random.sample(range(5), 2)
        tmp = s[r1*5 : r1*5+5].copy()
        s[r1*5 : r1*5+5] = s[r2*5 : r2*5+5]
        s[r2*5 : r2*5+5] = tmp

    elif r < 0.70:
        # Swap two columns
        c1, c2 = random.sample(range(5), 2)
        tmp = s[c1::5].copy()
        s[c1::5] = s[c2::5]
        s[c2::5] = tmp

    elif r < 0.80:
        # Reverse one row
        rr = random.randint(0, 4)
        s[rr*5 : rr*5+5] = s[rr*5 : rr*5+5][::-1]

    elif r < 0.90:
        # Reverse one column
        c = random.randint(0, 4)
        s[c::5] = s[c::5][::-1]

    else:
        # Transpose: reflect across main diagonal
        s = s.reshape(5, 5).T.flatten()

    return s


def _sq_int_to_str(sq_int: np.ndarray) -> list[str]:
    """Convert integer key square array to list-of-chars representation."""
    return [PF_ALPHA[i] for i in sq_int]


def solve(ciphertext: str,
          scorer: QuadgramScorer,
          seed: int = 42) -> tuple[list[str], str, float]:
    """
    Search for the best Playfair key via N_RESTARTS independent SA+HC runs.

    Each restart:
      1. Start from a freshly randomised integer key square.
      2. Run SA_STEPS annealing iterations (temperature T_START -> T_END).
      3. Run HC_MAX hill-climbing iterations from the SA result, stopping
         early after HC_PATIENCE consecutive non-improving steps.

    Returns (best_square_chars, best_plaintext, best_score).
    """
    random.seed(seed)
    np.random.seed(seed)

    # Pre-compute exponential decay so T reaches T_END after SA_STEPS
    decay = (T_END / T_START) ** (1.0 / SA_STEPS)

    # Convert ciphertext to integer pair arrays (do this once)
    ct_ai = np.array([ALPHA_IDX[ciphertext[2*i    ]] for i in range(len(ciphertext)//2)], dtype=np.int32)
    ct_bi = np.array([ALPHA_IDX[ciphertext[2*i + 1]] for i in range(len(ciphertext)//2)], dtype=np.int32)

    # Build fast numpy scorer from the string-based scorer
    fast_scorer = FastScorer.from_quadgram_scorer(scorer)

    best_sq_int : np.ndarray = np.array([], dtype=np.int32)
    best_score  : float      = float("-inf")
    best_plain  : str        = ""

    for restart in range(N_RESTARTS):

        # ---- Phase 1: Simulated Annealing ------------------------------------
        cur_sq    = np.random.permutation(25).astype(np.int32)
        cur_score = numpy_eval(cur_sq, ct_ai, ct_bi, fast_scorer)
        loc_sq    = cur_sq.copy()
        loc_score = cur_score
        T         = T_START

        for _ in range(SA_STEPS):
            cand    = _np_perturb(cur_sq)
            c_score = numpy_eval(cand, ct_ai, ct_bi, fast_scorer)
            delta   = c_score - cur_score

            # Accept improvement unconditionally; accept worsening via Boltzmann
            if delta > 0 or random.random() < math.exp(delta / T):
                cur_sq, cur_score = cand, c_score

            if cur_score > loc_score:
                loc_sq, loc_score = cur_sq.copy(), cur_score

            T *= decay

        # ---- Phase 2: Hill-Climbing ------------------------------------------
        cur_sq, cur_score = loc_sq.copy(), loc_score
        stale = 0

        for _ in range(HC_MAX):
            cand    = _np_perturb(cur_sq)
            c_score = numpy_eval(cand, ct_ai, ct_bi, fast_scorer)

            if c_score > cur_score:
                cur_sq, cur_score = cand, c_score
                stale = 0
            else:
                stale += 1
                if stale >= HC_PATIENCE:
                    break   # converged locally

        if cur_score > loc_score:
            loc_sq, loc_score = cur_sq.copy(), cur_score

        # ---- Update global best ----------------------------------------------
        if loc_score > best_score:
            best_sq_int = loc_sq.copy()
            best_score  = loc_score
            # Reconstruct readable plaintext via string-based decrypt
            sq_chars = _sq_int_to_str(best_sq_int)
            pos      = {ch: (i // 5, i % 5) for i, ch in enumerate(sq_chars)}
            best_plain = playfair_decrypt_str(ciphertext, sq_chars, pos)
            print(
                f"    restart {restart + 1:>3}/{N_RESTARTS}"
                f"  score={best_score:9.1f}"
                f"  {best_plain[:60]!r}",
                flush=True,
            )

    return _sq_int_to_str(best_sq_int), best_plain, best_score


# =============================================================================
# HELPERS
# =============================================================================

def load_ciphertext() -> str:
    """
    Read, uppercase, and clean the ciphertext.
    Strips non-alpha characters; maps J->I to match Playfair's merged I/J cell.
    Raises ValueError if the result has odd length.
    """
    raw     = CIPHER_FILE.read_text(encoding="utf-8", errors="ignore")
    cleaned = re.sub(r"[^A-Za-z]", "", raw).upper().replace("J", "I")
    if len(cleaned) % 2:
        raise ValueError(
            f"Ciphertext has odd length ({len(cleaned)}); "
            "Playfair requires an even number of characters."
        )
    return cleaned


def load_scorers() -> list[QuadgramScorer]:
    """Load all available language scorers from ../misc/."""
    scorers : list[QuadgramScorer] = []

    # English -- build from raw corpus
    corpus_paths = [MISC_DIR / "frankenstein.txt", MISC_DIR / "kjv.txt"]
    available    = [p for p in corpus_paths if p.exists()]
    if available:
        print(
            f"  English   -- building quadgrams from: {[p.name for p in available]}",
            flush=True,
        )
        scorers.append(QuadgramScorer.from_corpus(available, "English"))
    else:
        print("  English   -- WARNING: no corpus files found in misc/")

    # Other languages -- from pre-built frequency files
    lang_map = {
        "French" : MISC_DIR / "french_quadgrams.txt",
        "German" : MISC_DIR / "german_quadgrams.txt",
        "Italian": MISC_DIR / "italian_quadgrams.txt",
        "Spanish": MISC_DIR / "spanish_quadgrams.txt",
    }
    for lang, path in lang_map.items():
        if path.exists():
            print(f"  {lang:<9} -- loading from {path.name}", flush=True)
            scorers.append(QuadgramScorer.from_file(path, lang))
        else:
            print(f"  {lang:<9} -- not found ({path.name})")

    return scorers


def print_square(sq: list[str]) -> None:
    """Pretty-print a 5x5 Playfair key square to stdout."""
    for r in range(5):
        print("    " + "  ".join(sq[r*5 : r*5 + 5]))


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("=" * 65)
    print("  Stage 06 -- Playfair Cipher Solver")
    print("=" * 65)

    # Load ciphertext
    print(f"\nLoading ciphertext from {CIPHER_FILE} ...")
    ciphertext = load_ciphertext()
    print(f"  {len(ciphertext)} characters, {len(ciphertext) // 2} digraphs")

    # Load language scorers
    print("\nLoading language scorers ...")
    scorers = load_scorers()
    if not scorers:
        sys.exit("ERROR: Could not load any language scorer.")

    # Quick sanity check: try previous-stage codewords as Playfair keys
    print("\nTrying known codewords as Playfair key ...")
    eng_scorer = scorers[0]
    for kw in KNOWN_KEYS:
        sq, pos = build_square(kw)
        pt      = playfair_decrypt_str(ciphertext, sq, pos)
        s       = eng_scorer.score(pt)
        print(f"  {kw:<10} -> {pt[:55]!r}  (score={s:.1f})")

    # Full SA+HC search for each language scorer
    all_results : list[tuple[str, list[str], str, float]] = []

    for scorer in scorers:
        print(f"\n{'-' * 65}")
        print(f"  Solving with {scorer.name} scorer ({N_RESTARTS} restarts) ...")
        print(f"{'-' * 65}")
        sq, plain, score = solve(ciphertext, scorer)
        all_results.append((scorer.name, sq, plain, score))

    # Select globally best result
    all_results.sort(key=lambda x: x[3], reverse=True)
    best_lang, best_sq, best_plain, best_score = all_results[0]

    print(f"\n{'=' * 65}")
    print(f"  BEST RESULT  (scorer: {best_lang},  score: {best_score:.1f})")
    print(f"{'=' * 65}")

    print("\nKey square:")
    print_square(best_sq)

    print("\nDecrypted plaintext:")
    for i in range(0, len(best_plain), 60):
        print("  " + best_plain[i : i + 60])

    # Save output to file
    with OUTPUT_FILE.open("w", encoding="utf-8") as fh:
        fh.write(f"Solver        : Playfair SA+HC (numpy-vectorized)\n")
        fh.write(f"Language      : {best_lang}\n")
        fh.write(f"Fitness score : {best_score:.1f}\n\n")
        fh.write("Key square:\n")
        for r in range(5):
            fh.write("  " + "  ".join(best_sq[r*5 : r*5 + 5]) + "\n")
        fh.write("\nPlaintext:\n")
        for i in range(0, len(best_plain), 60):
            fh.write(best_plain[i : i + 60] + "\n")

    print(f"\nOutput written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
