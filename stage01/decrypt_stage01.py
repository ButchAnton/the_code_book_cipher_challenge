#!/usr/bin/env python3
"""
Stage 01 — Monoalphabetic Substitution Cipher Solver
=====================================================

Per the README: "This is a simple monoalphabetic substitution cipher."

Each letter in the ciphertext maps to exactly one plaintext letter; all
other characters (spaces, punctuation, newlines) are left unchanged.

Algorithm
---------
1.  Build quadgram log₁₀-probability statistics from ../misc/kjv.txt.
    The KJV Bible is an ideal reference corpus for Biblical English text.

2.  Pre-extract only the alphabetic characters from the ciphertext into
    a single string (`cipher_alpha`).  Applying a candidate key and
    scoring it never has to touch spaces or punctuation.

3.  Generate a frequency-analysis starting key:
      - Count how often each cipher letter appears.
      - Map the most-frequent cipher letter to 'E', second to 'T', etc.
    This is vastly better than a random start; it usually produces the
    correct key in a single hill-climbing run.

4.  Refine the key by hill-climbing:
      - Try all C(26,2) = 325 pairwise letter swaps.
      - Keep any swap that strictly increases the quadgram score
        (first-improvement greedy).
      - Repeat until no swap helps (local optimum reached).

5.  Restart with fresh random keys and keep the global best.

6.  Reconstruct the full plaintext (preserving punctuation/spaces) and
    report the embedded codeword.

Performance notes (pure Python, no third-party dependencies)
-------------------------------------------------------------
* Quadgram scores are stored as a plain dict {4-char string: float}.
  Python's str hash is highly optimised; 4-char strings are tiny and
  cache-friendly.

* Applying a key uses str.maketrans / str.translate — both are
  implemented in C and run far faster than a Python loop.

* The alpha-only ciphertext is computed once; subsequent score() calls
  work entirely on this shorter string (~1,000 chars vs ~1,500 with
  spaces/punctuation).
"""

import math
import os
import random
import string
import sys
from collections import Counter

# Force UTF-8 output on Windows consoles that default to cp1252.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
CIPHER_FILE = os.path.join(SCRIPT_DIR, "ciphertext_stage01.txt")
CORPUS_FILE = os.path.join(SCRIPT_DIR, "..", "misc", "kjv.txt")

ALPHABET     = string.ascii_uppercase
N_ALPHA      = 26
ORD_A        = ord('A')

NUM_RESTARTS = 20       # random-restart hill climbs; freq-init usually wins on restart 0
RANDOM_SEED  = 42       # fixed seed → reproducible output


# ---------------------------------------------------------------------------
# 1.  Build quadgram score table from the KJV Bible corpus
# ---------------------------------------------------------------------------

class QuadgramScorer:
    """Score a piece of text using log₁₀ quadgram probabilities.

    A higher (less-negative) score means the text looks more like the
    corpus language.  Unseen quadgrams receive a small floor probability
    to avoid -∞ and to penalise (but not catastrophically) rare letter
    combinations.

    Attributes
    ----------
    scores : dict[str, float]
        Maps every 4-char uppercase string seen in the corpus to its
        log₁₀ probability.
    floor : float
        log₁₀ probability used for quadgrams absent from the corpus.
    """

    def __init__(self, corpus_path: str) -> None:
        # Read entire corpus, strip every non-alpha character so that
        # quadgrams are never split by spaces, punctuation, or newlines.
        with open(corpus_path, encoding="utf-8", errors="ignore") as fh:
            text = "".join(c for c in fh.read().upper() if c.isalpha())

        n = len(text)
        print(f"  Corpus: {n:,} alphabetic characters")

        # Count every overlapping 4-gram.
        counts = Counter(text[i : i + 4] for i in range(n - 3))
        total  = sum(counts.values())

        # Floor probability for quadgrams not present in the corpus.
        # Using 0.01 counts (rather than 0) keeps the score finite while
        # strongly penalizing unnatural letter sequences.
        self.floor  = math.log10(0.01 / total)
        self.scores = {quad: math.log10(count / total)
                       for quad, count in counts.items()}

    def score(self, alpha_text: str) -> float:
        """Return total log₁₀ probability for *alpha_text* (letters only).

        The caller is responsible for passing a string that contains only
        A-Z characters.  No filtering is done here so this function can
        be called as tightly as possible in the hill-climbing inner loop.
        """
        sc  = self.scores   # local alias avoids attribute lookup in loop
        fl  = self.floor
        n   = len(alpha_text)
        return sum(sc.get(alpha_text[i : i + 4], fl) for i in range(n - 3))


# ---------------------------------------------------------------------------
# 2.  Key application helpers
# ---------------------------------------------------------------------------

def make_translate_table(key: list[int]) -> dict:
    """Build a str.translate mapping from *key*.

    key[i] = plain-letter index (0–25) for cipher letter i (0–25).
    Returns a translation table suitable for str.translate().
    """
    plain_str = "".join(ALPHABET[k] for k in key)
    return str.maketrans(ALPHABET, plain_str)


def build_plaintext(ciphertext: str, key: list[int]) -> str:
    """Reconstruct the full plaintext (spaces/punctuation preserved)."""
    return ciphertext.translate(make_translate_table(key))


# ---------------------------------------------------------------------------
# 3.  Frequency-analysis initial key
# ---------------------------------------------------------------------------

# English letter frequency order, most-to-least common.
_ENG_FREQ_ORDER = "ETAOINSHRDLCUMWFGYPBVKJXQZ"


def frequency_initial_key(cipher_alpha: str) -> list[int]:
    """Return a starting key based on letter-frequency matching.

    The most-frequent cipher letter is mapped to 'E', the second
    most-frequent to 'T', and so on.  For clear, sufficiently long
    English text this guess is usually close enough that a single
    hill-climb pass reaches the correct key.
    """
    counts = Counter(cipher_alpha)

    # Order cipher letters by descending frequency.
    cipher_by_freq = [ch for ch, _ in counts.most_common()]
    # Append any letters that never appeared, so we always have 26 entries.
    for ch in ALPHABET:
        if ch not in cipher_by_freq:
            cipher_by_freq.append(ch)

    # Map each cipher letter to the English letter of the same frequency rank.
    key = [0] * N_ALPHA
    for rank, ch in enumerate(cipher_by_freq):
        key[ord(ch) - ORD_A] = ord(_ENG_FREQ_ORDER[rank]) - ORD_A
    return key


# ---------------------------------------------------------------------------
# 4.  Hill-climbing
# ---------------------------------------------------------------------------

def hill_climb_once(cipher_alpha: str,
                    key: list[int],
                    scorer: QuadgramScorer) -> tuple[list[int], float]:
    """Run one hill-climb from *key* and return (local_best_key, score).

    Each pass tries all 325 pairwise key-position swaps.  Any swap that
    strictly raises the score is applied immediately (first-improvement
    greedy).  The pass repeats from the beginning until a full pass
    yields no improvement (local optimum).

    The translate table is rebuilt only when a swap is kept, which
    happens infrequently compared to the number of swaps tried.
    """
    key   = key[:]                                           # own copy
    table = make_translate_table(key)
    plain = cipher_alpha.translate(table)
    score = scorer.score(plain)

    improved = True
    while improved:
        improved = False
        for i in range(N_ALPHA):
            for j in range(i + 1, N_ALPHA):
                # Tentatively swap key positions i and j.
                key[i], key[j] = key[j], key[i]

                new_table = make_translate_table(key)
                new_plain = cipher_alpha.translate(new_table)
                new_score = scorer.score(new_plain)

                if new_score > score:
                    # Improvement — keep the swap and update state.
                    table    = new_table
                    plain    = new_plain
                    score    = new_score
                    improved = True
                else:
                    # No improvement — revert.
                    key[i], key[j] = key[j], key[i]

    return key, score


def solve(ciphertext: str,
          scorer: QuadgramScorer,
          num_restarts: int = NUM_RESTARTS,
          seed: int = RANDOM_SEED) -> tuple[str, list[int], float]:
    """Find the best substitution key via repeated hill-climbing.

    Restart 0 uses the frequency-analysis initial key; subsequent
    restarts use randomly shuffled keys to escape local optima.

    Returns (plaintext, best_key, best_score).
    """
    rng = random.Random(seed)

    # Pre-extract alpha-only ciphertext string (computed once).
    cipher_alpha = "".join(c for c in ciphertext if c.isalpha())

    best_key   = list(range(N_ALPHA))
    best_score = float("-inf")

    for restart in range(num_restarts):
        if restart == 0:
            init_key = frequency_initial_key(cipher_alpha)
            label    = "freq-init "
        else:
            init_key = list(range(N_ALPHA))
            rng.shuffle(init_key)
            label    = f"restart {restart:2d}"

        key, score = hill_climb_once(cipher_alpha, init_key, scorer)

        if score > best_score:
            best_score = score
            best_key   = key[:]
            print(f"  [{label}]  new best score = {score:.2f}")

    best_plain = build_plaintext(ciphertext, best_key)
    return best_plain, best_key, best_score


# ---------------------------------------------------------------------------
# 5.  Main
# ---------------------------------------------------------------------------

def main() -> None:
    # --- Load ciphertext ---------------------------------------------------
    with open(CIPHER_FILE) as fh:
        ciphertext = fh.read().upper()

    print("=" * 60)
    print("Stage 01 — Monoalphabetic Substitution Cipher Solver")
    print("=" * 60)
    print("\nCiphertext:\n")
    print(ciphertext)

    # --- Build quadgram scorer from KJV Bible corpus ----------------------
    print(f"Building quadgram statistics from KJV Bible …")
    scorer = QuadgramScorer(CORPUS_FILE)
    print(f"  {len(scorer.scores):,} unique quadgrams indexed\n")

    # --- Solve ------------------------------------------------------------
    print(f"Running hill-climbing solver ({NUM_RESTARTS} restarts) …")
    plaintext, key, score = solve(ciphertext, scorer)

    # --- Print substitution key -------------------------------------------
    print("\n" + "=" * 60)
    print("Substitution key (cipher letter → plaintext letter)")
    print("=" * 60)
    key_rows = [f"{ALPHABET[i]}→{ALPHABET[key[i]]}" for i in range(N_ALPHA)]
    for r in range(0, N_ALPHA, 6):
        print("  " + "   ".join(key_rows[r : r + 6]))

    # --- Print decrypted plaintext ----------------------------------------
    print("\n" + "=" * 60)
    print("Decrypted Plaintext")
    print("=" * 60)
    print(plaintext)

    # --- Extract codeword -------------------------------------------------
    # The last line of the plaintext explicitly identifies the codeword.
    lines    = [ln.strip() for ln in plaintext.strip().splitlines() if ln.strip()]
    last     = lines[-1]
    # Strip trailing punctuation from the final word.
    codeword = last.split()[-1].strip(".,;:!?'\"")

    print("\n" + "=" * 60)
    print(f"Last line : {last}")
    print(f"CODEWORD  : {codeword}")
    print("=" * 60)


if __name__ == "__main__":
    main()
