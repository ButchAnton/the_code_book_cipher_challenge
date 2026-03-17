"""
decrypt_stage02.py — Caesar cipher solver for stage 02 of the Cipher Challenge.

Approach:
  1. Try all 26 possible shift values (brute force, feasible for a single-key
     Caesar cipher).
  2. Score each candidate plaintext by counting how many of its uppercase words
     appear in the Latin vocabulary list (../misc/latin_words.txt).  Latin is
     the expected language because the hint says "Caesar shift cipher" — the
     most famous historical cipher, invented in Rome.
  3. The shift with the highest word-match score is selected as the plaintext.
  4. The codeword is extracted from the phrase "DICTUM ARCANUM EST <CODEWORD>",
     which literally translates as "The secret motto is <CODEWORD>".

Usage:
  python decrypt_stage02.py
"""

import os

# ---------------------------------------------------------------------------
# Paths (all relative to this script's directory)
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
CIPHER_FILE  = os.path.join(SCRIPT_DIR, "ciphertext_stage02.txt")
WORDLIST     = os.path.join(SCRIPT_DIR, "..", "misc", "latin_words.txt")


def load_ciphertext(path: str) -> str:
    """Read and return the ciphertext as a single whitespace-normalized string."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = fh.read()
    # Collapse all whitespace (spaces, newlines) into single spaces; strip ends.
    return " ".join(raw.split())


def load_latin_wordset(path: str) -> set[str]:
    """Return a set of uppercase Latin words from the vocabulary file."""
    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()
    # Uppercase for case-insensitive comparison; skip blank lines and
    # entries starting with '-' (those are suffixes, not standalone words).
    return {
        line.strip().upper()
        for line in lines
        if line.strip() and not line.strip().startswith("-")
    }


def caesar_decrypt(ciphertext: str, shift: int) -> str:
    """
    Shift every uppercase letter backward by `shift` positions (mod 26).
    Non-alpha characters (spaces, punctuation) are preserved unchanged.
    """
    result = []
    for ch in ciphertext:
        if ch.isalpha() and ch.isupper():
            result.append(chr((ord(ch) - ord("A") - shift) % 26 + ord("A")))
        else:
            result.append(ch)
    return "".join(result)


def score_against_wordlist(plaintext: str, wordset: set[str]) -> int:
    """
    Count how many whitespace-delimited tokens in `plaintext` are found
    in `wordset`.  Higher is better.
    """
    return sum(1 for token in plaintext.split() if token in wordset)


def find_codeword(plaintext: str) -> str | None:
    """
    Extract the codeword from the canonical disclosure phrase:
        DICTUM ARCANUM EST <CODEWORD>
    Returns the codeword string, or None if the phrase is absent.
    """
    tokens = plaintext.split()
    for i, token in enumerate(tokens):
        # Look for the three-word preamble followed by the answer.
        if (
            token == "DICTUM"
            and i + 3 < len(tokens)
            and tokens[i + 1] == "ARCANUM"
            and tokens[i + 2] == "EST"
        ):
            return tokens[i + 3]
    return None


def main() -> None:
    ciphertext = load_ciphertext(CIPHER_FILE)
    latin_words = load_latin_wordset(WORDLIST)

    print("=" * 60)
    print("Stage 02 — Caesar Cipher Brute Force")
    print("=" * 60)
    print(f"Ciphertext : {ciphertext}\n")

    # ------------------------------------------------------------------
    # Score every possible shift and track the best.
    # ------------------------------------------------------------------
    best_shift  = 0
    best_score  = -1
    best_plain  = ""

    for shift in range(26):
        candidate = caesar_decrypt(ciphertext, shift)
        score     = score_against_wordlist(candidate, latin_words)
        if score > best_score:
            best_score = score
            best_shift = shift
            best_plain = candidate

    # ------------------------------------------------------------------
    # Report results.
    # ------------------------------------------------------------------
    print(f"Best shift : {best_shift}  (score: {best_score} Latin word matches)")
    print(f"Plaintext  : {best_plain}\n")

    # Show the full decryption table for transparency.
    print("All shifts (shift | score | plaintext):")
    print("-" * 60)
    for shift in range(26):
        pt    = caesar_decrypt(ciphertext, shift)
        score = score_against_wordlist(pt, latin_words)
        marker = " <-- BEST" if shift == best_shift else ""
        print(f"  {shift:2d} | {score:2d} | {pt}{marker}")

    # ------------------------------------------------------------------
    # Extract and display the codeword.
    # ------------------------------------------------------------------
    print()
    codeword = find_codeword(best_plain)
    if codeword:
        print(f"Codeword   : {codeword}")
    else:
        print("Codeword   : (not found — review plaintext manually)")


if __name__ == "__main__":
    main()
