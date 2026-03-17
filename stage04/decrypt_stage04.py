#!/usr/bin/env python3
"""
Stage 04: Vigenere Cipher Decryption
=====================================
README hint: "This is a Vigenere cipher."

Strategy
--------
 1.  Strip the ciphertext to uppercase letters only (discard spaces/newlines).
 2.  Build an English letter-frequency table and quadgram table from kjv.txt
     and frankenstein.txt (both available in ../misc/).
 3.  Derive letter-frequency tables for Italian, French, German, and Spanish
     from their precomputed quadgram files in ../misc/ (weighted letter counts
     across all quadgrams serve as a good unigram approximation).
 4.  Load the precomputed quadgram scoring tables for those four languages.
 5.  Run Index of Coincidence (IOC) analysis for key lengths 1-30.  The true
     key length produces a mean IOC close to the target language's natural
     IOC value (~0.065 for English, ~0.074 for Italian, ~0.038 for random).
 6.  Run a Kasiski test (repeated trigram spacings) for an independent
     key-length estimate.
 7.  For each candidate key length and each language frequency table, use
     Pearson Chi-squared minimization on every individual Caesar stream
     (one per key position) to recover the Vigenere key one character at a time.
 8.  Additionally try each previous-stage codeword as a trial key
     (OTHELLO, NEUTRON, EQUATOR), since competition puzzles sometimes chain
     stage solutions together.
 9.  Score every candidate plaintext with all available quadgram tables;
     the highest-scoring result is declared best.
10.  Write the best result to decrypted_stage04.txt and print a summary.
"""

import re
import math
import collections
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
STAGE_DIR = Path(__file__).resolve().parent
MISC_DIR  = STAGE_DIR.parent / 'misc'

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Previous-stage codewords to trial as Vigenere keys (all length 7 — worth testing)
TRIAL_KEYS = ['OTHELLO', 'NEUTRON', 'EQUATOR']

# Reference Index of Coincidence values for natural languages
LANG_IOC: dict[str, float] = {
    'english': 0.0655,
    'italian': 0.0738,
    'french':  0.0778,
    'german':  0.0762,
    'spanish': 0.0770,
}
RANDOM_IOC = 0.0385   # IOC for uniformly random letters

# ---------------------------------------------------------------------------
# Ciphertext
# ---------------------------------------------------------------------------

def load_ciphertext() -> str:
    """Return ciphertext as a string of uppercase A-Z letters (spaces stripped)."""
    raw = (STAGE_DIR / 'ciphertext_stage04.txt').read_text()
    return re.sub(r'[^A-Za-z]', '', raw).upper()


# ---------------------------------------------------------------------------
# Letter-frequency tables
# ---------------------------------------------------------------------------

def build_letter_freq_from_text(*paths: Path) -> dict[str, float]:
    """
    Count every A-Z character across one or more plaintext corpus files and
    return a dict mapping each letter to its relative frequency (sums to 1.0).
    """
    counts: collections.Counter = collections.Counter()
    for path in paths:
        for c in path.read_text(errors='ignore').upper():
            if 'A' <= c <= 'Z':
                counts[c] += 1
    total = sum(counts.values())
    return {c: counts.get(c, 0) / total for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'}


def build_letter_freq_from_quadgrams(lang: str) -> dict[str, float]:
    """
    Derive a letter-frequency table from a language's quadgram file by summing
    every character occurrence in every quadgram, weighted by the quadgram's count.
    This gives a good unigram approximation without a separate plaintext corpus.
    """
    counts: collections.Counter = collections.Counter()
    with (MISC_DIR / f'{lang}_quadgrams.txt').open(errors='ignore') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            weight = int(parts[1])
            for c in parts[0].upper():
                if 'A' <= c <= 'Z':
                    counts[c] += weight
    total = sum(counts.values())
    return {c: counts.get(c, 0) / total for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'}


# ---------------------------------------------------------------------------
# Quadgram scoring tables
# ---------------------------------------------------------------------------

def build_quadgrams_from_text(*paths: Path) -> tuple[dict[str, float], float]:
    """
    Build a quadgram log-probability table from corpus files on the fly.
    Non-letter characters are removed so consecutive words contribute
    cross-boundary quadgrams (matching how Vigenere strips punctuation).
    Returns (qgrams: dict[quadgram -> log10_prob], floor_log_prob).
    """
    counts: collections.Counter = collections.Counter()
    for path in paths:
        text = re.sub(r'[^A-Za-z]', '', path.read_text(errors='ignore')).upper()
        for i in range(len(text) - 3):
            counts[text[i:i + 4]] += 1
    total = sum(counts.values())
    floor = math.log10(0.01 / total)          # assigned to unseen quadgrams
    qgrams = {qg: math.log10(c / total) for qg, c in counts.items()}
    return qgrams, floor


def load_quadgrams(lang: str) -> tuple[dict[str, float], float] | None:
    """
    Load a precomputed quadgram file (format: "ABCD count" per line).
    Returns (qgrams, floor_log_prob) or None if the file does not exist.
    """
    path = MISC_DIR / f'{lang}_quadgrams.txt'
    if not path.exists():
        return None
    counts: dict[str, int] = {}
    with path.open(errors='ignore') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                counts[parts[0].upper()] = int(parts[1])
    total = sum(counts.values())
    floor = math.log10(0.01 / total)
    return {qg: math.log10(c / total) for qg, c in counts.items()}, floor


def quadgram_score(text: str, qg_data: tuple[dict, float]) -> float:
    """
    Sum of log-probabilities over every overlapping quadgram in text.
    Higher score = better fit to the target language.
    """
    qgrams, floor = qg_data
    return sum(qgrams.get(text[i:i + 4], floor) for i in range(len(text) - 3))


# ---------------------------------------------------------------------------
# Index of Coincidence
# ---------------------------------------------------------------------------

def index_of_coincidence(text: str) -> float:
    """
    Friedman (1922) Index of Coincidence:
        IC = sum_i [ n_i * (n_i - 1) ] / [ N * (N - 1) ]
    where n_i = count of letter i, N = total letter count.
    English plain text: IC ~ 0.065.  Uniform random text: IC ~ 0.038.
    """
    n = len(text)
    if n < 2:
        return 0.0
    counts = collections.Counter(text)
    return sum(c * (c - 1) for c in counts.values()) / (n * (n - 1))


def ioc_analysis(ciphertext: str, max_kl: int = 30) -> list[tuple[int, float]]:
    """
    For each key length kl in 1..max_kl, split the ciphertext into kl
    independent streams (positions 0, kl, 2kl, ... and 1, kl+1, ... etc.)
    and compute the arithmetic mean IOC across all streams.

    The true key length — and its multiples — will produce high mean IOC;
    other lengths give IOC close to random.

    Returns list of (kl, avg_ioc) sorted by avg_ioc descending.
    """
    return sorted(
        [
            (kl,
             sum(index_of_coincidence(ciphertext[i::kl]) for i in range(kl)) / kl)
            for kl in range(1, max_kl + 1)
        ],
        key=lambda t: -t[1],
    )


# ---------------------------------------------------------------------------
# Kasiski test
# ---------------------------------------------------------------------------

def kasiski_analysis(ciphertext: str, substr_len: int = 3,
                     max_kl: int = 30) -> collections.Counter:
    """
    Kasiski (1863) test:
    Repeated substrings of length >= 3 in Vigenere ciphertext tend to be
    separated by a distance that is a multiple of the key length.  We find all
    such repeated trigrams, compute pairwise spacings between consecutive
    occurrences, and accumulate the factors of each spacing (up to max_kl).

    Returns a Counter mapping factor -> vote_count (higher = stronger evidence
    for that key length).
    """
    positions: dict[str, list[int]] = collections.defaultdict(list)
    n = len(ciphertext)
    for i in range(n - substr_len + 1):
        positions[ciphertext[i:i + substr_len]].append(i)

    factor_counts: collections.Counter = collections.Counter()
    for pos_list in positions.values():
        if len(pos_list) < 2:
            continue
        for j in range(len(pos_list) - 1):
            spacing = pos_list[j + 1] - pos_list[j]
            for f in range(2, min(spacing + 1, max_kl + 1)):
                if spacing % f == 0:
                    factor_counts[f] += 1
    return factor_counts


# ---------------------------------------------------------------------------
# Vigenere key recovery via Chi-squared
# ---------------------------------------------------------------------------

def chi_squared(text: str, freq: dict[str, float]) -> float:
    """
    Pearson's Chi-squared goodness-of-fit statistic:
        X^2 = sum_c [ (observed_c - expected_c)^2 / expected_c ]
    Compares observed letter distribution in text against expected frequency
    table freq.  Lower value = better fit to the target language.
    """
    n = len(text)
    if n == 0:
        return float('inf')
    counts = collections.Counter(text)
    return sum(
        (counts.get(c, 0) - freq[c] * n) ** 2 / (freq[c] * n)
        for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        if freq.get(c, 0) > 0
    )


def break_caesar(stream: str, freq: dict[str, float]) -> tuple[int, str, float]:
    """
    Try all 26 Caesar shifts on a single Vigenere stream and return the shift
    that minimizes Chi-squared against the expected letter frequencies.

    Returns (shift_value: int, key_character: str, chi2: float).
    The shift_value is the key character's ordinal (A=0, B=1, ..., Z=25).
    """
    best_s, best_k, best_chi2 = 0, 'A', float('inf')
    for s in range(26):
        # Reverse the Caesar shift: candidate plaintext = stream shifted back by s
        decrypted = ''.join(
            chr((ord(c) - ord('A') - s) % 26 + ord('A')) for c in stream
        )
        chi2 = chi_squared(decrypted, freq)
        if chi2 < best_chi2:
            best_s, best_k, best_chi2 = s, chr(s + ord('A')), chi2
    return best_s, best_k, best_chi2


def find_key(ciphertext: str, kl: int, freq: dict[str, float]) -> str:
    """
    Recover the Vigenere key of length kl by applying Chi-squared minimization
    independently to each of the kl Caesar streams.
    """
    return ''.join(
        break_caesar(ciphertext[i::kl], freq)[1]
        for i in range(kl)
    )


def decrypt_vigenere(ciphertext: str, key: str) -> str:
    """
    Standard Vigenere decryption:
        P[i] = (C[i] - K[i mod |K|]) mod 26
    All characters are uppercase A-Z (A=0, Z=25).
    """
    kl = len(key)
    return ''.join(
        chr((ord(c) - ord(key[i % kl])) % 26 + ord('A'))
        for i, c in enumerate(ciphertext)
    )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def fmt_blocks(text: str, block: int = 5, per_line: int = 10) -> str:
    """Format text as space-separated groups of `block` letters, `per_line` groups per row."""
    chunks = [text[i:i + block] for i in range(0, len(text), block)]
    rows   = [' '.join(chunks[j:j + per_line]) for j in range(0, len(chunks), per_line)]
    return '\n'.join(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    hr = '=' * 70
    print(hr)
    print('Stage 04: Vigenere Cipher Decryption')
    print(hr)

    # -- Load ciphertext -------------------------------------------------- #
    cipher = load_ciphertext()
    print(f'\nCiphertext length : {len(cipher)} letters')
    print(f'First 60 chars    : {cipher[:60]}')

    # -- Build letter-frequency tables ------------------------------------ #
    print('\nBuilding letter-frequency tables ...')
    freq_tables: dict[str, dict] = {}

    freq_tables['english'] = build_letter_freq_from_text(
        MISC_DIR / 'kjv.txt',
        MISC_DIR / 'frankenstein.txt',
    )
    print('  english  : built from kjv.txt + frankenstein.txt')

    for lang in ('italian', 'french', 'german', 'spanish'):
        qg_path = MISC_DIR / f'{lang}_quadgrams.txt'
        if qg_path.exists():
            freq_tables[lang] = build_letter_freq_from_quadgrams(lang)
            print(f'  {lang:<10}: derived from {lang}_quadgrams.txt')

    # -- Build/load quadgram scoring tables ------------------------------- #
    print('\nBuilding/loading quadgram scoring tables ...')
    qg_tables: dict[str, tuple] = {}

    print('  english  : building from kjv.txt + frankenstein.txt (may take a moment) ...')
    qg_tables['english'] = build_quadgrams_from_text(
        MISC_DIR / 'kjv.txt',
        MISC_DIR / 'frankenstein.txt',
    )
    print(f'             {len(qg_tables["english"][0])} unique quadgrams indexed')

    for lang in ('italian', 'french', 'german', 'spanish'):
        data = load_quadgrams(lang)
        if data:
            qg_tables[lang] = data
            print(f'  {lang:<10}: {len(data[0])} quadgrams')

    # -- IOC analysis ----------------------------------------------------- #
    print('\n' + '-' * 60)
    print('Index of Coincidence analysis  (key lengths 1-30)')
    print('-' * 60)
    ioc_list = ioc_analysis(cipher, max_kl=30)
    ioc_dict  = dict(ioc_list)
    print('  Reference: English=0.0655  Italian=0.0738  Random=0.0385\n')
    for kl, ioc_val in ioc_list[:15]:
        bar = '#' * int(ioc_val * 500)
        print(f'  kl={kl:2d}  IOC={ioc_val:.4f}  {bar}')

    # -- Kasiski test ----------------------------------------------------- #
    print('\n' + '-' * 60)
    print('Kasiski test  (trigram spacing factors, key lengths 1-30)')
    print('-' * 60)
    kasiski = kasiski_analysis(cipher, substr_len=3, max_kl=30)
    for kl, cnt in kasiski.most_common(15):
        print(f'  kl={kl:2d}  factor votes={cnt}')

    # -- Select candidate key lengths ------------------------------------- #
    # Accept any key length whose mean IOC exceeds the midpoint between the
    # random baseline and the English IOC (the lowest natural-language value
    # we test).  This avoids committing to a single language assumption.
    threshold = RANDOM_IOC + 0.5 * (LANG_IOC['english'] - RANDOM_IOC)
    candidates = sorted({kl for kl, v in ioc_list if v >= threshold})
    if not candidates:
        # Fall back to the top 10 IOC candidates if nothing clears the threshold
        candidates = [kl for kl, _ in ioc_list[:10]]
    print(f'\nCandidates (IOC >= {threshold:.4f}): {candidates}')

    # -- Frequency attack on all candidate key lengths -------------------- #
    print('\n' + '-' * 60)
    print('Running frequency attack ...')
    print('-' * 60)

    all_results: list[dict] = []

    def score_and_store(kl: int, source: str, key: str, plaintext: str) -> None:
        """Compute quadgram scores for plaintext and append a result record."""
        scores = {
            lang: quadgram_score(plaintext, qd)
            for lang, qd in qg_tables.items()
        }
        best = max(scores.values()) if scores else float('-inf')
        all_results.append({
            'kl':     kl,
            'source': source,
            'key':    key,
            'plain':  plaintext,
            'ioc':    ioc_dict.get(kl, 0.0),
            'scores': scores,
            'best':   best,
        })

    # For each candidate key length, attack using every language frequency table
    for kl in candidates[:15]:
        for lang, freq in freq_tables.items():
            key   = find_key(cipher, kl, freq)
            plain = decrypt_vigenere(cipher, key)
            score_and_store(kl, f'freq/{lang}', key, plain)

    # Also trial previous-stage codewords as literal Vigenere keys
    for trial_key in TRIAL_KEYS:
        plain = decrypt_vigenere(cipher, trial_key)
        score_and_store(len(trial_key), 'trial_key', trial_key, plain)

    # Sort by best quadgram score across all languages (higher = better)
    all_results.sort(key=lambda r: -r['best'])

    # -- Display top unique results --------------------------------------- #
    print('\nTop 8 results  (unique keys, ranked by best quadgram score):')
    seen_keys: set[str] = set()
    shown = 0
    for r in all_results:
        if r['key'] in seen_keys:
            continue
        seen_keys.add(r['key'])
        shown += 1
        if shown > 8:
            break
        print(f"\n  [{shown}] key='{r['key']}' | kl={r['kl']} | {r['source']} | IOC={r['ioc']:.4f}")
        for lang, sc in r['scores'].items():
            print(f"       {lang:<10} : {sc:.1f}")
        print(f"       plain[:80] : {r['plain'][:80]}")

    # -- Best result ------------------------------------------------------ #
    best = all_results[0]
    print('\n' + hr)
    print('BEST RESULT')
    print(hr)
    print(f"  Key length : {best['kl']}")
    print(f"  Key        : {best['key']}")
    print(f"  IOC        : {best['ioc']:.4f}")
    best_lang = max(best['scores'], key=best['scores'].get)
    print(f"  Best lang  : {best_lang}  (score {best['scores'][best_lang]:.1f})")
    print()
    print(fmt_blocks(best['plain']))

    # -- Save output ------------------------------------------------------ #
    out_path = STAGE_DIR / 'decrypted_stage04.txt'
    with out_path.open('w') as f:
        f.write(f"Key length : {best['kl']}\n")
        f.write(f"Key        : {best['key']}\n")
        f.write(f"Best lang  : {best_lang}  (score {best['scores'][best_lang]:.1f})\n\n")
        f.write(fmt_blocks(best['plain']) + '\n')
    print(f'\nSaved to {out_path}')


if __name__ == '__main__':
    main()
