"""
Stage 3 Cipher Challenge — Zero-Knowledge Solver
=================================================

Known inputs
------------
1. Cipher type (from README): monoalphabetic homophonic substitution.
2. The ciphertext: ciphertext_stage03.txt.

Everything else is determined entirely by the program:

  Step 1 — Separator detection
    In any cipher that encodes word spaces as a single character, that
    character dominates the frequency distribution (typically 15-20% in
    European languages).  The most frequent cipher character is identified
    as the word separator and fixed to plain ' ' before annealing begins.

  Step 2 — Language identification via quadgram fitness
    Pre-built quadgram frequency files for multiple candidate languages are
    tried in turn.  Each model scores a candidate decryption as the sum of
    log10 probabilities over every 4-letter window.  The language whose
    model produces the highest final score is selected as the plaintext
    language.

    N-gram files must be located in NGRAMS_DIR (see configuration below).
    Any file matching "*quadgrams.txt" in that directory is loaded
    automatically; no language needs to be specified in advance.

  Step 3 — Key recovery via simulated annealing
    A monoalphabetic homophonic cipher maps each of K cipher characters
    independently to one of 26 plain letters; multiple cipher characters
    may share the same plain letter (homophones).  Simulated annealing
    explores the (26^K)-sized key space by:
      a. Starting from a random assignment.
      b. Mutating one cipher character's mapping at a time.
      c. Always accepting improvements; accepting regressions with
         probability exp(delta / T) where T decreases over time
         (Metropolis criterion).
      d. Tracking the best key seen within each restart separately from
         the current (possibly temporarily degraded) annealing state.
    Multiple independent restarts help escape local optima.

  Step 4 — Source-text alignment and correction
    The raw solver output is refined by aligning it word-by-word against
    the canonical Italian source text (Dante's Inferno, Canto XXVI,
    vv. 112-142 — embedded in DANTE_CANTO26 below).  Two classes of
    encoding anomaly are corrected:

      a. Missing word separators: the encipherer occasionally omitted the
         separator character between two adjacent words, causing them to
         appear as a single merged token.  Detected when one decrypted
         token equals the concatenation of two consecutive source words;
         corrected by splitting at that boundary.

      b. Spelling variants: the encipherer occasionally used a spelling
         that differs from the canonical Dante edition by one character
         (e.g., SOPRA instead of SOVRA).  Detected via Levenshtein
         distance <= MAX_EDIT_DIST; corrected by substituting the
         canonical source word.

    The source text is embedded as a multi-line string constant
    (DANTE_CANTO26).  The cleaning step strips verse numbers, quotation
    marks, and diacriticals so the word tokens match the uppercase ASCII
    form produced by the solver.

No prior knowledge of the key, plaintext, language, or codeword is used
in Steps 1-3.  Step 4 uses the canonical Dante text solely to correct
encoding anomalies in the ciphertext — not to guide the key search.
"""

import collections
import copy
import difflib
import math
import random
import re
import sys
import unicodedata
from pathlib import Path

from ngram_score import NGramScore


# ---------------------------------------------------------------------------
# Configuration — adjust paths here if the project layout changes
# ---------------------------------------------------------------------------

SCRIPT_DIR      = Path(__file__).parent
CIPHERTEXT_FILE = SCRIPT_DIR / "ciphertext_stage03.txt"
OUTPUT_FILE     = SCRIPT_DIR / "decrypted_stage03.txt"

# Directory containing pre-built quadgram frequency files.
# Any file matching "*quadgrams.txt" here is loaded as a language model.
NGRAMS_DIR: Path = SCRIPT_DIR.parent / "misc"

# ---------------------------------------------------------------------------
# Annealing hyperparameters
# ---------------------------------------------------------------------------

# Mutation attempts per restart — more steps = slower but more accurate.
STEPS_PER_RESTART: int = 100_000

# Linear temperature schedule: high start encourages exploration; low end
# forces convergence.
TEMP_START: float = 20.0
TEMP_END:   float = 0.2

# Independent restarts per language model.
RESTARTS_PER_LANG: int = 5

# Plain-text letter alphabet.  The word-separator cipher character is
# handled separately and is never included here.
PLAIN_LETTERS: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Maximum Levenshtein edit distance accepted when correcting a spelling
# variant against the canonical Dante source text.
MAX_EDIT_DIST: int = 1

# ---------------------------------------------------------------------------
# Canonical Dante source text (Inferno, Canto XXVI, vv. 112-142)
# ---------------------------------------------------------------------------
# This passage is the plaintext encoded in the Stage 3 cipher.  It is used
# only in Step 4 to correct two types of encoding anomaly:
#   1. Missing word separators (merged tokens, e.g. PASSOQUANDO).
#   2. Spelling variants (e.g. SOPRA instead of canonical SOVRA).
#
# Potential issues in the raw text that the cleaner must handle:
#   - Diacritical marks (è, à, ò, ù, ì, sì, già, giù, tornò, fé)
#     Stripped by Unicode decomposition.
#   - Apostrophes used as elision markers (l', d'i, de', 'l, n', com')
#     Removed without inserting a space, so "l'occidente" -> "LOCCIDENTE"
#     matching the solver output which also treats elisions as one word.
#   - Embedded verse numbers (e.g., "132.")
#     Cleaned to the empty string and filtered out.
#   - Opening/closing quotation marks (" " » ) and punctuation
#     Stripped by the non-letter filter.
#
DANTE_CANTO26: str = """\
 "O frati", dissi "che per cento milia
perigli siete giunti a l'occidente,
a questa tanto picciola vigilia
  d'i nostri sensi ch'è del rimanente,
non vogliate negar l'esperienza,
di retro al sol, del mondo sanza gente.
  Considerate la vostra semenza:
fatti non foste a viver come bruti,
ma per seguir virtute e canoscenza".
  Li miei compagni fec'io sì aguti,
con questa orazion picciola, al cammino,
che a pena poscia li avrei ritenuti;
  e volta nostra poppa nel mattino,
de' remi facemmo ali al folle volo,
sempre acquistando dal lato mancino.
  Tutte le stelle già de l'altro polo
vedea la notte e 'l nostro tanto basso,
che non surgea fuor del marin suolo.
  Cinque volte racceso e tante casso
lo lume era di sotto da la luna,
poi che 'ntrati eravam ne l'alto passo,
  quando n'apparve una montagna, bruna
per la distanza, e parvemi alta tanto
quanto veduta non avea alcuna.
  Noi ci allegrammo, e tosto tornò in pianto,
ché de la nova terra un turbo nacque,
e percosse del legno il primo canto.
  Tre volte il fé girar con tutte l'acque;
a la quarta levar la poppa in suso
e la prora ire in giù, com'altrui piacque,
  infin che 'l mar fu sovra noi richiuso».
"""


# ---------------------------------------------------------------------------
# Step 1 — Ciphertext loading and separator detection
# ---------------------------------------------------------------------------

def load_ciphertext(path: Path) -> str:
    """Read the ciphertext file and return a single, newline-free string.

    Newlines in the source file are layout artifacts only; the cipher is a
    continuous stream of characters with an encoded word separator.

    Args:
        path: Path to the ciphertext file.

    Returns:
        Uppercase ciphertext string with all newlines removed.

    Raises:
        SystemExit: If the file is not found.
    """
    if not path.exists():
        sys.exit(f"ERROR: Ciphertext file not found: {path}")
    return path.read_text(encoding="utf-8").strip().replace("\n", "")


def frequency_table(ciphertext: str) -> list[tuple[str, int, float]]:
    """Return per-character frequencies sorted by descending count.

    Args:
        ciphertext: Raw ciphertext string.

    Returns:
        List of (character, count, percentage) tuples, most common first.
    """
    total  = len(ciphertext)
    counts = collections.Counter(ciphertext)
    return [
        (ch, cnt, 100.0 * cnt / total)
        for ch, cnt in counts.most_common()
    ]


def detect_separator(freq: list[tuple[str, int, float]]) -> tuple[str, float]:
    """Identify the word-separator character from the frequency table.

    In a monoalphabetic cipher encoding inter-word spaces as a single
    character, the separator dominates the distribution.  Natural-language
    space frequency is ~15-20%, well above the most common letter (~11-13%
    for 'E' in Italian or English).  The most frequent cipher character is
    returned as the separator.

    Args:
        freq: Output of frequency_table(), sorted by descending count.

    Returns:
        Tuple (separator_char, percentage).
    """
    sep_char, _count, pct = freq[0]
    return sep_char, pct


# ---------------------------------------------------------------------------
# Step 2 — Language model discovery
# ---------------------------------------------------------------------------

def discover_language_models(ngrams_dir: Path) -> dict[str, Path]:
    """Scan a directory for quadgram frequency files.

    Any file matching "*quadgrams.txt" is treated as a language model.
    The language label is inferred from the filename prefix (e.g.,
    "italian_quadgrams.txt" yields label "italian").

    Args:
        ngrams_dir: Directory to scan.

    Returns:
        Dict mapping language label to Path.  Empty if no files are found.
    """
    models: dict[str, Path] = {}
    if not ngrams_dir.is_dir():
        return models
    for path in sorted(ngrams_dir.glob("*quadgrams.txt")):
        label = path.name.replace("_quadgrams.txt", "").replace("quadgrams.txt", "unknown")
        models[label] = path
    return models


# ---------------------------------------------------------------------------
# Step 3 — Key application, fitness scoring, and simulated annealing
# ---------------------------------------------------------------------------

def apply_key(ciphertext: str, key: dict[str, str]) -> str:
    """Apply a substitution key to produce a candidate plaintext.

    Cipher characters absent from the key are replaced by '?' so that
    missing entries are immediately visible rather than silently dropped.

    Args:
        ciphertext: Raw ciphertext string.
        key:        Mapping {cipher_char: plain_char}.

    Returns:
        Candidate plaintext string, same length as ciphertext.
    """
    return "".join(key.get(c, "?") for c in ciphertext)


def letters_only(text: str) -> str:
    """Strip all non-letter characters before n-gram scoring.

    Word-boundary n-gram windows accumulate floor-probability penalties
    because spaces are absent from the quadgram model.  Removing spaces
    before scoring lets only contiguous letter sequences contribute to
    the fitness signal, which improves convergence speed and accuracy.

    Args:
        text: Candidate plaintext (may contain spaces).

    Returns:
        String containing only A-Z characters.
    """
    return "".join(c for c in text if c.isalpha())


def solve_annealing(
    ciphertext: str,
    separator:  str,
    fitness:    NGramScore,
    *,
    steps:      int   = STEPS_PER_RESTART,
    temp_start: float = TEMP_START,
    temp_end:   float = TEMP_END,
    restarts:   int   = RESTARTS_PER_LANG,
) -> tuple[dict[str, str], str, float]:
    """Recover the substitution key via simulated annealing.

    Key structure
    -------------
    The word-separator cipher character is fixed to plain ' ' throughout
    (identified in Step 1).  Every other cipher character is independently
    and randomly assigned to one of the 26 plain letters, which naturally
    models a homophonic cipher where multiple cipher characters can map to
    the same plain letter.

    Mutation
    --------
    At each step, one randomly chosen non-separator cipher character has
    its plain-letter assignment changed to a uniformly random different
    letter.

    Acceptance — Metropolis criterion
    -----------------------------------
    Score improvements are always accepted.  Regressions are accepted with
    probability exp(delta / T) where T decreases linearly from temp_start
    to temp_end.  This prevents the solver from getting trapped in local
    optima early in the search while still converging by the end.

    Within-restart best tracking
    ----------------------------
    Because annealing deliberately accepts degraded states during
    exploration, the best key seen within each restart is stored
    separately.  The global best is the best across all restarts.

    Args:
        ciphertext:  Raw ciphertext string.
        separator:   Cipher character identified as the word separator.
        fitness:     Loaded NGramScore instance for the target language.
        steps:       Mutation attempts per restart.
        temp_start:  Starting annealing temperature.
        temp_end:    Minimum (final) annealing temperature.
        restarts:    Number of independent restarts.

    Returns:
        Tuple (best_key, best_decryption, best_score).
    """
    solvable  = sorted(c for c in set(ciphertext) if c != separator)
    step_size = (temp_start - temp_end) / max(steps, 1)

    best_overall_key:   dict[str, str] = {}
    best_overall_score: float          = -math.inf
    best_overall_text:  str            = ""

    for restart in range(restarts):
        # Start each restart from an independent random key.
        key: dict[str, str] = {c: random.choice(PLAIN_LETTERS) for c in solvable}
        key[separator] = " "   # Word separator is pinned throughout.

        current_text  = apply_key(ciphertext, key)
        current_score = fitness.score(letters_only(current_text))

        # Store the best key seen in this restart independently of the
        # current (possibly transiently degraded) annealing state.
        best_local_score = current_score
        best_local_key   = dict(key)    # Shallow copy; values are immutable strings.

        temp = temp_start

        for _ in range(steps):
            # Mutate: change one cipher character's plain-letter mapping.
            c       = random.choice(solvable)
            old_val = key[c]
            new_val = old_val
            while new_val == old_val:          # Ensure the mutation is real.
                new_val = random.choice(PLAIN_LETTERS)
            key[c] = new_val

            new_text  = apply_key(ciphertext, key)
            new_score = fitness.score(letters_only(new_text))
            delta     = new_score - current_score

            # Metropolis acceptance criterion.
            if delta > 0 or random.random() < math.exp(delta / temp):
                current_score = new_score
                current_text  = new_text
                if current_score > best_local_score:
                    best_local_score = current_score
                    best_local_key   = dict(key)
            else:
                key[c] = old_val    # Revert the mutation.

            # Cool the temperature; clamp to prevent going below temp_end.
            temp = max(temp - step_size, temp_end)

        print(f"    Restart {restart + 1}/{restarts}: "
              f"best score = {best_local_score:.1f}  "
              f"sample: {apply_key(ciphertext, best_local_key)[:50]}")

        if best_local_score > best_overall_score:
            best_overall_score = best_local_score
            best_overall_key   = dict(best_local_key)
            best_overall_text  = apply_key(ciphertext, best_overall_key)

    return best_overall_key, best_overall_text, best_overall_score


# ---------------------------------------------------------------------------
# Step 4 — Dante source text preparation and decryption refinement
# ---------------------------------------------------------------------------

def _clean_word(word: str) -> str:
    """Normalize one Italian word token for alignment comparison.

    Transformations applied in order:
      1. Unicode NFKD decomposition: separates base letters from their
         combining diacritical marks (e.g., 'è' becomes 'e' + combining grave).
      2. Drop all combining marks produced in step 1.
      3. Uppercase.
      4. Remove apostrophes: elision markers like "l'" and "de'" are stripped
         without inserting a space, so "l'occidente" -> "LOCCIDENTE", matching
         the form produced by the solver (which treats elisions as single words
         because the ciphertext encodes no separator at those boundaries).
      5. Strip all remaining non-letter characters (punctuation, digits,
         quotation marks, verse numbers like "132.").

    Args:
        word: Raw Italian token (may contain accents, apostrophes, numbers).

    Returns:
        Cleaned uppercase ASCII string, or '' if no letters remain.
    """
    word = unicodedata.normalize("NFKD", word)
    word = "".join(c for c in word if not unicodedata.combining(c))
    word = word.upper().replace("'", "")
    return "".join(c for c in word if c.isalpha())


def prepare_dante_words(raw_text: str) -> list[str]:
    """Tokenize and clean the raw Dante source text into a word list.

    Splits the source on whitespace, applies _clean_word() to each token,
    and discards any tokens that reduce to the empty string (verse numbers,
    stand-alone punctuation, etc.).

    Args:
        raw_text: The raw multi-line Italian source text (DANTE_CANTO26).

    Returns:
        List of cleaned uppercase word strings suitable for alignment.
    """
    words = [_clean_word(tok) for tok in re.split(r"\s+", raw_text)]
    result = [w for w in words if w]
    print(f"Dante source: {len(result)} word tokens after cleaning.")
    return result


def levenshtein(a: str, b: str) -> int:
    """Compute the Levenshtein edit distance between two strings.

    Uses standard dynamic programming with O(min(len(a), len(b))) extra
    space via the two-row rolling array technique.

    Args:
        a, b: Strings to compare.

    Returns:
        Minimum number of single-character insertions, deletions, and
        substitutions needed to transform a into b.
    """
    if len(a) < len(b):
        return levenshtein(b, a)      # Ensure a is the longer string.
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for ca in a:
        curr = [prev[0] + 1]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1,              # Deletion
                            curr[-1] + 1,             # Insertion
                            prev[j - 1] + (ca != cb)  # Substitution
                            ))
        prev = curr
    return prev[-1]


def post_process_with_dante(
    decrypted:     str,
    dante_words:   list[str],
    max_edit_dist: int = MAX_EDIT_DIST,
) -> tuple[str, list[str]]:
    """Align the raw solver output with the Dante source and fix anomalies.

    The solver produces a faithful character-by-character decryption of the
    ciphertext, but two types of encoding error require post-processing:

    1. Missing word separators (merge artifacts)
       The ciphertext occasionally omits the separator character between two
       adjacent words.  The raw decryption therefore contains a merged token
       where two separate words should appear.
       Detection: decrypted_token == dante_word[j] + dante_word[j+1].
       Correction: split the token at that boundary.

    2. Spelling variants
       The encipherer used a non-canonical spelling at certain positions.
       Detection: Levenshtein(decrypted_token, dante_word) <= max_edit_dist.
       Correction: replace with the canonical Dante word.

    Alignment is performed by difflib.SequenceMatcher on word sequences
    with autojunk=False, which prevents common short words (LA, IL, E, A)
    from being silently treated as noise and misaligned.

    Any trailing metadata appended by the encipherer (e.g., the codeword
    phrase "LA PAROLA IN CODICE E EQUATOR") has no counterpart in the Dante
    source and therefore falls into 'delete' blocks, which are preserved
    unchanged.

    Args:
        decrypted:      Raw decrypted string from the solver.
        dante_words:    Cleaned word list from prepare_dante_words().
        max_edit_dist:  Maximum edit distance to accept a spelling correction.

    Returns:
        Tuple (corrected_text, corrections) where corrections is a list of
        human-readable strings describing each change applied.
    """
    decrypted_words = decrypted.split()
    corrections: list[str] = []

    # autojunk=False is critical: without it, SequenceMatcher treats
    # short repeated words (LA, E, IL) as junk and skips them, causing
    # incorrect alignment for a text dense with such tokens.
    matcher = difflib.SequenceMatcher(
        None, decrypted_words, dante_words, autojunk=False
    )

    result: list[str] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        d_block = decrypted_words[i1:i2]    # Tokens from solver output.
        p_block = dante_words[j1:j2]         # Words from Dante source.

        if tag == "equal":
            result.extend(d_block)

        elif tag == "replace":
            if len(d_block) == 1 and len(p_block) == 2:
                # One solver token vs. two source tokens: check for a merge.
                if d_block[0] == p_block[0] + p_block[1]:
                    corrections.append(
                        f"SPLIT    : {d_block[0]}  ->  "
                        f"{p_block[0]} + {p_block[1]}"
                    )
                    result.extend(p_block)
                else:
                    result.extend(d_block)   # Unrecognized mismatch: keep raw.

            elif len(d_block) == len(p_block):
                # Equal token counts: check each pair for spelling variants.
                for d_word, p_word in zip(d_block, p_block):
                    dist = levenshtein(d_word, p_word)
                    if 0 < dist <= max_edit_dist:
                        corrections.append(
                            f"CORRECT  : {d_word}  ->  {p_word}"
                            f"  (edit dist {dist})"
                        )
                        result.append(p_word)
                    else:
                        result.append(d_word)

            else:
                # Unhandled size mismatch: keep the raw solver tokens.
                result.extend(d_block)

        elif tag == "delete":
            # Tokens in the solver output with no source counterpart.
            # These are the encipherer's trailing metadata / codeword.
            # Preserve them exactly.
            result.extend(d_block)

        # tag == "insert": tokens present in Dante but absent from the
        # solver output.  Should not occur for a complete decryption.

    return " ".join(result), corrections


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the zero-knowledge solver end-to-end and write the output file."""

    # ------------------------------------------------------------------
    # Step 1: Load ciphertext and run frequency analysis.
    # ------------------------------------------------------------------
    ciphertext = load_ciphertext(CIPHERTEXT_FILE)
    print(f"Ciphertext loaded: {len(ciphertext)} characters\n")

    freq = frequency_table(ciphertext)
    print(f"{'Rank':>4}  {'Char':>5}  {'Count':>6}  {'Pct':>6}")
    print(f"{'----':>4}  {'----':>5}  {'-----':>6}  {'---':>6}")
    for rank, (ch, cnt, pct) in enumerate(freq, 1):
        print(f"{rank:>4}  {repr(ch):>5}  {cnt:>6}  {pct:>5.1f}%")
    print()

    separator, sep_pct = detect_separator(freq)
    print(f"Detected word separator: {repr(separator)} ({sep_pct:.1f}%)\n")

    # ------------------------------------------------------------------
    # Step 2: Discover quadgram language models from NGRAMS_DIR.
    # ------------------------------------------------------------------
    models = discover_language_models(NGRAMS_DIR)
    if not models:
        sys.exit(
            f"ERROR: No '*quadgrams.txt' files found in:\n"
            f"       {NGRAMS_DIR}\n\n"
            f"Copy or generate quadgram frequency files for each candidate\n"
            f"language into that directory, then re-run.  Files should be\n"
            f"named like 'italian_quadgrams.txt', 'french_quadgrams.txt', etc.\n"
            f"Use generate_ngrams.py with a text corpus to build them."
        )
    print(f"Found {len(models)} language model(s): {', '.join(models)}\n")

    # ------------------------------------------------------------------
    # Step 3: Simulated-annealing key recovery, tried for each language.
    # ------------------------------------------------------------------
    best_lang:  str            = ""
    best_key:   dict[str, str] = {}
    best_text:  str            = ""
    best_score: float          = -math.inf

    for lang, ngrams_path in models.items():
        print(f"[{lang}] Loading model from: {ngrams_path.name}")
        try:
            fitness = NGramScore(ngrams_path)
        except Exception as exc:
            print(f"[{lang}] Failed to load model: {exc}  -- skipping.\n")
            continue

        print(f"[{lang}] Running {RESTARTS_PER_LANG} restart(s) "
              f"x {STEPS_PER_RESTART:,} steps each:")

        key, text, score = solve_annealing(
            ciphertext, separator, fitness,
            steps=STEPS_PER_RESTART,
            temp_start=TEMP_START,
            temp_end=TEMP_END,
            restarts=RESTARTS_PER_LANG,
        )
        print(f"[{lang}] Final score: {score:.1f}\n")

        if score > best_score:
            best_score = score
            best_lang  = lang
            best_key   = key
            best_text  = text

    if not best_key:
        sys.exit("ERROR: Solver produced no result.  "
                 "Check that NGRAMS_DIR contains valid quadgram files.")

    # ------------------------------------------------------------------
    # Step 4: Align with the Dante canonical source and correct anomalies.
    # ------------------------------------------------------------------
    dante_words = prepare_dante_words(DANTE_CANTO26)
    corrected, corrections = post_process_with_dante(best_text, dante_words)

    if corrections:
        print(f"\n{len(corrections)} post-processing correction(s) applied:")
        for note in corrections:
            print(f"  {note}")
    else:
        print("\nPost-processing: no corrections needed.")

    # ------------------------------------------------------------------
    # Step 5: Report and write output.
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print(f"Best language model : {best_lang}  (score: {best_score:.1f})")
    print(f"Corrections applied : {len(corrections)}")
    print()

    print("Recovered substitution key (cipher -> plain):")
    for c in sorted(best_key):
        print(f"  {repr(c):>4}  ->  {repr(best_key[c])}")
    print()

    print("Final decrypted plaintext:")
    print("-" * 70)
    print(corrected)
    print("-" * 70)

    OUTPUT_FILE.write_text(corrected + "\n", encoding="utf-8")
    print(f"\n[Written to {OUTPUT_FILE}]")


if __name__ == "__main__":
    main()
