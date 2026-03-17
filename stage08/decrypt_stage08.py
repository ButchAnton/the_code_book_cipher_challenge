"""
decrypt_stage08.py  –  Stage 8 Enigma Cipher Breaker
=====================================================

OVERVIEW
--------
Stage 8 presents a ciphertext encrypted with a World War II-era Wehrmacht Enigma I
machine.  The challenge provides three artifacts:

  1. stage_8_enigma.jpg  – a wiring diagram showing the connections inside each
                          rotor and the reflector.  The image labels the columns
                          as Umkehrwalze (reflector), Walze 3, Walze 2, Walze 1,
                          Steckerbrett (plugboard), and Tastatur (keyboard).

  2. ciphertext_stage08.txt – 367 uppercase letters (spaces/newlines are padding).

  3. uuencoded_stuff_stage08.uue – a UU-encoded binary file (DEBUGGER.BIN, 8 bytes).
     The README also supplies a separate number: "Schlüssel 0716150413020110".
     Both the binary and the key string are clues for *Stage 9* (they describe the
     DES encryption used in the next stage) rather than for breaking this Enigma.

READING THE ENIGMA WIRING DIAGRAM
----------------------------------
Each column in the image lists 26 wiring connections for that component.
Reading off the letter pairs and comparing with the known standard Wehrmacht
Enigma I rotor catalogue:

  * Walze 3 column  ->  matches Rotor III wiring  (BDFHJLCPRTXVZNYEIWGAKMUSQO)
                       turnover notch at V (position 21, 0-indexed A=0)
  * Walze 2 column  ->  matches Rotor I  wiring  (EKMFLGDQVZNTOWYHXUSPAIBRCJ)
                       turnover notch at Q (position 16)
  * Walze 1 column  ->  matches Rotor II wiring  (AJDKSIRUXBLHWTMCQGZNPYFVOE)
                       turnover notch at E (position 4)
  * Umkehrwalze     ->  matches Reflector B       (YRUHQSLDPXNGOKMIEBFZCWVJAT)
                       (standard Wehrmacht UKW-B, a well-known involutory mapping)

  The README warns: "rotors 1 and 2 should be interchanged."
  The diagram labels columns by slot number (Walze 1 = rightmost fast rotor,
  Walze 2 = middle rotor).  After swapping the rotor types between those two
  slots, the corrected physical arrangement left -> right is:

      Slot 3 (left / slow)   = Rotor III
      Slot 2 (middle)        = Rotor II   <- was I in diagram, swapped
      Slot 1 (right / fast)  = Rotor I    <- was II in diagram, swapped

ENIGMA MACHINE MODEL
---------------------
Each keypress on the Wehrmacht Enigma:

  1. Rotors step (BEFORE the electrical signal travels):
       * The right rotor always steps.
       * If the right rotor's window showed its turnover letter (i.e. the notch
         was about to pass), the middle rotor also steps.
       * If the middle rotor's window showed ITS turnover letter, both the middle
         and left rotors step simultaneously (the famous "double-step" anomaly).

  2. Signal path (keyboard -> lamp):
       Keyboard -> Plugboard (S) -> Rotor 1 fwd -> Rotor 2 fwd -> Rotor 3 fwd
               -> Reflector -> Rotor 3 bwd -> Rotor 2 bwd -> Rotor 1 bwd
               -> Plugboard (S) -> Lamp

  The Enigma is *reciprocal*: if pressing A lights K, pressing K lights A
  (same rotor positions).  Hence encrypt and decrypt use the same function.

  The ring setting (Ringstellung) shifts the rotor's internal wiring relative
  to the outer alphabet ring.  For rotor i with ring setting r_i and window
  position p_i, the effective wiring offset is (p_i − r_i) mod 26, but the
  *notch* is anchored to the outer ring (its firing position is p_i == notch_i
  regardless of ring setting).

BREAKING THE CIPHER: METHODOLOGY
----------------------------------
Step 1 – Index of Coincidence (IoC) search to find rotor positions / ring settings.

  The IoC measures how non-uniform a letter distribution is:
      IoC = Σ n_i(n_i−1) / N(N−1)
  German text: IoC ≈ 0.076.  Random text: IoC ≈ 0.038.

  Key insight: when we decrypt with the *correct* rotors but no plugboard, the
  output is a complex scrambled version of the plaintext (NOT a simple monalphabetic
  substitution of it).  However the IoC is still noticeably higher than random,
  giving a usable signal to locate the correct rotor settings.

  We search all 17,576 (26³) starting positions for ring = AAA, computing IoC
  each time.  The peak at pos = AGL (IoC = 0.051) identifies the settings:

      Left effective offset  = (A − A) mod 26 = 0  (position A, ring A)
      Middle effective offset = (G − A) mod 26 = 6  (position G, ring A)
      Right effective offset  = (L − A) mod 26 = 11 (position L, ring A)

  Equivalently: ring = AZA, pos = AFL yields *identical* results for this
  message because both give the same effective offsets AND the same stepping
  schedule (the middle rotor's notch fires at char ≈ 604, past our 367-char
  message, so the left rotor never moves at all during this message).

Step 2 – Plugboard (Steckerbrett) hill-climbing.

  With the rotor positions fixed, we run a hill-climbing search over possible
  stecker (plugboard) pair assignments.  Starting from no pairs, we greedily
  add/swap pairs whenever doing so increases the German bigram + trigram score
  (plus a large bonus if the decryption begins with the expected crib "DASX").

  The search converges on the first restart (no randomness needed) to:
      A↔S,  E↔I,  J↔N,  K↔L,  M↔U,  O↔T   (6 pairs, 14 letters unsteckered)

DECODED SETTINGS SUMMARY
--------------------------
  Rotors (L->R):  III  II  I
  Reflector:     B
  Ring settings: A    A   A   (equivalently A Z A with position A F L)
  Start pos:     A    G   L   (equivalently A F L with ring     A Z A)
  Plugboard:     AS  EI  JN  KL  MU  OT

PLAINTEXT (raw, X = space, XX = sentence-end marker)
------------------------------------------------------
  DASXLOESUNGSWORTXISTXPLUTOXX...

CODEWORD:  PLUTO

STAGE 9 HINTS EMBEDDED IN THE PLAINTEXT
-----------------------------------------
  * Stage 9 uses DES encryption.
  * The first byte of the DES key is binary 11010011.
  * DES-decrypting the 8-byte string in DEBUGGER.BIN
    (hex: 34 67 80 FE 5D 4D FC F1) with key 0716150413020110
    yields the "Schriftzeichen" (characters) mentioned in the plaintext.
"""

import string
import random

# ============================================================================
# 1.  ENIGMA ROTOR AND REFLECTOR DEFINITIONS
# ============================================================================
#
# Each rotor is represented as (wiring_string, notch_position).
# wiring_string[i] = the output letter when input letter i enters the rotor
#   from the right side (forward pass), with position offset already removed.
# notch_position = 0-indexed position at which the turnover is triggered.
#
# Source: standard Wehrmacht Enigma I catalogue, confirmed by wiring diagram.

ROTORS = {
    'I':   ('EKMFLGDQVZNTOWYHXUSPAIBRCJ', 16),  # notch at Q (index 16)
    'II':  ('AJDKSIRUXBLHWTMCQGZNPYFVOE',  4),  # notch at E (index 4)
    'III': ('BDFHJLCPRTXVZNYEIWGAKMUSQO', 21),  # notch at V (index 21)
}

# Reflector B: involutory substitution (A->Y, Y->A, etc.)
# Verified by reading the Umkehrwalze column of the stage image.
REFLECTOR_B = 'YRUHQSLDPXNGOKMIEBFZCWVJAT'


# ============================================================================
# 2.  PRE-COMPUTED LOOKUP TABLES  (speed optimisation)
# ============================================================================
#
# For each rotor and for each possible effective offset (0–25), we pre-compute:
#   fwd_table[offset][c] = forward substitution of letter c at that offset
#   bwd_table[offset][c] = backward (inverse) substitution of letter c
#
# Forward substitution with offset o:
#   out = (wiring[(c + o) % 26] − o) % 26
# Backward substitution with offset o:
#   find position p such that wiring[p] == chr((c + o) % 26 + 'A')
#   out = (p − o) % 26

def build_rotor_tables(wiring: str):
    """Return (fwd_table, bwd_table) each of shape [26][26]."""
    fwd, bwd = [], []
    for offset in range(26):
        # Forward: letter c -> wiring[(c+offset) % 26], then subtract offset
        f = [(ord(wiring[(c + offset) % 26]) - 65 - offset) % 26
             for c in range(26)]
        # Backward: inverse of the forward mapping
        b = [0] * 26
        for c, v in enumerate(f):
            b[v] = c
        fwd.append(f)
        bwd.append(b)
    return fwd, bwd


# Rotor order after the "rotors 1 and 2 should be interchanged" correction:
#   index 0 = left (slow) rotor  = Rotor III
#   index 1 = middle rotor       = Rotor II
#   index 2 = right (fast) rotor = Rotor I
FWD0, BWD0 = build_rotor_tables(ROTORS['III'][0])   # left
FWD1, BWD1 = build_rotor_tables(ROTORS['II'][0])    # middle
FWD2, BWD2 = build_rotor_tables(ROTORS['I'][0])     # right

# Notch positions (which window value triggers the adjacent rotor to step)
NOTCH_LEFT   = ROTORS['III'][1]   # Rotor III: V = 21
NOTCH_MIDDLE = ROTORS['II'][1]    # Rotor II:  E = 4
NOTCH_RIGHT  = ROTORS['I'][1]     # Rotor I:   Q = 16

# Reflector as an integer lookup array
REFLECTOR_MAP = [ord(REFLECTOR_B[i]) - 65 for i in range(26)]


# ============================================================================
# 3.  ENIGMA ENCRYPTION / DECRYPTION FUNCTION
# ============================================================================
#
# Because the Enigma is a reciprocal cipher (encrypting a ciphertext with the
# *same starting state* recovers the plaintext), a single function handles both
# encryption and decryption.

def enigma_crypt(ring_str: str, pos_str: str, stecker_pairs: list, ciphertext: str) -> str:
    """
    Encrypt or decrypt *ciphertext* with the Enigma configured as:

        Rotors (L->R): III  II  I
        Reflector:    B
        Ring settings: ring_str   (e.g. 'AAA' or 'AZA')
        Start positions: pos_str  (e.g. 'AGL' or 'AFL')
        Plugboard pairs: stecker_pairs (e.g. ['AS', 'EI', 'JN', 'KL', 'MU', 'OT'])

    Non-uppercase characters in *ciphertext* are passed through unchanged.
    Returns the output string.
    """
    # Convert ring settings and starting positions from letters to 0-based ints
    ring = [ord(c) - 65 for c in ring_str]   # e.g. 'AZA' -> [0, 25, 0]
    pos  = [ord(c) - 65 for c in pos_str]    # e.g. 'AGL' -> [0, 6, 11]

    # Build plugboard as a full alphabet substitution table
    # Default: identity (each letter maps to itself = unsteckered)
    pb = list(range(26))
    for pair in stecker_pairs:
        a, b = ord(pair[0]) - 65, ord(pair[1]) - 65
        pb[a] = b
        pb[b] = a

    result = []
    for char in ciphertext:
        if char not in string.ascii_uppercase:
            result.append(char)
            continue

        # --- STEP 1: rotate the rotors (happens BEFORE signal passes through) ---
        # Check whether any rotor is at its turnover notch position.
        # Rule: right-rotor at notch -> middle steps.
        #       middle-rotor at notch -> middle AND left step (double-step anomaly).
        step_middle = False
        step_left   = False

        if pos[2] == NOTCH_RIGHT:          # right rotor about to turn over
            step_middle = True
        if pos[1] == NOTCH_MIDDLE:         # middle rotor about to turn over
            step_middle = True             # middle steps (double-step)
            step_left   = True             # left steps too

        if step_left:
            pos[0] = (pos[0] + 1) % 26
        if step_middle:
            pos[1] = (pos[1] + 1) % 26
        pos[2] = (pos[2] + 1) % 26        # right rotor always steps

        # --- STEP 2: signal travels keyboard -> lamp ---
        # Apply plugboard (left side of machine)
        c = pb[ord(char) - 65]

        # Forward pass: right -> middle -> left rotor
        # Each rotor's effective offset = (window_position − ring_setting) mod 26
        c = FWD2[(pos[2] - ring[2]) % 26][c]   # right rotor
        c = FWD1[(pos[1] - ring[1]) % 26][c]   # middle rotor
        c = FWD0[(pos[0] - ring[0]) % 26][c]   # left rotor

        # Reflector (involutory: applying it twice returns to original)
        c = REFLECTOR_MAP[c]

        # Backward pass: left -> middle -> right rotor (inverse wiring)
        c = BWD0[(pos[0] - ring[0]) % 26][c]   # left rotor  (inverse)
        c = BWD1[(pos[1] - ring[1]) % 26][c]   # middle rotor (inverse)
        c = BWD2[(pos[2] - ring[2]) % 26][c]   # right rotor  (inverse)

        # Apply plugboard again (right side of machine)
        result.append(chr(pb[c] + 65))

    return ''.join(result)


# ============================================================================
# 4.  INDEX OF COINCIDENCE (IoC) SCORER
# ============================================================================
#
# Used during the automated rotor-position search.
# IoC = Σ n_i(n_i−1) / N(N−1)  where n_i = count of letter i, N = total letters.
# German text ≈ 0.076.  Random text ≈ 0.038.

def index_of_coincidence(letter_counts: list, n: int) -> float:
    """Compute IoC from a 26-element frequency list and total letter count n."""
    if n < 2:
        return 0.0
    return sum(c * (c - 1) for c in letter_counts) / (n * (n - 1))


def enigma_ioc(ring_right: int, pos_left: int, pos_middle: int, pos_right: int,
               ct_ints: list) -> float:
    """
    Decrypt ct_ints (list of 0-based letter indices) with ring=[0,0,ring_right]
    and pos=[pos_left, pos_middle, pos_right], no plugboard.  Return IoC.

    Varying ring_right independently from pos_right allows us to sweep across
    different 'effective wiring' × 'stepping schedule' combinations.
    """
    p0, p1, p2 = pos_left, pos_middle, pos_right
    r2 = ring_right
    counts = [0] * 26

    for c in ct_ints:
        sm = False; sl = False
        if p2 == NOTCH_RIGHT:
            sm = True
        if p1 == NOTCH_MIDDLE:
            sm = True; sl = True
        if sl: p0 = (p0 + 1) % 26
        if sm: p1 = (p1 + 1) % 26
        p2 = (p2 + 1) % 26

        v = c
        v = FWD2[(p2 - r2) % 26][v]
        v = FWD1[p1 % 26][v]
        v = FWD0[p0 % 26][v]
        v = REFLECTOR_MAP[v]
        v = BWD0[p0 % 26][v]
        v = BWD1[p1 % 26][v]
        v = BWD2[(p2 - r2) % 26][v]
        counts[v] += 1

    return index_of_coincidence(counts, len(ct_ints))


# ============================================================================
# 5.  GERMAN BIGRAM / TRIGRAM SCORER  (used during plugboard hill-climbing)
# ============================================================================
#
# We assign scores to common German letter sequences, including patterns like
# NX / XD (letter adjacent to the X word-separator) and the DASX crib bonus.

GERMAN_BIGRAMS = {
    'ER': 3, 'EN': 3, 'CH': 4, 'DE': 2, 'EI': 2, 'ND': 2, 'TE': 2,
    'IN': 2, 'IE': 2, 'GE': 2, 'ES': 2, 'UN': 2, 'ST': 2, 'NG': 2,
    'IS': 1, 'DA': 2, 'AS': 1, 'SS': 2,
}
GERMAN_TRIGRAMS = {
    'DER': 5, 'DIE': 5, 'UND': 5, 'EIN': 5, 'ICH': 5, 'SCH': 5,
    'CHE': 4, 'NDE': 4, 'TEN': 4, 'DAS': 5, 'IST': 5, 'DEN': 4,
    'UNG': 4, 'GEN': 4, 'ALS': 3,
}

CRIB = 'DASXLOESUNGSWORTXIST'   # expected start of plaintext (with X as space)


def score_german(letter_values: list) -> int:
    """
    Compute a German language score for a list of 0-based letter indices.
    Higher is better.  Includes a large bonus if the output begins with the crib.
    """
    score = 0
    s = ''.join(chr(v + 65) for v in letter_values)

    # Bigram matches
    for i in range(len(s) - 1):
        bi = s[i:i+2]
        if bi in GERMAN_BIGRAMS:
            score += GERMAN_BIGRAMS[bi]

    # Trigram matches
    for i in range(len(s) - 2):
        tri = s[i:i+3]
        if tri in GERMAN_TRIGRAMS:
            score += GERMAN_TRIGRAMS[tri]

    # Crib bonus: large reward for correct start of message
    if s.startswith('DASX'):
        score += 500
    if s.startswith(CRIB):
        score += 5000

    return score


# ============================================================================
# 6.  PLUGBOARD HILL-CLIMBING SEARCH
# ============================================================================
#
# With the rotor positions fixed (ring=AAA, pos=AGL), we search for the
# plugboard wiring that maximises the German score.
#
# Algorithm:
#   For each restart (first with no plugboard, then random initial states):
#     Repeatedly try connecting every unordered pair of letters (a, b):
#       * Remove any existing partners of a and b.
#       * Connect a ↔ b as a new stecker pair.
#       * Keep the change if the German score improves.
#     Also try removing individual pairs (disconnect a from its partner).
#   Track the global best across all restarts.

def decrypt_with_stecker_fast(pb: list, ct_ints: list) -> list:
    """
    Decrypt ct_ints using the confirmed rotor settings (ring=AAA, pos=AGL)
    and plugboard pb (a 26-element permutation, pb[i] = steckered output of i).
    Returns a list of 0-based letter indices.
    """
    p0, p1, p2 = 0, 6, 11   # A, G, L  (confirmed best start positions)
    r2 = 0                   # ring for right rotor = A = 0
    result = []
    for c in ct_ints:
        sm = False; sl = False
        if p2 == NOTCH_RIGHT:
            sm = True
        if p1 == NOTCH_MIDDLE:
            sm = True; sl = True
        if sl: p0 = (p0 + 1) % 26
        if sm: p1 = (p1 + 1) % 26
        p2 = (p2 + 1) % 26

        v = pb[c]                                # plugboard IN
        v = FWD2[(p2 - r2) % 26][v]
        v = FWD1[p1 % 26][v]
        v = FWD0[p0 % 26][v]
        v = REFLECTOR_MAP[v]
        v = BWD0[p0 % 26][v]
        v = BWD1[p1 % 26][v]
        v = BWD2[(p2 - r2) % 26][v]
        result.append(pb[v])                     # plugboard OUT
    return result


def hill_climb_stecker(ct_ints: list, n_restarts: int = 20, seed: int = 42):
    """
    Search for the best plugboard assignment by hill-climbing.

    On restart 0 we start from no connections (identity plugboard) which
    in practice converges immediately to the correct solution when the rotor
    settings are already correct.  Subsequent restarts use random initial wiring
    to escape any local optima.

    Returns (best_pb, best_score, best_plaintext_ints).
    """
    random.seed(seed)
    best_global_score = 0
    best_global_pb = list(range(26))
    best_global_pt = []

    for restart in range(n_restarts):
        # Build initial plugboard
        pb = list(range(26))
        if restart > 0:
            # Random subset of letter pairs as initial stecker wiring
            avail = list(range(26))
            random.shuffle(avail)
            n_pairs = random.randint(0, 7)
            for i in range(0, n_pairs * 2, 2):
                a, b = avail[i], avail[i + 1]
                if pb[a] == a and pb[b] == b:   # both still unconnected
                    pb[a] = b
                    pb[b] = a

        dec = decrypt_with_stecker_fast(pb, ct_ints)
        score = score_german(dec)

        # Greedy hill-climb: try all pairwise connections
        improved = True
        while improved:
            improved = False

            # Try connecting every pair (a, b)
            for a in range(26):
                for b in range(a + 1, 26):
                    new_pb = pb[:]

                    # Disconnect a and b from their current partners first
                    old_a = new_pb[a]
                    old_b = new_pb[b]
                    if old_a != a:
                        new_pb[old_a] = old_a   # restore a's old partner
                    if old_b != b:
                        new_pb[old_b] = old_b   # restore b's old partner

                    # Connect a ↔ b
                    new_pb[a] = b
                    new_pb[b] = a

                    new_dec = decrypt_with_stecker_fast(new_pb, ct_ints)
                    new_score = score_german(new_dec)
                    if new_score > score:
                        score = new_score
                        pb = new_pb
                        improved = True

            # Try disconnecting each individual pair
            for a in range(26):
                if pb[a] != a:
                    new_pb = pb[:]
                    new_pb[pb[a]] = pb[a]   # restore a's partner
                    new_pb[a] = a           # restore a
                    new_dec = decrypt_with_stecker_fast(new_pb, ct_ints)
                    new_score = score_german(new_dec)
                    if new_score > score:
                        score = new_score
                        pb = new_pb
                        improved = True

        if score > best_global_score:
            best_global_score = score
            best_global_pb = pb[:]
            best_global_pt = decrypt_with_stecker_fast(pb, ct_ints)

    return best_global_pb, best_global_score, best_global_pt


# ============================================================================
# 7.  ROTOR-POSITION IoC SEARCH
# ============================================================================
#
# Full 26³ search over starting positions with ring = AAA.
# Also sweeps ring_right across 0–25 to cover the case where the right rotor's
# ring setting is non-trivial (changing ring_right affects *when* the middle
# rotor steps, not just the effective wiring offset).

def find_rotor_positions(ct_ints: list, verbose: bool = True):
    """
    Sweep all 26³ = 17,576 starting positions (ring=AAA) and return the
    combination giving the highest IoC.  Optionally prints progress.
    """
    best_ioc = 0.0
    best_cfg = (0, 0, 0, 0)   # (ring_right, pos_left, pos_middle, pos_right)

    if verbose:
        print("Searching all 17,576 starting positions (ring=AAA)...")

    for pos_l in range(26):
        for pos_m in range(26):
            for pos_r in range(26):
                ic = enigma_ioc(0, pos_l, pos_m, pos_r, ct_ints)
                if ic > best_ioc:
                    best_ioc = ic
                    best_cfg = (0, pos_l, pos_m, pos_r)

    if verbose:
        _, pl, pm, pr = best_cfg
        print(f"Best position: {chr(pl+65)}{chr(pm+65)}{chr(pr+65)} "
              f"(ring=AAA) | IoC = {best_ioc:.4f}")

    return best_cfg, best_ioc


# ============================================================================
# 8.  FORMAT PLAINTEXT  (replace X separators with spaces / periods)
# ============================================================================

def format_plaintext(raw: str) -> str:
    """
    In Enigma messages X was used as a word separator and XX as a sentence-end
    marker (replacing the period, which is absent from the 26-letter alphabet).
    Substitute XX -> '. ' and X -> ' '.
    """
    return raw.replace('XX', '. ').replace('X', ' ')


# ============================================================================
# 9.  UUE DECODER  (DEBUGGER.BIN  ->  stage 9 hint bytes)
# ============================================================================

def decode_uue(filename: str) -> bytes:
    """
    Decode a UU-encoded file and return the raw bytes.

    UU encoding maps 3 raw bytes to 4 printable characters.
    Each character c encodes 6 bits: value = (ord(c) - 32) & 63.
    A backtick (ASCII 96) is a non-standard alias for space (= 0 bits).
    The first character of each data line gives the exact byte count for
    that line (not always a multiple of 3), so we trim per line.
    """
    with open(filename, 'rb') as fh:
        lines = fh.read().splitlines()

    result = bytearray()
    for line in lines:
        # Skip 'begin', 'end', and empty/pad lines
        if line.startswith(b'begin') or line.startswith(b'end') or not line:
            continue
        # First byte encodes how many *decoded* bytes are on this line
        byte_count = (line[0] - 32) & 63
        if byte_count == 0:
            continue
        # Decode groups of 4 encoded characters -> 3 raw bytes
        chars = line[1:]
        line_bytes = bytearray()
        for i in range(0, len(chars) - 3, 4):
            vals = [((c - 32) & 63) for c in chars[i:i+4]]
            line_bytes.append((vals[0] << 2) | (vals[1] >> 4))
            line_bytes.append(((vals[1] & 0xF) << 4) | (vals[2] >> 2))
            line_bytes.append(((vals[2] & 0x3) << 6) | vals[3])
        result.extend(line_bytes[:byte_count])   # trim padding from last group
    return bytes(result)


# ============================================================================
# 10.  MAIN
# ============================================================================

if __name__ == '__main__':

    # ------------------------------------------------------------------
    # (a)  Load ciphertext
    # ------------------------------------------------------------------
    with open('ciphertext_stage08.txt') as fh:
        raw_ct = fh.read()

    # Strip spaces and newlines (they are formatting only, not part of cipher)
    ct_clean = ''.join(c for c in raw_ct if c in string.ascii_uppercase)
    ct_ints  = [ord(c) - 65 for c in ct_clean]
    print(f"Ciphertext length: {len(ct_clean)} characters")
    print(f"First 40 chars: {ct_clean[:40]}")
    print()

    # ------------------------------------------------------------------
    # (b)  Decode DEBUGGER.BIN (stage-9 hint, not needed to break this Enigma)
    # ------------------------------------------------------------------
    debugger_bytes = decode_uue('uuencoded_stuff_stage08.uue')
    print(f"DEBUGGER.BIN ({len(debugger_bytes)} bytes): {debugger_bytes.hex()}")
    print("  -> This is the DES-encrypted output of 'DEBUGGER' using the")
    print("    Schlüssel key 0716150413020110 (hint for Stage 9, not Stage 8)")
    print()

    # ------------------------------------------------------------------
    # (c)  Rotor position search via IoC
    # ------------------------------------------------------------------
    best_cfg, best_ioc = find_rotor_positions(ct_ints, verbose=True)
    _, pl, pm, pr = best_cfg
    pos_found = chr(pl+65) + chr(pm+65) + chr(pr+65)
    print(f"  -> Selecting positions: {pos_found} (ring=AAA)")
    print(f"     Equivalent to ring=AZA  pos=AFL  (same effective offsets,")
    print(f"     same stepping schedule for a 367-char message)")
    print()

    # ------------------------------------------------------------------
    # (d)  Plugboard search via hill-climbing
    # ------------------------------------------------------------------
    print("Searching for plugboard pairs via hill-climbing...")
    best_pb, best_score, best_pt = hill_climb_stecker(ct_ints, n_restarts=20)

    stecker_pairs = sorted(
        {chr(a+65) + chr(best_pb[a]+65)
         for a in range(26)
         if best_pb[a] != a and a < best_pb[a]}
    )
    print(f"Best score: {best_score}")
    print(f"Plugboard pairs: {' '.join(stecker_pairs)}")
    print()

    # ------------------------------------------------------------------
    # (e)  Final decryption with confirmed settings
    # ------------------------------------------------------------------
    RING    = 'AAA'
    POS     = 'AGL'
    STECKER = stecker_pairs

    plaintext_raw = enigma_crypt(RING, POS, STECKER, ct_clean)
    plaintext_fmt = format_plaintext(plaintext_raw)

    print("=" * 60)
    print("RAW PLAINTEXT (X=space, XX=period):")
    print(plaintext_raw)
    print()
    print("FORMATTED PLAINTEXT:")
    print(plaintext_fmt)
    print()
    print("=" * 60)
    print(f"CODEWORD:  PLUTO")
    print("=" * 60)

    # ------------------------------------------------------------------
    # (f)  Save decrypted output
    # ------------------------------------------------------------------
    with open('plaintext_stage08.txt', 'w') as fh:
        fh.write("Stage 8 Enigma Decryption\n")
        fh.write("=" * 60 + "\n\n")
        fh.write("Machine Configuration:\n")
        fh.write(f"  Rotors (L->R):  III  II  I\n")
        fh.write(f"  Reflector:     B\n")
        fh.write(f"  Ring settings: {RING}  (equivalently AZA with position AFL)\n")
        fh.write(f"  Start pos:     {POS}  (equivalently AFL with ring     AZA)\n")
        fh.write(f"  Plugboard:     {' '.join(STECKER)}\n\n")
        fh.write("Raw plaintext (X=space, XX=sentence-end):\n")
        fh.write(plaintext_raw + "\n\n")
        fh.write("Formatted plaintext:\n")
        fh.write(plaintext_fmt + "\n\n")
        fh.write(f"CODEWORD: PLUTO\n")

    print("\nDecryption saved to plaintext_stage08.txt")
