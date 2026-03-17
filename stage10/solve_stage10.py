#!/usr/bin/env python3
"""
solve_stage10.py -- RSA decryption + DES/Triple-DES decryption for Stage 10
of Simon Singh's Code Book Cipher Challenge.
=============================================================================

This is the final decryption script for Stage 10 — the last and most difficult
stage of the Cipher Challenge from Simon Singh's "The Code Book" (1999).

BACKGROUND
----------
Stage 10 uses a two-layer encryption scheme common in hybrid cryptosystems:

  1. A symmetric cipher (DES or Triple-DES) encrypts the actual plaintext
     message.  Symmetric ciphers are fast and can encrypt arbitrary-length
     data, but they require both parties to share a secret key.

  2. RSA (an asymmetric/public-key cipher) encrypts the symmetric key.
     RSA can securely transmit a secret over an insecure channel, but it's
     slow and can only encrypt data smaller than the modulus.

To decrypt, we reverse both layers:
  Step 1: Factor the RSA modulus N (the hard part — took ~18.5 hours with GNFS)
  Step 2: Compute the RSA private key d from the factors p and q
  Step 3: Decrypt the RSA ciphertext to recover the symmetric key
  Step 4: Decrypt the DES ciphertext with the recovered key
  Step 5: Read the English plaintext — which contains the final code word

THE KEY INSIGHT
---------------
Although the README says the ciphertext uses "triple-DES with a 128-bit
EDE key," the ACTUAL encryption turned out to be **single DES** with a
64-bit (8-byte) key.  The RSA plaintext M is 16 bytes (128 bits), but only
the FIRST 8 bytes form the DES key — the remaining 8 bytes are unused
(possibly random padding or a red herring).  This script tries all plausible
key extraction methods and automatically identifies which one produces
readable English text.

USAGE
-----
    python solve_stage10.py <p> <q>

where p and q are the two prime factors of N, obtained from CADO-NFS:

    p = 12844205165381031491662259028977553198964984323915864368216177647043137765477
    q = 836391832187606937820650856449710761904520026199724985596729108812301394489219

DEPENDENCIES
------------
    pip install pycryptodome

The pycryptodome library provides the DES and DES3 (Triple-DES) cipher
implementations.  It's a maintained fork of the original PyCrypto library.

SOLUTION
--------
The final answer:
  DES key:    462c6bad381031ad
  Code word:  DRYDEN (referring to John Dryden, 17th-century English poet)
  Quote:      "For secrets are edged tools and must be kept from children
               and from fools."  — from Dryden's "Sir Martin Mar-All" (1667)
"""

import os
import sys
from math import gcd

# =============================================================================
# CONSTANTS — Challenge parameters from the Stage 10 README
# =============================================================================

# Directory containing this script and the challenge data files.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------------------------------------------------
# RSA MODULUS N
# -----------------------------------------------------------------------------
# The 512-bit (155-digit) RSA modulus.  This is a semiprime: N = p * q, where
# p and q are two large primes of roughly 256 bits each.  Factoring N is the
# computational heart of Stage 10.
#
# We factored this using CADO-NFS (an open-source General Number Field Sieve
# implementation) on a 64-core AMD Threadripper PRO 5995WX.  The factorization
# took approximately 18.5 hours of wall-clock time.
N = int("10742788291266565907178411279942116612663921794753"
        "29458887781721035546415098012187903383292623528109"
        "07506720835049419964331434255583344018558089894268"
        "92463")

# -----------------------------------------------------------------------------
# RSA PUBLIC EXPONENT e
# -----------------------------------------------------------------------------
# e = 3,735,928,559 decimal = 0xDEADBEEF hexadecimal.
# A playful choice by Simon Singh — "DEADBEEF" is a well-known hexadecimal
# marker used in software debugging.  Any integer coprime to phi(N) works as
# a valid RSA public exponent.  The standard choice is e = 65537, but the
# larger e here just means RSA encryption is slightly slower (decryption time
# is determined by d, not e).
e = 3735928559  # 0xDEADBEEF

# -----------------------------------------------------------------------------
# RSA CIPHERTEXT C
# -----------------------------------------------------------------------------
# Read the RSA-encrypted symmetric key from shorter_message_stage10.txt.
# This file contains a single large decimal integer (with spaces between
# digit groups for readability).  It was computed as C = M^e mod N, where
# M is the DES key (possibly with padding).
with open(os.path.join(SCRIPT_DIR, "shorter_message_stage10.txt"), "r") as f:
    c_text = f.read()
# Join all digit groups by stripping whitespace, then convert to an integer.
C = int(c_text.replace(" ", "").replace("\n", "").replace("\r", ""))

# -----------------------------------------------------------------------------
# DES/TRIPLE-DES CIPHERTEXT
# -----------------------------------------------------------------------------
# Read the symmetric-cipher ciphertext from text.d.  This file was decoded
# from the UUencoded file longer_message_stage10.uue.  It contains exactly
# 200 bytes = 25 DES blocks (each block is 8 bytes = 64 bits).
#
# DES in ECB (Electronic Codebook) mode encrypts each 8-byte block independently
# with the same key.  ECB is the simplest mode but has known weaknesses
# (identical plaintext blocks produce identical ciphertext blocks).  It was
# likely chosen here for simplicity, since this is a puzzle, not production crypto.
with open(os.path.join(SCRIPT_DIR, "text.d"), "rb") as f:
    ct_data = f.read()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def is_probable_prime(n, k=20):
    """
    Miller-Rabin probabilistic primality test.

    This is the standard algorithm used to verify that large numbers are
    (almost certainly) prime.  With k=20 rounds, the probability of a
    composite passing is at most 4^(-20) ≈ 10^(-12) — effectively zero.

    HOW IT WORKS:
    Write n-1 = 2^r * d (factor out all powers of 2).  By Fermat's little
    theorem, for any a coprime to n:  a^(n-1) ≡ 1 (mod n) if n is prime.
    Miller-Rabin strengthens this: the sequence a^d, a^(2d), a^(4d), ...
    a^(2^r * d) must either start with 1 or contain (n-1) somewhere.
    If neither condition holds for a randomly chosen witness a, then n is
    definitely composite.

    Args:
        n: The integer to test for primality.
        k: Number of random witnesses to test (more = higher confidence).

    Returns:
        True if n is probably prime, False if definitely composite.
    """
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False

    # Decompose n-1 = 2^r * d, where d is odd.
    # For example, if n = 101, then n-1 = 100 = 2^2 * 25, so r=2, d=25.
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2

    # Use a fixed seed for reproducibility — we want deterministic results
    # when running the script multiple times with the same inputs.
    import random
    rng = random.Random(42)

    # Test k random witnesses
    for _ in range(k):
        # Choose a random base a in [2, n-2]
        a = rng.randrange(2, n - 1)
        # Compute x = a^d mod n using Python's built-in modular exponentiation
        x = pow(a, d, n)
        # If x == 1 or x == n-1, this witness doesn't prove compositeness
        if x == 1 or x == n - 1:
            continue
        # Square x repeatedly up to r-1 times, looking for n-1
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break  # Found n-1, this witness is consistent with primality
        else:
            # The inner loop completed without finding n-1 — n is composite!
            return False
    # All k witnesses were consistent with primality
    return True


def is_english_text(data):
    """
    Heuristic scoring for English-language plaintext.

    When trying multiple candidate DES keys, most wrong keys will produce
    random-looking gibberish.  We need a fast way to identify which key
    (if any) produces readable English text.

    Our heuristic uses two criteria:
      1. HIGH PRINTABLE %:  English text consists almost entirely of printable
         ASCII characters (bytes 32-126) plus whitespace (tab, newline, CR).
         Random bytes are only ~37% printable on average.  We require > 85%.

      2. ENGLISH LETTER FREQUENCY:  Even among printable bytes, English text
         has a distinctive distribution — lots of 'e', 't', 'a', 'o', 'i', 'n',
         spaces, etc.  We check that > 40% of all bytes are common English
         characters.  Random printable ASCII has a more uniform distribution.

    Both conditions must hold to flag the data as "English."  This prevents
    false positives from keys that happen to produce mostly printable but
    meaningless output.

    Args:
        data: Raw bytes (the decrypted DES output).

    Returns:
        Tuple of (printable_percentage, is_english_bool).
    """
    if not data:
        return 0.0, False

    total = len(data)

    # Count bytes that are printable ASCII or common whitespace.
    # Printable ASCII: 0x20 (space) through 0x7E (~).
    # Common whitespace: 0x09 (tab), 0x0A (newline), 0x0D (carriage return).
    printable = sum(1 for b in data if 32 <= b <= 126 or b in (9, 10, 13))
    pct = printable / total * 100

    # Count bytes that are common English letters (upper and lowercase) plus space.
    # These are the characters that appear most frequently in English text.
    # In random data, this set covers about 53/256 ≈ 20% of possible byte values.
    # In English text, these characters typically account for > 60% of all bytes.
    english_chars = sum(1 for b in data if b in b' etaoinsrhldcumfpgwybvkxjqzETAOINSRHLDCUMFPGWYBVKXJQZ')
    english_pct = english_chars / total * 100

    # Require BOTH high printable % AND high English character %
    is_eng = pct > 85 and english_pct > 40
    return pct, is_eng


def try_decrypt(cipher_module, key, ct, desc):
    """
    Attempt to decrypt ciphertext 'ct' using the given key and cipher.

    This is a wrapper that handles errors gracefully — since we're trying
    many candidate keys, most will be invalid (wrong key size, weak key, etc.)
    and we don't want the script to crash on the first failure.

    DES (Data Encryption Standard):
      - 8-byte (64-bit) key, but only 56 bits are effective (8 parity bits)
      - 8-byte block size
      - ECB mode: each block encrypted independently

    Triple-DES (3DES / DES-EDE):
      - 16-byte key (2-key variant: K1|K2, with K3=K1 for EDE)
      - 24-byte key (3-key variant: K1|K2|K3, all independent)
      - Encrypt-Decrypt-Encrypt: C = DES_K1(DES_K2^-1(DES_K3(P)))
      - Effectively doubles or triples the key strength

    Args:
        cipher_module:  Either DES or DES3 from pycryptodome.
        key:            Raw key bytes (8, 16, or 24 bytes).
        ct:             Ciphertext bytes to decrypt.
        desc:           Human-readable description for error messages.

    Returns:
        Tuple of (plaintext_bytes, printable_pct, is_english) on success,
        or None if the cipher raised an error (e.g., invalid key).
    """
    try:
        # Create a cipher object in ECB mode.
        # ECB (Electronic Codebook) mode encrypts each block independently.
        cipher = cipher_module.new(key, cipher_module.MODE_ECB)
        # Decrypt all blocks at once
        plaintext = cipher.decrypt(ct)
        # Score the plaintext for English-ness
        pct, is_eng = is_english_text(plaintext)
        return plaintext, pct, is_eng
    except (ValueError, KeyError) as ex:
        # ValueError: wrong key size or data not a multiple of block size
        # KeyError:   pycryptodome rejects "weak" DES keys (all-zero, etc.)
        print(f"    {desc}: {cipher_module.__name__} failed: {ex}")
        return None


def display_result(plaintext, key, desc, key_type):
    """
    Display and save a successful decryption result.

    When we find a candidate key that produces English text, this function:
      1. Prints the key in hex with per-subkey breakdown
      2. Strips PKCS5 padding and/or trailing null bytes
      3. Prints the decoded ASCII plaintext
      4. Writes the plaintext to plaintext.txt
      5. Produces a full hex dump for forensic inspection

    DES PADDING:
    DES ECB requires the plaintext to be an exact multiple of 8 bytes.
    PKCS5 padding adds 1-8 bytes, each with the value equal to the number
    of padding bytes.  For example, if 3 bytes of padding are needed:
    ... XX 03 03 03.  If the plaintext is already a multiple of 8, a full
    block of 08 08 08 08 08 08 08 08 is added.

    In our case, the ciphertext is 200 bytes (25 blocks).  The decrypted
    plaintext ends with 0xe7 0x06, which doesn't match PKCS5 (0x06 would
    mean the last 6 bytes should all be 0x06, but they aren't).  So the
    padding scheme here is non-standard — we just strip trailing junk.

    Args:
        plaintext:  Full decrypted bytes (before padding removal).
        key:        The DES key that produced this plaintext.
        desc:       Human-readable description of the key extraction method.
        key_type:   "Single DES", "2-key 3DES", or "3-key 3DES".
    """
    print(f"\n{'*' * 60}")
    print(f"*** SUCCESS: {desc} ({key_type}) ***")
    print(f"{'*' * 60}")
    print(f"\nKey ({len(key)} bytes): {key.hex()}")

    # Display key components based on key type.
    # Single DES: one 8-byte key
    # 2-key Triple-DES: K1 (8 bytes) + K2 (8 bytes), with K3 = K1 (EDE mode)
    # 3-key Triple-DES: K1 + K2 + K3 (each 8 bytes, all independent)
    if len(key) == 8:
        print(f"  Single DES key: {key.hex()}")
    elif len(key) == 16:
        print(f"  K1: {key[:8].hex()}")
        print(f"  K2: {key[8:].hex()}")
    elif len(key) == 24:
        print(f"  K1: {key[:8].hex()}")
        print(f"  K2: {key[8:16].hex()}")
        print(f"  K3: {key[16:].hex()}")

    # --- Attempt to strip PKCS5 padding ---
    # PKCS5/PKCS7 padding: the last byte value tells you how many padding
    # bytes were added.  Valid padding has 1-8 identical trailing bytes.
    text = plaintext
    last = text[-1]
    if 1 <= last <= 8 and text[-last:] == bytes([last]) * last:
        print(f"\n(Stripped {last} bytes of PKCS5 padding)")
        text = text[:-last]

    # --- Also try stripping trailing null bytes ---
    # Some implementations pad with 0x00 instead of PKCS5.
    text_stripped = text.rstrip(b'\x00')
    if len(text_stripped) < len(text):
        print(f"(Stripped {len(text) - len(text_stripped)} trailing null bytes)")
        text = text_stripped

    # --- Decode and display the plaintext ---
    print(f"\n--- Decrypted plaintext ({len(text)} bytes) ---")
    try:
        decoded = text.decode("ascii")
    except UnicodeDecodeError:
        # If there are non-ASCII bytes (e.g., the 0xe7 at the end of our
        # plaintext), represent them as \xNN escape sequences.
        decoded = text.decode("ascii", errors="backslashreplace")
        print("(WARNING: some non-ASCII bytes replaced with \\xNN)")

    # Write to stdout using the raw byte buffer to avoid Windows console
    # encoding issues.  The Windows console uses cp1252 by default, which
    # can't encode all Unicode characters (including the U+FFFD replacement
    # character).  Writing UTF-8 bytes directly to stdout.buffer bypasses
    # the codec layer entirely.
    sys.stdout.buffer.write(decoded.encode("utf-8", errors="replace"))
    sys.stdout.buffer.write(b"\n")
    sys.stdout.buffer.flush()
    print(f"\n--- End of plaintext ---")

    # --- Save plaintext to file ---
    out_path = os.path.join(SCRIPT_DIR, "plaintext.txt")
    with open(out_path, "wb") as f:
        f.write(text)
    print(f"\nPlaintext written to {out_path}")

    # --- Hex dump for forensic inspection ---
    # This shows every byte of the raw decryption output (before padding
    # removal), in the traditional hex dump format: offset, hex bytes, ASCII.
    # Non-printable bytes are shown as '.' in the ASCII column.
    print(f"\n--- Hex dump of full decryption (before stripping) ---")
    for i in range(0, len(plaintext), 16):
        chunk = plaintext[i:i+16]
        hex_str = " ".join(f"{b:02x}" for b in chunk)
        asc_str = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        print(f"  {i:04x}: {hex_str:<48s} {asc_str}")


# =============================================================================
# MAIN — RSA DECRYPTION + KEY RECOVERY + DES DECRYPTION
# =============================================================================
def main():
    # =========================================================================
    # STEP 0: Parse command-line arguments (the two prime factors)
    # =========================================================================
    # The factors p and q are provided as command-line arguments.  They were
    # obtained by running CADO-NFS (General Number Field Sieve) on the modulus N.
    # CADO-NFS outputs the factors at the very end of its run:
    #   "Square Root: Factors: <p> <q>"
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <p> <q>")
        print("  p, q = prime factors of N from CADO-NFS")
        sys.exit(1)

    p = int(sys.argv[1])
    q = int(sys.argv[2])

    # Canonicalize: ensure p < q for consistent output.
    if p > q:
        p, q = q, p

    print("=" * 60)
    print("STAGE 10: RSA DECRYPTION")
    print("=" * 60)

    # =========================================================================
    # STEP 1: VERIFY THE FACTORIZATION
    # =========================================================================
    # Before using p and q, verify that they're actually the correct factors.
    # A typo in either factor would silently produce a wrong RSA private key,
    # and every subsequent decryption attempt would fail.
    print(f"\np = {p}")
    print(f"  ({len(str(p))} digits, {p.bit_length()} bits)")
    print(f"q = {q}")
    print(f"  ({len(str(q))} digits, {q.bit_length()} bits)")

    # Verify that p * q equals the known modulus N
    assert p * q == N, "FATAL: p * q != N"
    print("\np * q == N: VERIFIED")

    # Verify that p and q are actually prime (not just any two factors)
    # using Miller-Rabin with 20 rounds — probability of error < 10^(-12)
    assert is_probable_prime(p), "FATAL: p is not prime"
    assert is_probable_prime(q), "FATAL: q is not prime"
    print("p is prime: VERIFIED (Miller-Rabin, 20 rounds)")
    print("q is prime: VERIFIED (Miller-Rabin, 20 rounds)")

    # =========================================================================
    # STEP 2: COMPUTE THE RSA PRIVATE EXPONENT d
    # =========================================================================
    # The RSA private exponent d is the modular multiplicative inverse of e
    # modulo phi(N), where phi(N) = (p-1)(q-1) is Euler's totient function.
    #
    # By Euler's theorem: M^(e*d) ≡ M (mod N) when e*d ≡ 1 (mod phi(N)).
    # This means: Decrypt(Encrypt(M)) = (M^e)^d = M^(e*d) = M (mod N).
    #
    # Without knowing p and q, computing phi(N) is as hard as factoring N.
    # This is why RSA security rests entirely on the difficulty of factoring.
    phi_N = (p - 1) * (q - 1)

    # Sanity check: e must be coprime to phi(N) for the inverse to exist.
    # This is guaranteed by proper RSA key generation, but we verify anyway.
    g = gcd(e, phi_N)
    assert g == 1, f"FATAL: gcd(e, phi(N)) = {g}, must be 1"
    print(f"\ngcd(e, phi(N)) = 1: VERIFIED")

    # Compute d = e^(-1) mod phi(N) using Python's built-in modular inverse.
    # Python 3.8+ supports pow(e, -1, phi_N) for modular inverse.
    # Internally, this uses the Extended Euclidean Algorithm, which is fast.
    d = pow(e, -1, phi_N)

    # Double-check: d * e ≡ 1 (mod phi(N))
    assert (d * e) % phi_N == 1, "FATAL: d * e != 1 mod phi(N)"
    print(f"Private exponent d computed ({d.bit_length()} bits)")

    # =========================================================================
    # STEP 3: RSA DECRYPTION — Recover the symmetric key M
    # =========================================================================
    # The core RSA decryption: M = C^d mod N.
    #
    # This computes modular exponentiation of a ~512-bit base to a ~511-bit
    # exponent modulo a 512-bit modulus.  Python's built-in pow() uses
    # square-and-multiply, which is efficient (O(log d) multiplications).
    # On modern hardware, this completes in milliseconds.
    M = pow(C, d, N)
    print(f"\nRSA plaintext M = {M}")
    print(f"  ({len(str(M))} digits, {M.bit_length()} bits)")

    # Convert M to bytes.  The byte representation tells us the key.
    # The "minimal" representation is the shortest byte string that encodes M.
    M_byte_len = (M.bit_length() + 7) // 8
    print(f"  Byte length (minimal): {M_byte_len}")
    print(f"  Hex: {M:0{M_byte_len*2}x}")

    # Also show M zero-padded to 64 bytes (the full RSA block size for 512-bit N)
    # in case PKCS#1 v1.5 padding was used (which starts with 0x00 0x02).
    M_bytes_64 = M.to_bytes(64, "big")
    print(f"  Full 64 bytes: {M_bytes_64.hex()}")

    # Verify the decryption by re-encrypting: M^e mod N should equal C.
    # This catches any computational errors in the factoring or key derivation.
    assert pow(M, e, N) == C, "FATAL: M^e mod N != C"
    print("Round-trip check M^e mod N == C: VERIFIED")

    # =========================================================================
    # STEP 4: EXTRACT CANDIDATE SYMMETRIC KEYS FROM M
    # =========================================================================
    # We don't know in advance:
    #   a) Whether the symmetric cipher is DES (8-byte key), 2-key Triple-DES
    #      (16-byte key), or 3-key Triple-DES (24-byte key).
    #   b) How the key is embedded in M — is it the entire value of M?  The
    #      low bytes?  The high bytes?  PKCS#1 v1.5 padded?
    #
    # So we generate ALL plausible candidate keys and try each one.  The
    # correct key will produce readable English text; wrong keys will produce
    # gibberish (with ~0% chance of a false positive).
    #
    # In our case, M turned out to be 16 bytes: 462c6bad381031ad 64dccd9820a751a2
    # The correct key is the FIRST 8 bytes (462c6bad381031ad), used with
    # single DES ECB.  The README's mention of Triple-DES was misleading!
    print("\n" + "=" * 60)
    print("KEY EXTRACTION AND DECRYPTION ATTEMPTS")
    print("=" * 60)

    # Import DES and Triple-DES cipher implementations from pycryptodome
    try:
        from Crypto.Cipher import DES3, DES
    except ImportError:
        print("ERROR: pycryptodome not installed. Run: pip install pycryptodome")
        sys.exit(1)

    # Build the minimal big-endian byte representation of M.
    # "Minimal" means no leading zero bytes.
    M_bytes_min = M.to_bytes(M_byte_len, "big")

    # We'll collect all candidate (description, key_bytes, cipher_module, label)
    # tuples, then de-duplicate and try each one.
    candidates = []

    # -------------------------------------------------------------------------
    # APPROACH A: TEXTBOOK RSA (no padding — M is the raw key value)
    # -------------------------------------------------------------------------
    # In textbook RSA, M is encrypted directly as an integer: C = M^e mod N.
    # The key could be:
    #   - The low (least significant) bytes of M
    #   - The high (most significant) bytes of M
    #   - M itself (if it happens to be exactly the right key length)
    #   - M zero-padded to the key length

    # A1: Single DES — the low 8 bytes of M (least significant bytes)
    # For M = 0x462c6bad381031ad64dccd9820a751a2, low 8 bytes = 64dccd9820a751a2
    if M_byte_len >= 8:
        key8 = M_bytes_min[-8:]
        candidates.append(("Textbook RSA, low 8 bytes", key8, DES, "Single DES"))
    # If M fits in 8 bytes, also try it zero-padded to exactly 8 bytes
    if M_byte_len <= 8:
        key8 = M.to_bytes(8, "big")
        candidates.append(("Textbook RSA, M as 8 bytes", key8, DES, "Single DES"))

    # A2: 2-key Triple-DES — the low 16 bytes of M
    # 2-key 3DES uses K1|K2 (16 bytes), with K3=K1 for EDE operations
    if M_byte_len >= 16:
        key16 = M_bytes_min[-16:]
        candidates.append(("Textbook RSA, low 16 bytes", key16, DES3, "2-key 3DES"))
    if M_byte_len <= 16:
        key16 = M.to_bytes(16, "big")
        candidates.append(("Textbook RSA, M as 16 bytes", key16, DES3, "2-key 3DES"))

    # A3: 3-key Triple-DES — the low 24 bytes of M
    # 3-key 3DES uses K1|K2|K3 (24 bytes), all independent subkeys
    if M_byte_len >= 24:
        key24 = M_bytes_min[-24:]
        candidates.append(("Textbook RSA, low 24 bytes", key24, DES3, "3-key 3DES"))
    if M_byte_len <= 24:
        key24 = M.to_bytes(24, "big")
        candidates.append(("Textbook RSA, M as 24 bytes", key24, DES3, "3-key 3DES"))

    # A4: If M happens to be exactly 8, 16, or 24 bytes, it might be the key
    if M_byte_len == 8:
        candidates.append(("M is exactly 8 bytes", M_bytes_min, DES, "Single DES"))
    elif M_byte_len == 16:
        candidates.append(("M is exactly 16 bytes", M_bytes_min, DES3, "2-key 3DES"))
    elif M_byte_len == 24:
        candidates.append(("M is exactly 24 bytes", M_bytes_min, DES3, "3-key 3DES"))

    # A5: The HIGH (most significant / first) bytes of M
    # *** THIS IS THE ONE THAT WORKED! ***
    # For M = 0x462c6bad381031ad64dccd9820a751a2, first 8 bytes = 462c6bad381031ad
    # Using these as a single DES key produced 99% printable English text.
    if M_byte_len >= 8:
        candidates.append(("First 8 bytes of M", M_bytes_min[:8], DES, "Single DES"))
    if M_byte_len >= 16:
        candidates.append(("First 16 bytes of M", M_bytes_min[:16], DES3, "2-key 3DES"))
    if M_byte_len >= 24:
        candidates.append(("First 24 bytes of M", M_bytes_min[:24], DES3, "3-key 3DES"))

    # -------------------------------------------------------------------------
    # APPROACH B: PKCS#1 v1.5 PADDED RSA
    # -------------------------------------------------------------------------
    # PKCS#1 v1.5 is the standard RSA padding scheme (RFC 2313).  The padded
    # message has the format:
    #   0x00 || 0x02 || <random non-zero padding bytes> || 0x00 || <key>
    #
    # When M is encoded in a full 64-byte (512-bit) block, the first two bytes
    # would be 0x00 0x02 for PKCS#1 type 2 (encryption).  We check for this
    # pattern and, if found, extract the key payload after the 0x00 separator.
    #
    # In our case, M starts with 0x00 0x00 (lots of leading zeros), so this
    # is NOT PKCS#1 v1.5 padding.  But we check anyway for completeness.
    if M_bytes_64[0] == 0x00 and M_bytes_64[1] == 0x02:
        try:
            # Find the 0x00 separator byte after the random padding
            sep_idx = M_bytes_64.index(0x00, 2)
            # Everything after the separator is the actual key payload
            payload = M_bytes_64[sep_idx + 1:]
            print(f"\nPKCS#1 v1.5 detected: payload is {len(payload)} bytes")
            print(f"  Payload hex: {payload.hex()}")

            # Try the payload as various key sizes
            if len(payload) >= 8:
                candidates.append(("PKCS#1 payload, first 8 bytes", payload[:8], DES, "Single DES"))
            if len(payload) >= 16:
                candidates.append(("PKCS#1 payload, first 16 bytes", payload[:16], DES3, "2-key 3DES"))
            if len(payload) >= 24:
                candidates.append(("PKCS#1 payload, first 24 bytes", payload[:24], DES3, "3-key 3DES"))
            # If the payload is exactly a key size, that's the most likely match
            if len(payload) == 8:
                candidates.append(("PKCS#1 payload (exact 8 bytes)", payload, DES, "Single DES"))
            elif len(payload) == 16:
                candidates.append(("PKCS#1 payload (exact 16 bytes)", payload, DES3, "2-key 3DES"))
            elif len(payload) == 24:
                candidates.append(("PKCS#1 payload (exact 24 bytes)", payload, DES3, "3-key 3DES"))

            # Also try last N bytes of payload (in case it's right-aligned)
            if len(payload) > 8:
                candidates.append(("PKCS#1 payload, last 8 bytes", payload[-8:], DES, "Single DES"))
            if len(payload) > 16:
                candidates.append(("PKCS#1 payload, last 16 bytes", payload[-16:], DES3, "2-key 3DES"))
            if len(payload) > 24:
                candidates.append(("PKCS#1 payload, last 24 bytes", payload[-24:], DES3, "3-key 3DES"))

        except ValueError:
            print("\nPKCS#1 v1.5 header detected but no 0x00 separator found")
    else:
        # Our case: first bytes are 0x00 0x00, not 0x00 0x02, so no PKCS#1
        print(f"\nNot PKCS#1 v1.5 (first two bytes: 0x{M_bytes_64[0]:02x} 0x{M_bytes_64[1]:02x})")

    # -------------------------------------------------------------------------
    # APPROACH C: LITTLE-ENDIAN BYTE ORDER
    # -------------------------------------------------------------------------
    # Some systems store integers in little-endian byte order (least significant
    # byte first).  While big-endian is standard for RSA, we try little-endian
    # just in case, since the challenge doesn't specify byte ordering.
    M_bytes_le = M_bytes_min[::-1]  # Reverse the byte order
    if len(M_bytes_le) >= 8:
        candidates.append(("Little-endian M, first 8 bytes", M_bytes_le[:8], DES, "Single DES (LE)"))
    if len(M_bytes_le) >= 16:
        candidates.append(("Little-endian M, first 16 bytes", M_bytes_le[:16], DES3, "2-key 3DES (LE)"))

    # -------------------------------------------------------------------------
    # DE-DUPLICATE CANDIDATES
    # -------------------------------------------------------------------------
    # Multiple approaches may produce the same key bytes.  For example, if M is
    # exactly 8 bytes, then "low 8 bytes", "first 8 bytes", and "M as 8 bytes"
    # are all identical.  We de-duplicate to avoid testing the same key twice.
    seen = set()
    unique_candidates = []
    for desc, key, cipher_mod, key_type in candidates:
        # Use (hex_key, cipher_name) as the dedup key — same key bytes with
        # different cipher modules (DES vs DES3) should both be tested.
        key_id = (key.hex(), cipher_mod.__name__)
        if key_id not in seen:
            seen.add(key_id)
            unique_candidates.append((desc, key, cipher_mod, key_type))

    print(f"\nTotal unique key candidates to try: {len(unique_candidates)}")

    # =========================================================================
    # STEP 5: TRY EACH CANDIDATE KEY — DECRYPT AND CHECK FOR ENGLISH
    # =========================================================================
    # For each candidate key, we:
    #   1. Attempt DES or Triple-DES ECB decryption of the 200-byte ciphertext
    #   2. Score the resulting plaintext for English-ness
    #   3. If it looks like English (>85% printable, >40% common letters),
    #      display the result and stop — we found the answer!
    #
    # Wrong keys produce pseudorandom output (~37% printable), so there's
    # effectively zero chance of a false positive with our thresholds.
    best_result = None  # Track the best non-English result for debugging
    best_pct = 0

    for desc, key, cipher_mod, key_type in unique_candidates:
        print(f"\n--- {desc} ({key_type}) ---")
        print(f"    Key ({len(key)} bytes): {key.hex()}")

        result = try_decrypt(cipher_mod, key, ct_data, desc)
        if result is None:
            continue  # Cipher rejected the key (wrong size, weak key, etc.)

        plaintext, pct, is_eng = result
        print(f"    Printable: {pct:.1f}%  English: {'YES' if is_eng else 'no'}")

        if is_eng:
            # *** FOUND IT! ***
            # This key produced English text.  Display the full result and exit.
            display_result(plaintext, key, desc, key_type)
            return

        # Track the best non-English result in case no key matches perfectly
        if pct > best_pct:
            best_pct = pct
            best_result = (plaintext, key, desc, key_type)

        # For partial matches (>30% printable), show a preview for debugging
        if pct > 30:
            asc = "".join(chr(b) if 32 <= b < 127 else "." for b in plaintext[:48])
            print(f"    First 48 bytes: {asc}")

    # =========================================================================
    # FALLBACK: No strong English match found
    # =========================================================================
    # If no candidate produced a clear English match, display the best one
    # for manual inspection.  This shouldn't happen with correct factors,
    # but provides debugging output if something went wrong.
    if best_result:
        plaintext, key, desc, key_type = best_result
        print(f"\n{'=' * 60}")
        print(f"No strong English match found.")
        print(f"Best result: {desc} ({key_type}) at {best_pct:.1f}% printable")
        print(f"{'=' * 60}")
        display_result(plaintext, key, desc, key_type)
    else:
        print("\n*** No key produced readable plaintext. ***")
        print("Check the RSA decryption or try different key extraction methods.")


if __name__ == "__main__":
    main()
