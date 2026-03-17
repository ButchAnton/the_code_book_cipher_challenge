#!/usr/bin/env python3
"""
validate_inputs.py -- Parse and validate Stage 10 inputs, then try quick factoring.
====================================================================================

Simon Singh's Code Book Cipher Challenge — Stage 10 (the final stage)
---------------------------------------------------------------------
This is the last and hardest stage of the challenge published in Simon Singh's
book "The Code Book" (1999).  The prize was £10,000 for solving all ten stages.

Stage 10 uses a two-layer encryption scheme:
  Layer 1 — RSA public-key encryption protects the symmetric key.
            The RSA public key has a 512-bit (155-digit) modulus N and a
            deliberately humorous public exponent e = 0xDEADBEEF.
  Layer 2 — DES (or Triple-DES) symmetric-key encryption protects the
            actual plaintext message.  The DES key is the RSA-encrypted
            "shorter message."

To break Stage 10 one must:
  1. FACTOR the 512-bit RSA modulus N into its two prime factors p and q.
     This is the computationally hard part — it requires the General Number
     Field Sieve (GNFS), the fastest known algorithm for factoring large
     semiprimes.  I used CADO-NFS, an open-source GNFS implementation, and
     it took ~18.5 hours on a 64-core AMD Threadripper PRO 5995WX.
  2. COMPUTE the RSA private key d from the factors and decrypt the shorter
     message to recover the DES key.
  3. DECRYPT the longer message (text.d) with the recovered DES key.

This script handles the first step of the process: it parses and validates
all of the cryptographic inputs from the challenge files, runs a few quick
(but long-shot) factoring algorithms, and then writes N.txt for CADO-NFS.

Input files (from the book / companion website):
  - README                       — Challenge description with N and e
  - shorter_message_stage10.txt  — RSA ciphertext C (155-digit integer)
  - longer_message_stage10.uue   — UUencoded DES ciphertext
  - text.d                       — Raw binary DES ciphertext (decoded from .uue)

Output:
  - N.txt  — Plain decimal modulus N, ready for CADO-NFS input
"""

import os
import sys
from math import gcd, isqrt

# Resolve the directory containing this script, so file paths work regardless
# of the current working directory when the script is invoked.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# 1. RSA MODULUS N
# =============================================================================
# The RSA modulus N is given in the Stage 10 README.  It is a 155-digit
# (512-bit) semiprime — the product of exactly two large primes p and q.
# 512-bit RSA was considered secure in the late 1990s, but is now breakable
# with GNFS.  The RSA-155 challenge (also 512 bits) was first factored in
# 1999 by a large academic team.  Today, a single powerful workstation can
# do it in under 24 hours.
#
# N is hard-coded here rather than parsed from the README to avoid OCR /
# formatting issues.  The modulus in the README was originally scanned from
# the printed book.
N = int("10742788291266565907178411279942116612663921794753"
        "29458887781721035546415098012187903383292623528109"
        "07506720835049419964331434255583344018558089894268"
        "92463")

# =============================================================================
# 2. RSA PUBLIC EXPONENT e
# =============================================================================
# The public exponent e = 3,735,928,559 = 0xDEADBEEF.
# This is a playful choice by Simon Singh — "DEADBEEF" is a famous hexadecimal
# magic number used in debugging.  Normally e is chosen to be 65537 (0x10001)
# for efficiency, but any odd integer coprime to phi(N) works.  The larger e
# here makes RSA decryption slightly slower but doesn't affect security.
e = 3735928559  # 0xDEADBEEF

# =============================================================================
# 3. RSA CIPHERTEXT C
# =============================================================================
# The "shorter message" file contains the RSA ciphertext C — a large integer
# that was computed as C = M^e mod N, where M is the DES key (padded or raw).
# The file contains space-separated decimal digits which are joined and
# converted to an integer.
with open(os.path.join(SCRIPT_DIR, "shorter_message_stage10.txt"), "r") as f:
    c_text = f.read()
# Strip whitespace and newlines, then convert to integer.
# The original file uses spaces between digit groups for readability.
C = int(c_text.replace(" ", "").replace("\n", "").replace("\r", ""))

# =============================================================================
# 4. DES/TRIPLE-DES CIPHERTEXT (text.d)
# =============================================================================
# The "longer message" was UUencoded in longer_message_stage10.uue.
# This file was previously decoded it to text.d — a raw binary file of exactly
# 200 bytes.
# DES uses 8-byte (64-bit) blocks, so 200 bytes = 25 complete DES blocks.
# This is the actual encrypted English message to recover.
with open(os.path.join(SCRIPT_DIR, "text.d"), "rb") as f:
    ct_data = f.read()

# =============================================================================
# 5. VALIDATE EVERYTHING
# =============================================================================
# Before attempting the expensive factorization (hours of CPU time), verify
# that all inputs are self-consistent.  A single wrong digit in N or C would
# cause the entire factoring run to produce useless results.
print("=" * 60)
print("STAGE 10 INPUT VALIDATION")
print("=" * 60)

# --- Display N properties ---
print(f"\nRSA modulus N:")
print(f"  Decimal digits : {len(str(N))}")     # Should be 155
print(f"  Bit length     : {N.bit_length()}")   # Should be 512
print(f"  Is odd         : {N & 1 == 1}")       # Must be odd (product of two odd primes)
print(f"  N = {N}")

# --- Display e properties ---
print(f"\nPublic exponent e:")
print(f"  Decimal : {e}")
print(f"  Hex     : 0x{e:X}")
# gcd(e, N) must be 1 for RSA to work.  If it weren't, e would share a
# factor with N, which would trivially reveal a factor of N.
print(f"  gcd(e,N): {gcd(e, N)}")

# --- Display C properties ---
print(f"\nRSA ciphertext C:")
print(f"  Decimal digits : {len(str(C))}")
print(f"  Bit length     : {C.bit_length()}")
# C must be strictly less than N for RSA — ciphertext is computed mod N.
print(f"  C < N          : {C < N}")
# If gcd(C, N) != 1, we'd have accidentally found a factor of N for free!
# This almost never happens with properly generated RSA keys.
print(f"  gcd(C,N)       : {gcd(C, N)}")
print(f"  C = {C}")

# --- Display text.d properties ---
print(f"\nTriple-DES ciphertext (text.d):")
print(f"  Size           : {len(ct_data)} bytes")           # Should be 200
print(f"  Blocks (8-byte): {len(ct_data) // 8}")            # Should be 25
print(f"  Multiple of 8  : {len(ct_data) % 8 == 0}")        # Must be True for ECB
print(f"  First block    : {ct_data[:8].hex()}")             # For visual verification
print(f"  Last block     : {ct_data[-8:].hex()}")

# --- Run validation checks ---
# Collect all errors rather than failing on the first one, so the user can
# fix everything in a single pass.
errors = []
if len(str(N)) != 155:
    errors.append(f"N should be 155 digits, got {len(str(N))}")
if N.bit_length() != 512:
    errors.append(f"N should be 512 bits, got {N.bit_length()}")
if not (N & 1):
    # If N is even, one factor is 2 — trivially factored!
    errors.append("N is even!")
if not (C < N):
    # RSA ciphertext must be in the range [0, N-1]
    errors.append("C >= N!")
if gcd(e, N) != 1:
    errors.append(f"gcd(e,N) = {gcd(e,N)}, should be 1")
if gcd(C, N) != 1:
    # This would actually be GREAT news — we'd have a factor of N!
    # But with proper RSA keys, this essentially never happens.
    errors.append(f"gcd(C,N) = {gcd(C,N)}, should be 1 (if not, I found a factor!)")
if len(ct_data) != 200:
    errors.append(f"text.d should be 200 bytes, got {len(ct_data)}")
if len(ct_data) % 8 != 0:
    # DES ECB requires ciphertext to be a multiple of the 8-byte block size
    errors.append(f"text.d size not a multiple of 8")

if errors:
    print("\n*** ERRORS ***")
    for err in errors:
        print(f"  - {err}")
    sys.exit(1)
else:
    print("\n*** ALL CHECKS PASSED ***")

# =============================================================================
# 6. WRITE N.txt FOR CADO-NFS
# =============================================================================
# CADO-NFS accepts the number to factor as a command-line argument or from a
# file.  I write N as a plain decimal string so it can be easily copy/pasted
# into the CADO-NFS command:
#   ./cado-nfs.py $(cat N.txt) -t all
n_path = os.path.join(SCRIPT_DIR, "N.txt")
with open(n_path, "w") as f:
    f.write(str(N) + "\n")
print(f"\nWrote N to {n_path}")

# =============================================================================
# 7. QUICK FACTORING: TRIAL DIVISION
# =============================================================================
# Before investing hours in GNFS, try three simple factoring methods that
# are essentially free (seconds of CPU time).  They won't work on a properly
# generated RSA modulus — both primes will be too large and too "random" — but
# it's always worth checking in case the challenge designer made them easy.
#
# Trial division: divide N by every odd number up to 10^7.  If either prime
# factor were below 10^7, this would find it instantly.  Cryptographic primes
# are ~256 bits (77+ digits), so this has zero chance, but it's a good sanity
# check and takes only seconds.
print("\n" + "=" * 60)
print("QUICK FACTORING ATTEMPTS")
print("=" * 60)

print("\n--- Trial division up to 10^7 ---")


def trial_division(n, limit):
    """
    Try dividing n by every small integer up to 'limit'.

    If n has a small prime factor, this will find it.  The time complexity
    is O(limit), which is fast for limit <= 10^7.  Cryptographic primes are
    far larger than 10^7, so this is just a sanity check.

    Args:
        n:     The integer to factor.
        limit: The largest trial divisor to try.

    Returns:
        The smallest prime factor found, or None if none exists up to limit.
    """
    # Check divisibility by 2 first (handles the even case)
    if n % 2 == 0:
        return 2
    # Then try odd divisors 3, 5, 7, 9, ...  (don't need to skip
    # composites — if n is divisible by 9, it's also divisible by 3,
    # which was already checked.)
    d = 3
    while d <= limit:
        if n % d == 0:
            return d
        d += 2  # Skip even numbers
    return None


factor = trial_division(N, 10**7)
if factor:
    print(f"  FACTOR FOUND: {factor}")
    print(f"  Other factor: {N // factor}")
else:
    print(f"  No factor found up to 10^7.")

# =============================================================================
# 8. QUICK FACTORING: POLLARD'S RHO (BRENT VARIANT)
# =============================================================================
# Pollard's rho is a probabilistic factoring algorithm that finds a factor of
# n in expected O(n^(1/4)) time.  For a 512-bit N whose smallest factor is
# ~256 bits, this means ~O(2^64) iterations — far too many to be practical.
# But if either factor happened to be small-ish (< ~60 bits), rho would find
# it.  Try 5 different random seeds to improve our chances.
#
# The "Brent variant" uses Floyd's cycle-detection but with a different
# stepping strategy that's slightly more efficient in practice.
print("\n--- Pollard's rho (Brent variant), 5 seeds, 10^7 iterations each ---")


def pollard_rho_brent(n, seed=2, max_iter=10**7):
    """
    Pollard's rho factoring with Brent's cycle detection.

    This algorithm works by iterating a pseudo-random function f(x) = x^2 + c
    modulo n and looking for a cycle.  When two iterates x_i and x_j collide
    modulo a prime factor p of n (but not modulo n itself), then gcd(x_i - x_j, n)
    reveals p.  The birthday paradox ensures this happens after ~O(sqrt(p))
    iterations on average.

    For a 256-bit prime factor, sqrt(p) ~ 2^128, which is way beyond 10^7
    iterations.  So this is purely a "what if they made it easy?" check.

    Args:
        n:        The integer to factor.
        seed:     Starting value and polynomial constant c.
        max_iter: Maximum number of iterations before giving up.

    Returns:
        A non-trivial factor of n, or None if none is found within max_iter.
    """
    if n % 2 == 0:
        return 2
    x = seed       # "tortoise" — the slow iterator
    y = seed       # "hare" — the fast iterator (advances 2 steps per round)
    c = seed       # Constant in the polynomial f(x) = x^2 + c
    d = 1          # Will hold gcd(|x - y|, n)
    while d == 1:
        # Advance x by one step: x = f(x)
        x = (x * x + c) % n
        # Advance y by two steps: y = f(f(y))
        y = (y * y + c) % n
        y = (y * y + c) % n
        # Check if the difference reveals a factor
        d = gcd(abs(x - y), n)
        max_iter -= 1
        if max_iter <= 0:
            return None  # Gave up — too many iterations
    if d != n:
        return d   # Found a non-trivial factor!
    return None    # d == n means the algorithm "failed" (unlucky seed)


# Try multiple seeds — different seeds explore different random walks,
# giving independent chances of finding a factor.
for seed in [2, 3, 5, 7, 11]:
    factor = pollard_rho_brent(N, seed=seed, max_iter=10**7)
    if factor and factor != N:
        print(f"  FACTOR FOUND (seed={seed}): {factor}")
        print(f"  Other factor: {N // factor}")
        break
    else:
        print(f"  Seed {seed}: no factor found.")
else:
    # The for/else construct: the else block runs only if the loop
    # completed WITHOUT hitting a 'break' — i.e., no factor was found.
    print("  Pollard's rho found nothing. GNFS required.")

# =============================================================================
# 9. QUICK FACTORING: POLLARD'S p-1
# =============================================================================
# Pollard's p-1 method exploits Fermat's little theorem: for any a coprime
# to p, a^(p-1) ≡ 1 (mod p).  If p-1 is "B-smooth" (all prime factors of
# p-1 are <= B), then a^(B!) ≡ 1 (mod p), but probably NOT ≡ 1 (mod q).
# So gcd(a^(B!) - 1, N) reveals p.
#
# This works when p-1 has only small prime factors — which is why cryptographic
# prime generation specifically avoids this by requiring p-1 to have at least
# one large prime factor.  But again, worth checking in case the challenge
# primes have this weakness.
print("\n--- Pollard's p-1 (B=10^6) ---")


def pollard_p_minus_1(n, B=10**6):
    """
    Pollard's p-1 factoring method.

    If one of n's prime factors p is such that (p-1) is B-smooth (meaning
    all prime factors of p-1 are at most B), then this method will find p.

    The algorithm computes a^(lcm(1,2,...,B)) mod n by iteratively raising
    a to each prime power p^k <= B.  If the result shares a factor with n
    (but isn't n itself), we've found a non-trivial factor.

    Cryptographic primes are specifically chosen so that p-1 has a large
    prime factor, making this attack fail.  Try it anyway as a free check.

    Args:
        n: The integer to factor.
        B: Smoothness bound — check if (p-1) is B-smooth.

    Returns:
        A non-trivial factor of n, or None.
    """
    a = 2  # Base for exponentiation (any a > 1 works)
    # Iterate through all potential prime powers up to B.
    # For each prime p, find the largest power p^k <= B, then compute
    # a = a^(p^k) mod n.  After processing all primes up to B, if
    # (p_factor - 1) divides lcm(1..B), then a ≡ 1 (mod p_factor).
    p = 2
    while p <= B:
        # Find the largest power of p that doesn't exceed B.
        # For example, if p=2 and B=10^6, then pk = 2^19 = 524288.
        pk = p
        while pk * p <= B:
            pk *= p
        # Raise a to the pk-th power modulo n.
        # After all primes are processed, a = 2^(lcm(1..B)) mod n.
        a = pow(a, pk, n)
        # Check if we've found a factor.
        # If (p_factor - 1) | lcm(1..B), then a ≡ 1 (mod p_factor),
        # so gcd(a - 1, n) will be divisible by p_factor.
        d = gcd(a - 1, n)
        if 1 < d < n:
            return d      # Non-trivial factor found!
        if d == n:
            return None   # Unlucky: a ≡ 1 (mod n), meaning both factors divide
        # Advance to the next candidate prime.
        # After 2, skip even numbers (they can't be prime).
        p += 1 + (p > 2)  # +1 if p==2 (next is 3), +2 otherwise (skip evens)

    # Final check after processing all primes up to B
    d = gcd(a - 1, n)
    if 1 < d < n:
        return d
    return None


factor = pollard_p_minus_1(N, B=10**6)
if factor:
    print(f"  FACTOR FOUND: {factor}")
    print(f"  Other factor: {N // factor}")
else:
    print(f"  No factor found. p-1 is not 10^6-smooth.")

# =============================================================================
# CONCLUSION
# =============================================================================
# If I reach here without finding a factor (expected for crypto-grade primes),
# the only option is the General Number Field Sieve (GNFS).  CADO-NFS is the
# tool used for this — a state-of-the-art open-source GNFS implementation.
#
# To run the factorization:
#   cd ~/cado-nfs
#   ./cado-nfs.py $(cat /path/to/N.txt) -t all
#
# On my 64-core AMD Threadripper PRO 5995WX with 512GB RAM, the GNFS
# factorization completed in approximately 18.5 hours, going through these
# phases:
#   1. Polynomial Selection (size-optimized):  ~47 min
#   2. Polynomial Selection (root-optimized):  ~5 min
#   3. Lattice Sieving (collecting relations):  ~14 hours (the bottleneck!)
#   4. Filtering (duplicate removal, merging):  ~30 min
#   5. Linear Algebra (Krylov + lingen + mksol): ~2.5 hours
#   6. Square Root (extracting the factors):    ~3 min
#
# The resulting factors p and q are then fed into solve_stage10.py to recover
# the DES key and decrypt the final message.
print("\n" + "=" * 60)
print("DONE. If no factor found, proceed with CADO-NFS (GNFS).")
print("=" * 60)
