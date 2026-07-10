"""Fetch reference randomness from Cloudflare drand, ANU QRNG, and Geiger files."""

import os
import sys
import time
from pathlib import Path

import numpy as np
import requests

from process_frames import bytes_to_uniform_floats

DRAND_URL = "https://drand.cloudflare.com"
DRAND_CHAIN = "/8990e7a9aaed2ffed73dbd7092123d6f289930540d7651336225dc172e51b2c/public"

ANU_PUBLIC_URL = "https://qrng.anu.edu.au/API/jsonI.php"
ANU_API_URL = "https://api.quantumnumbers.anu.edu.au"


def fetch_cloudflare_stream(n_bytes_needed: int) -> tuple[np.ndarray, bytes, list[int]]:
    """
    Fetch Cloudflare drand rounds for comparison with RT output.
    Returns (uniform_floats, raw_bytes, round_numbers).
    """
    n_rounds = max(1, (n_bytes_needed + 31) // 32)
    print(f"[cf]   Fetching {n_rounds} drand rounds ({n_rounds * 32} bytes)...")

    all_bytes = bytearray()
    rounds = []

    try:
        resp = requests.get(f"{DRAND_URL}/public/latest", timeout=8)
        resp.raise_for_status()
        latest = resp.json()
        latest_round = latest["round"]
        all_bytes.extend(bytes.fromhex(latest["randomness"]))
        rounds.append(latest_round)

        for i in range(1, n_rounds):
            r = latest_round - i
            if r < 1:
                break
            resp = requests.get(f"{DRAND_URL}/public/{r}", timeout=8)
            resp.raise_for_status()
            data = resp.json()
            all_bytes.extend(bytes.fromhex(data["randomness"]))
            rounds.append(r)
            if i % 50 == 0:
                print(f"[cf]   ... {i}/{n_rounds - 1} rounds fetched")
            time.sleep(0.01)

    except Exception as e:
        print(f"[cf]   Warning: {e}")
        print("[cf]   Falling back to OS random for Cloudflare comparison baseline.")
        all_bytes = bytearray(np.random.bytes(n_bytes_needed + 32))
        rounds = [-1]

    raw_bytes = bytes(all_bytes)[:n_bytes_needed]
    floats = bytes_to_uniform_floats(raw_bytes)
    print(f"[cf]   Got {len(floats)} samples from rounds {min(rounds)}–{max(rounds)}")
    return floats, raw_bytes, rounds


def _fetch_anu_public(n_bytes: int) -> bytes:
    """
    Fetch uint8 values from the free ANU QRNG public endpoint.

    The endpoint advertises up to 1024 values but returns HTTP 500 above
    ~100 in practice.  We use a conservative batch size of 100 and page.
    """
    ANU_BATCH = 100
    all_vals: list[int] = []
    remaining = n_bytes
    batch_num = 0
    while remaining > 0:
        length = min(ANU_BATCH, remaining)
        resp = requests.get(
            ANU_PUBLIC_URL,
            params={"length": length, "type": "uint8"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("success"):
            raise RuntimeError(f"ANU API returned success=false: {data}")
        all_vals.extend(data["data"])
        remaining -= length
        batch_num += 1
        if remaining > 0:
            if batch_num % 10 == 0:
                print(f"[anu]  ... {len(all_vals)}/{n_bytes} bytes fetched")
            time.sleep(0.25)

    return bytes(all_vals[:n_bytes])


def _fetch_anu_qrandom_pkg(n_bytes: int) -> bytes:
    """
    Fallback: use the quantum-random package (requires QRANDOM_API_KEY).
    Fetches hex16 words and concatenates.
    """
    api_key = os.environ.get("QRANDOM_API_KEY", "")
    if not api_key:
        try:
            import configparser
            import pathlib
            cfg_path = pathlib.Path.home() / ".config" / "qrandom" / "qrandom.ini"
            if cfg_path.exists():
                cp = configparser.ConfigParser()
                cp.read(cfg_path)
                api_key = cp["default"]["key"]
        except Exception:
            pass

    if not api_key:
        raise RuntimeError(
            "quantum-random package requires QRANDOM_API_KEY env var "
            "or `qrandom-init` to be run first."
        )

    n_hex_words = max(1, (n_bytes + 3) // 4)
    batch_size = min(1024, n_hex_words)
    collected = bytearray()

    fetched = 0
    while fetched < n_hex_words:
        this_batch = min(batch_size, n_hex_words - fetched)
        resp = requests.get(
            ANU_API_URL,
            params={"length": this_batch, "type": "hex16", "size": 4},
            headers={"x-api-key": api_key},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("success"):
            raise RuntimeError(f"ANU API v2 returned success=false: {data}")
        for hex_word in data["data"]:
            collected.extend(bytes.fromhex(hex_word))
        fetched += this_batch
        if fetched < n_hex_words:
            time.sleep(0.1)

    return bytes(collected[:n_bytes])


def fetch_anu_qrng(n_bytes_needed: int) -> tuple[np.ndarray, bytes]:
    """
    Fetch quantum random bytes from ANU QRNG.
    Tries the public endpoint first; falls back to the quantum-random package.
    Returns (uniform_floats, raw_bytes).
    """
    print(f"[anu]  Fetching {n_bytes_needed} bytes of quantum randomness from ANU QRNG...")

    raw_bytes: bytes | None = None

    try:
        raw_bytes = _fetch_anu_public(n_bytes_needed)
        print(f"[anu]  Got {len(raw_bytes)} bytes from ANU public endpoint.")
    except Exception as e:
        print(f"[anu]  Public endpoint failed: {e}")

    if raw_bytes is None:
        print("[anu]  Trying quantum-random package (needs QRANDOM_API_KEY) ...")
        try:
            raw_bytes = _fetch_anu_qrandom_pkg(n_bytes_needed)
            print(f"[anu]  Got {len(raw_bytes)} bytes via quantum-random package.")
        except Exception as e2:
            print(f"[anu]  quantum-random fallback failed: {e2}")

    if raw_bytes is None:
        print("[anu]  WARNING: All ANU sources unavailable — using OS random as placeholder.")
        raw_bytes = np.random.bytes(n_bytes_needed)

    floats = bytes_to_uniform_floats(raw_bytes)
    return floats, raw_bytes


def load_geiger_bytes(path: str) -> tuple[np.ndarray, bytes]:
    """
    Load uint8 values from a Geiger counter text file (one integer per line).
    Values must be in [0, 255]. Lines starting with '#' are treated as comments.

    Float conversion uses per-byte mapping (value / 256) rather than packing
    8 bytes into a uint64.  This preserves all N samples at 8-bit resolution,
    which is statistically correct for KS / entropy / autocorrelation tests and
    gives 8× more samples than the packed approach — important when the file is
    small.  The raw bytes are still passed to the bit-based NIST tests unchanged.

    Returns (uniform_floats, raw_bytes).
    """
    p = Path(path)
    if not p.exists():
        sys.exit(f"Geiger file not found: {path}")

    values: list[int] = []
    with open(p) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                v = int(line)
            except ValueError:
                print(f"[geig] Warning: skipping non-integer on line {lineno}: {line!r}")
                continue
            if not (0 <= v <= 255):
                print(f"[geig] Warning: value {v} out of uint8 range on line {lineno}, clamping.")
                v = max(0, min(255, v))
            values.append(v)

    if not values:
        sys.exit(f"No valid uint8 values found in {path}")

    raw_bytes = bytes(values)
    floats = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float64) / 256.0
    print(
        f"[geig] Loaded {len(values)} Geiger samples "
        f"→ {len(floats)} float64 values (per-byte, 8-bit resolution)."
    )
    return floats, raw_bytes
