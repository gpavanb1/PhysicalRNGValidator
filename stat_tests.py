"""NIST SP 800-22 inspired statistical tests for randomness validation."""

import numpy as np
from scipy import stats

from process_frames import bytes_to_bits


def test_frequency(bits: np.ndarray) -> dict:
    """
    NIST Frequency (Monobit) Test.
    Tests whether number of 1s and 0s is approximately equal.
    """
    n = len(bits)
    if n == 0:
        return {"test": "Frequency (Monobit)", "statistic": None, "p_value": None,
                "pass": None, "note": "No bits available"}
    s = np.sum(bits) * 2 - n
    s_obs = abs(s) / np.sqrt(n)
    from scipy.special import erfc
    p_value = erfc(s_obs / np.sqrt(2))
    return {"test": "Frequency (Monobit)", "statistic": s_obs, "p_value": p_value,
            "pass": p_value >= 0.01, "n_bits": n}


def test_runs(bits: np.ndarray) -> dict:
    """
    NIST Runs Test.
    Tests whether oscillation between 0s and 1s is too fast or slow.
    """
    n = len(bits)
    if n == 0:
        return {"test": "Runs", "statistic": None, "p_value": None,
                "pass": None, "note": "No bits available"}
    pi = np.mean(bits)
    if abs(pi - 0.5) >= 2 / np.sqrt(n):
        return {"test": "Runs", "statistic": None, "p_value": 0.0,
                "pass": False, "note": "Pre-test failed (frequency too far from 0.5)"}

    v_obs = 1 + np.sum(bits[:-1] != bits[1:])
    numerator = abs(v_obs - 2 * n * pi * (1 - pi))
    denominator = 2 * np.sqrt(2 * n) * pi * (1 - pi)

    from scipy.special import erfc
    p_value = erfc(numerator / denominator)
    return {"test": "Runs", "statistic": v_obs, "p_value": p_value,
            "pass": p_value >= 0.01, "n_bits": n}


def test_block_frequency(bits: np.ndarray, block_size: int = 128) -> dict:
    """
    NIST Block Frequency Test.
    Tests frequency within non-overlapping blocks.
    """
    n = len(bits)
    n_blocks = n // block_size
    if n_blocks < 10:
        return {"test": "Block Frequency", "p_value": None,
                "pass": None, "note": "Too few bits for block test"}

    blocks = bits[:n_blocks * block_size].reshape(n_blocks, block_size)
    proportions = blocks.mean(axis=1)
    chi_sq = 4 * block_size * np.sum((proportions - 0.5)**2)
    from scipy.stats import chi2
    p_value = chi2.sf(chi_sq, df=n_blocks)
    return {"test": f"Block Frequency (M={block_size})", "statistic": chi_sq,
            "p_value": p_value, "pass": p_value >= 0.01, "n_blocks": n_blocks}


def test_serial(bits: np.ndarray) -> dict:
    """
    NIST Serial Test (2-bit patterns).
    Simplified for smaller datasets.
    """
    n = len(bits)
    if n < 128:
        return {"test": "Serial", "p_value": None, "pass": None, "note": "Too few bits"}

    v1 = np.bincount(bits, minlength=2)
    v2 = np.bincount(bits[:-1] * 2 + bits[1:], minlength=4)
    v3 = np.bincount(bits[:-2] * 4 + bits[1:-1] * 2 + bits[2:], minlength=8)

    psi1 = (2/n) * np.sum(v1**2) - n
    psi2 = (4/n) * np.sum(v2**2) - n
    psi3 = (8/n) * np.sum(v3**2) - n

    del1 = psi2 - psi1
    del2 = psi3 - 2*psi2 + psi1

    from scipy.special import gammaincc
    p1 = gammaincc(1/2, del1/2)
    p2 = gammaincc(1, del2/2)

    p_value = min(p1, p2)
    return {"test": "Serial", "statistic": del1, "p_value": p_value,
            "pass": p_value >= 0.01, "n_bits": n}


def test_entropy(floats: np.ndarray) -> dict:
    """
    Approximate entropy via histogram-based Shannon entropy.
    """
    if len(floats) == 0:
        return {"test": "Shannon Entropy", "entropy_bits": None,
                "max_bits": None, "efficiency_pct": None,
                "pass": None, "note": "No samples available"}

    n = len(floats)
    bins = max(2, min(256, n // 20))
    hist, _ = np.histogram(floats, bins=bins, range=(0, 1))
    hist_norm = hist[hist > 0] / hist.sum()
    h = -np.sum(hist_norm * np.log2(hist_norm))
    h_max = np.log2(bins)
    efficiency = h / h_max

    threshold = 0.98 if n < 2000 else 0.99
    return {"test": "Shannon Entropy", "entropy_bits": h,
            "max_bits": h_max, "efficiency_pct": efficiency * 100,
            "pass": efficiency > threshold}


def test_ks_uniformity(floats: np.ndarray) -> dict:
    """
    Kolmogorov-Smirnov test for uniformity on [0,1].
    """
    if len(floats) == 0:
        return {"test": "KS Uniformity", "statistic": None, "p_value": None,
                "pass": None, "note": "No samples available"}
    stat, p_value = stats.kstest(floats, 'uniform')
    return {"test": "KS Uniformity", "statistic": stat, "p_value": p_value,
            "pass": p_value >= 0.01, "n_samples": len(floats)}


def test_autocorrelation(floats: np.ndarray, lag: int = 1) -> dict:
    """
    Lag-1 autocorrelation — should be near zero for random sequence.
    """
    n = len(floats)
    if n < 50:
        return {"test": "Autocorrelation", "p_value": None, "pass": None}
    ac = np.corrcoef(floats[:-lag], floats[lag:])[0, 1]
    z = ac * np.sqrt(n)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return {"test": f"Autocorrelation (lag={lag})", "statistic": ac,
            "p_value": p_value, "pass": p_value >= 0.01}


def run_all_tests(floats: np.ndarray, label: str, raw_bytes: bytes) -> list[dict]:
    """Run full test battery on raw entropy bytes and floats."""
    print(f"  Running tests on {len(floats)} samples [{label}]...")

    bits = bytes_to_bits(raw_bytes)

    results = [
        test_runs(bits),
        test_block_frequency(bits),
        test_entropy(floats),
        test_ks_uniformity(floats),
        test_autocorrelation(floats),
    ]
    return results
