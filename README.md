# RT Instability Physical RNG Validator

**Medium Article** - [link](https://gpavanb.medium.com/generating-highly-random-numbers-from-fluid-instability-simulations-e37c38749fa9)

Extract randomness from Rayleigh-Taylor (RT) instability simulation output and statistically compare it against cryptographic and physical reference sources: [Cloudflare drand](https://drand.cloudflare.com), [ANU QRNG](https://qrng.anu.edu.au), and optional Geiger counter data.

Entropy extraction follows Cloudflare's [lavarand](https://blog.cloudflare.com/randomness-101-lavarand-in-production/) principle: capture a chaotic physical system as an image (or interface state), hash the pixel/field data, and run statistical tests alongside established randomness beacons.

## Project layout

| File | Role |
|------|------|
| `main.py` | CLI entry point and orchestration |
| `process_frames.py` | VTK/PNG loading, interface extraction, byte hashing |
| `fetch_data.py` | Cloudflare drand, ANU QRNG, and Geiger file loaders |
| `stat_tests.py` | NIST SP 800-22 inspired statistical tests |
| `summary.py` | Side-by-side comparison report |

## Requirements

```bash
pip install numpy scipy requests vtk pillow python-dotenv
```

For ANU QRNG fallback when the public endpoint is unavailable, add your API key to `.env`:

```
QRANDOM_API_KEY=your_key
```

## Usage

Point at a directory of ParaView PNG frames. The script always compares against Cloudflare drand, ANU QRNG, and `data/geiger.txt`.

```bash
python main.py /path/to/anim/
```

## Statistical tests

The validator runs a battery of tests on each source:

- **Bit-based (NIST-inspired):** Runs, Block Frequency
- **Float-based:** Shannon entropy, Kolmogorov–Smirnov uniformity, lag-1 autocorrelation

Results are printed in a multi-column report comparing RT output against reference sources.

## Data formats

**Animation frames:** PNG files (`frame*.png` or `*.png`) in the given directory.

**Geiger file:** `data/geiger.txt` — plain text, one integer 0–255 per line. Lines starting with `#` are comments.

VTK loading helpers live in `process_frames.py` if you want to build a VTK-based workflow separately.
