"""SplitFXM RT Instability Physical RNG + Statistical Validation."""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fetch_data import fetch_anu_qrng, fetch_cloudflare_stream, load_geiger_bytes
from process_frames import bytes_to_uniform_floats, find_anim_frames, frames_to_random_bytes
from stat_tests import run_all_tests
from summary import print_comparison_report

GEIGER_FILE = Path(__file__).parent / "data" / "geiger.txt"


def main():
    parser = argparse.ArgumentParser(
        description="RT Instability RNG validation vs Cloudflare drand, ANU QRNG, and Geiger counter"
    )
    parser.add_argument("anim_dir", help="Directory of ParaView PNG animation frames")
    args = parser.parse_args()

    print("=" * 70)
    print("  SplitFXM · RT Instability Physical RNG Validator")
    print("  Sources: RT frames | Cloudflare drand | ANU QRNG | Geiger counter")
    print("=" * 70)

    files = find_anim_frames(args.anim_dir)
    if not files:
        sys.exit("No animation frames found.")

    print(f"[anim] Using {len(files)} frame(s): {files[0].name} – {files[-1].name}")
    print("[rng]  Hashing each frame individually...")
    rt_bytes, total_pixels = frames_to_random_bytes(files)
    print(f"[anim] Snapshot: {total_pixels:,} RGB pixels ({total_pixels * 3 / 1e6:.2f} MB)")

    rt_floats = bytes_to_uniform_floats(rt_bytes)
    if len(rt_floats) == 0:
        sys.exit("RT random sample generation produced no output bytes.")

    print(f"[rng]  RT random samples: {len(rt_floats)} (from {len(rt_bytes)} bytes)")

    n_hashes = len(rt_bytes) // 32
    if n_hashes > 0:
        unique_hashes = len(set(rt_bytes[i * 32:(i + 1) * 32] for i in range(n_hashes)))
        if unique_hashes < n_hashes:
            print(f"[warn] ONLY {unique_hashes}/{n_hashes} HASHES UNIQUE — frames may be identical.")

    n_ref_bytes = len(rt_bytes)
    cf_floats, cf_bytes, round_nums = fetch_cloudflare_stream(n_ref_bytes)
    anu_floats, anu_bytes = fetch_anu_qrng(n_ref_bytes)
    geig_floats, geig_bytes = load_geiger_bytes(str(GEIGER_FILE))

    n_packed = min(len(rt_floats), len(cf_floats), len(anu_floats))
    packed_floats = [rt_floats[:n_packed], cf_floats[:n_packed], anu_floats[:n_packed]]
    packed_bytes = [rt_bytes[:n_packed * 8], cf_bytes[:n_packed * 8], anu_bytes[:n_packed * 8]]

    labels = ["RT ParaView (lavarand)", "Cloudflare drand", "ANU QRNG (vacuum QM)"]
    all_floats = packed_floats
    all_rawbytes = packed_bytes
    all_n = [n_packed, n_packed, n_packed]

    geig_n = len(geig_floats)
    geig_n_bits = (len(geig_bytes) // 8) * 8
    labels.append("Geiger (nuclear decay)")
    all_floats.append(geig_floats)
    all_rawbytes.append(geig_bytes[:geig_n_bits])
    all_n.append(geig_n)

    print(f"\n[stat] Running tests ...")
    print(f"       Packed sources (RT/CF/ANU): {n_packed} samples each")
    print(f"       Geiger: {geig_n} per-byte samples ({len(geig_bytes)} raw bytes)")

    all_results = [
        run_all_tests(floats, label, raw)
        for label, floats, raw in zip(labels, all_floats, all_rawbytes)
    ]

    print_comparison_report(
        list(zip(labels, all_results, all_n)),
        round_nums=round_nums,
    )


if __name__ == "__main__":
    main()
