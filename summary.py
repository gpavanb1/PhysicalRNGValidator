"""Multi-source comparison report formatting and printing."""


def _fmt_val(r: dict) -> str:
    """Format the primary metric for a test result."""
    if "efficiency_pct" in r:
        v = r["efficiency_pct"]
        return f"{v:.2f}%" if v is not None else "N/A"
    pv = r.get("p_value")
    if pv is not None:
        return f"p={pv:.4f}"
    return r.get("note", "N/A")[:8]


def _sym(r: dict) -> str:
    p = r.get("pass")
    if p is None:
        return "?"
    return "✓" if bool(p) else "✗"


def print_comparison_report(
    sources: list[tuple[str, list[dict], int]],
    round_nums: list[int] | None = None,
):
    """
    Print a side-by-side comparison table for an arbitrary number of sources.

    sources: list of (label, test_results, n_samples)
    """
    N_TESTS = len(sources[0][1])
    N_SRC = len(sources)
    COL_W = max(14, max(len(s[0]) for s in sources) + 1)
    TEST_W = 35
    W = TEST_W + 2 + (COL_W + 3) * N_SRC

    print("\n" + "=" * W)
    print("  RANDOMNESS VALIDATION REPORT")
    print("  SplitFXM RT Instability  ·  Multi-Source Comparison")
    print("=" * W)
    for label, _, n in sources:
        print(f"  {label:<{TEST_W}} n={n}")
    if round_nums and round_nums[0] >= 0:
        print(f"  Cloudflare drand rounds: {min(round_nums)} – {max(round_nums)}")
    print("=" * W)

    hdr = f"  {'Test':<{TEST_W}}"
    for label, _, _ in sources:
        short = label[:COL_W - 1]
        hdr += f"  {short:^{COL_W}}"
    print(hdr)
    print("  " + "-" * (W - 2))

    all_pass = True
    for test_idx in range(N_TESTS):
        row_results = [src_results[test_idx] for _, src_results, _ in sources]
        test_name = row_results[0]["test"][:TEST_W - 1]
        row_str = f"  {test_name:<{TEST_W}}"
        for r in row_results:
            cell = f"{_sym(r)} {_fmt_val(r)}"
            row_str += f"  {cell:<{COL_W}}"
            p = r.get("pass")
            if p is not None and not bool(p):
                all_pass = False
        print(row_str)

    print("  " + "-" * (W - 2))
    summary = f"  {'OVERALL':<{TEST_W}}"
    for _, src_results, _ in sources:
        passes = sum(1 for r in src_results if r.get("pass") is not None and bool(r.get("pass")))
        total = sum(1 for r in src_results if r.get("pass") is not None)
        sym = "✓ ALL" if passes == total else f"~ {passes}/{total}"
        summary += f"  {sym:<{COL_W}}"
    print(summary)
    print("=" * W)

    if all_pass:
        print("\n  ✓ All sources pass all tests — physical/quantum chaos is")
        print("    statistically indistinguishable from cryptographic beacons.")
    else:
        print("\n  ~ Partial pass — see individual test results above.")
        print("    More samples may help marginal tests.")
    print("=" * W)
