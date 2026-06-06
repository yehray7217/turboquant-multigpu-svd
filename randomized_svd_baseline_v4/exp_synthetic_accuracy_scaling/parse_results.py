#!/usr/bin/env python3
"""Parse synthetic accuracy + scaling logs into a tidy table.

Reads all accuracy_sweep_*.out and scaling_*.out logs in output_logs/,
splits them into CASE blocks (each with a Timing phase and a Final-error
phase), and extracts the headline metrics. Prints both a raw dump and the
two README tables (accuracy sweep, scaling) with derived metrics.
"""
import glob
import os
import re
import sys

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_logs")


def num(s):
    return float(s.replace("%", ""))


def first_val(block, label):
    """Return the mean (first numeric) for a summary row whose stripped text
    begins with `label` followed by whitespace and a number."""
    pat = re.compile(r"^\s*" + re.escape(label) + r"\s+([-\d.eE+]+)\s*%?\s", re.M)
    m = pat.search(block)
    return num(m.group(1)) if m else None


def parse_case_header(line):
    """CASE p=0.4 method=none m=32768 n=8192 ...  OR  size=16384x4096 ..."""
    d = dict(re.findall(r"(\w+)=([\w.]+)", line))
    return d


def split_cases(text):
    """Yield (header_dict, timing_block, final_block) per CASE."""
    parts = re.split(r"^=+$\n(CASE .*)$", text, flags=re.M)
    # parts: [pre, header1, body1, header2, body2, ...]
    for i in range(1, len(parts), 2):
        header = parse_case_header(parts[i])
        body = parts[i + 1]
        # split body into Timing phase and Final-error phase
        seg = re.split(r"^Final-error:.*$", body, flags=re.M)
        timing = seg[0]
        final = seg[1] if len(seg) > 1 else ""
        yield header, timing, final


def parse_logs():
    rows = []
    for path in sorted(glob.glob(os.path.join(LOG_DIR, "accuracy_sweep_*.out"))) + \
                sorted(glob.glob(os.path.join(LOG_DIR, "scaling_*.out"))):
        text = open(path).read()
        kind = "scaling" if "scaling_" in os.path.basename(path) else "accuracy"
        for header, timing, final in split_cases(text):
            if "size" in header:
                m, n = header["size"].split("x")
            else:
                m, n = header.get("m"), header.get("n")
            row = {
                "kind": kind,
                "log": os.path.basename(path),
                "p": header.get("p"),
                "size": (m, n),
                "method": header.get("method"),
                "total_ms": first_val(timing, "Total Time"),
                "gpu_ms": first_val(timing, "GPU Compute Time"),
                "host_ms": first_val(timing, "Host/Staging Time"),
                "nvlink_ms": first_val(timing, "NVLink Time"),
                "ib_ms": first_val(timing, "InfiniBand Time"),
                "other_ms": first_val(timing, "Other/Sync Time"),
                "nvlink_payload": first_val(timing, "NVLink Payload"),
                "ib_payload": first_val(timing, "InfiniBand Payload"),
                "final_err": first_val(final, "Final Reconstruction Error"),
                "global_b_err": first_val(final, "Global B Relative Error"),
            }
            # theoretical + err ratio from the Final Reconstruction Error row
            m = re.search(
                r"Final Reconstruction Error\s+([-\d.eE+]+)%\s+([-\d.eE+]+)%\s+"
                r"([-\d.eE+]+)%\s+([-\d.eE+]+)%\s+([-\d.eE+]+)", final)
            if m:
                row["theoretical"] = num(m.group(4))
                row["err_ratio"] = num(m.group(5))
            else:
                row["theoretical"] = None
                row["err_ratio"] = None
            rows.append(row)
    return rows


def fmt(v, prec=4, suffix=""):
    return f"{v:.{prec}f}{suffix}" if isinstance(v, float) else "-"


def main():
    rows = parse_logs()

    # group accuracy rows by p
    acc = [r for r in rows if r["kind"] == "accuracy"]
    scl = [r for r in rows if r["kind"] == "scaling"]

    def by_method(group):
        d = {}
        for r in group:
            d[r["method"]] = r
        return d

    print("=" * 100)
    print("ACCURACY SWEEP (32768 x 8192)")
    print("=" * 100)
    print("| p | Method | Total ms | Speedup | Final Err % | Theo % | ErrRatio | ErrInfl | GlobalB % | IB ms | NVLink ms | IB MiB | NVLink MiB |")
    print("|---|--------|---------:|--------:|------------:|-------:|---------:|--------:|----------:|------:|----------:|-------:|-----------:|")
    for p in ["0.4", "0.6", "0.8", "1.0"]:
        grp = by_method([r for r in acc if r["p"] == p])
        base = grp.get("none")
        for method in ["none", "tq8", "tq4"]:
            r = grp.get(method)
            if not r:
                continue
            speedup = base["total_ms"] / r["total_ms"] if base and r["total_ms"] else None
            infl = (r["err_ratio"] / base["err_ratio"]) if base and r["err_ratio"] and base["err_ratio"] else None
            print(f"| {p} | {method} | {fmt(r['total_ms'],3)} | "
                  f"{fmt(speedup,3,'x') if speedup else '1.000x'} | {fmt(r['final_err'],4)} | "
                  f"{fmt(r['theoretical'],4)} | {fmt(r['err_ratio'],5)} | "
                  f"{fmt(infl,5) if infl else '1.00000'} | {fmt(r['global_b_err'],4) if r['global_b_err'] else '-'} | "
                  f"{fmt(r['ib_ms'],3)} | {fmt(r['nvlink_ms'],3)} | {fmt(r['ib_payload'],2)} | {fmt(r['nvlink_payload'],2)} |")

    print()
    print("=" * 100)
    print("SCALING (p=0.6)  [16384x4096 from scaling job; 32768x8192 reused from accuracy p=0.6]")
    print("=" * 100)
    print("| Size | Method | Total ms | Speedup | IB ms | NVLink ms | Final Err % | ErrRatio | ErrInfl | IB MiB | NVLink MiB |")
    print("|------|--------|---------:|--------:|------:|----------:|------------:|---------:|--------:|-------:|-----------:|")
    # build size groups: scaling logs (16384x4096) + accuracy p=0.6 (32768x8192)
    size_groups = {}
    for r in scl:
        size_groups.setdefault(r["size"], {})[r["method"]] = r
    p06 = by_method([r for r in acc if r["p"] == "0.6"])
    if p06:
        size_groups[("32768", "8192")] = p06
    for size in sorted(size_groups, key=lambda s: int(s[0])):
        grp = size_groups[size]
        base = grp.get("none")
        for method in ["none", "tq8", "tq4"]:
            r = grp.get(method)
            if not r:
                continue
            speedup = base["total_ms"] / r["total_ms"] if base and r["total_ms"] else None
            infl = (r["err_ratio"] / base["err_ratio"]) if base and r["err_ratio"] and base["err_ratio"] else None
            print(f"| {size[0]}x{size[1]} | {method} | {fmt(r['total_ms'],3)} | "
                  f"{fmt(speedup,3,'x') if speedup else '1.000x'} | {fmt(r['ib_ms'],3)} | {fmt(r['nvlink_ms'],3)} | "
                  f"{fmt(r['final_err'],4)} | {fmt(r['err_ratio'],5)} | {fmt(infl,5) if infl else '1.00000'} | "
                  f"{fmt(r['ib_payload'],2)} | {fmt(r['nvlink_payload'],2)} |")


if __name__ == "__main__":
    main()
