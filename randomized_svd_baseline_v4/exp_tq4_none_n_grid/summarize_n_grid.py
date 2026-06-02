#!/usr/bin/env python3
import re
import sys


CASE_RE = re.compile(r"GRID_CASE method=(\S+) m=(\d+) n=(\d+) repeat=(\d+) ngpus=(\d+)")
MEAN_RE = re.compile(r"^\s*Total Time\s+([0-9.]+)\s+ms")
PAYLOAD_RE = re.compile(r"^\s*(Host-GPU Payload|NVLink Payload|InfiniBand Payload)\s+([0-9.]+)\s+(\S+)")


def to_mib(value, unit):
    value = float(value)
    unit = unit.lower()
    if unit.startswith("gib"):
        return value * 1024.0
    if unit.startswith("mib"):
        return value
    if unit.startswith("kib"):
        return value / 1024.0
    if unit.startswith("b"):
        return value / (1024.0 * 1024.0)
    return value


def matrix_label(n):
    if n % 1024 == 0:
        return f"{n // 1024}k"
    return str(n)


def parse(path):
    rows = []
    current = None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = CASE_RE.search(line)
            if m:
                if current:
                    rows.append(current)
                method, m_rows, n_cols, repeat, ngpus = m.groups()
                current = {
                    "method": method,
                    "m": int(m_rows),
                    "n": int(n_cols),
                    "repeat": int(repeat),
                    "ngpus": int(ngpus),
                    "total_ms": "",
                    "host_gpu_mib": "",
                    "nvlink_mib": "",
                    "ib_mib": "",
                }
                continue
            if not current:
                continue
            m = MEAN_RE.match(line)
            if m and current["total_ms"] == "":
                current["total_ms"] = float(m.group(1))
                continue
            m = PAYLOAD_RE.match(line)
            if m:
                label, value, unit = m.groups()
                key = {
                    "Host-GPU Payload": "host_gpu_mib",
                    "NVLink Payload": "nvlink_mib",
                    "InfiniBand Payload": "ib_mib",
                }[label]
                current[key] = to_mib(value, unit)
    if current:
        rows.append(current)
    return rows


def main():
    if len(sys.argv) != 2:
        print("usage: summarize_n_grid.py tq4_none_n_grid_8gpu_<jobid>.out", file=sys.stderr)
        return 2
    rows = parse(sys.argv[1])
    none_by_n = {row["n"]: row for row in rows if row["method"] == "none"}

    print("method,matrix_size,m,n,gpu,repeat,total_ms,speedup_vs_none,host_gpu_mib,nvlink_mib,ib_mib")
    for row in rows:
        base = none_by_n.get(row["n"])
        speedup = ""
        if base and row["total_ms"] and base["total_ms"]:
            speedup = f"{float(base['total_ms']) / float(row['total_ms']):.6g}"
        print(
            f"{row['method']},{matrix_label(row['n'])},{row['m']},{row['n']},"
            f"{row['ngpus']},{row['repeat']},{row['total_ms']},{speedup},"
            f"{row['host_gpu_mib']},{row['nvlink_mib']},{row['ib_mib']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
