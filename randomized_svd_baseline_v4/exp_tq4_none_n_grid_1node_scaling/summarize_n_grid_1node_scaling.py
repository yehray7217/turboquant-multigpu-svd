#!/usr/bin/env python3
import re
import sys


CASE_RE = re.compile(
    r"GRID_CASE method=(\S+) nodes=(\d+) mpi_ranks=(\d+) "
    r"m=(\d+) n=(\d+) repeat=(\d+) ngpus=(\d+)"
)
MEAN_RE = re.compile(r"^\s*Total Time\s+([0-9.]+)\s+ms")
PAYLOAD_RE = re.compile(
    r"^\s*(Host-GPU Payload|NVLink Payload|InfiniBand Payload)\s+([0-9.]+)\s+(\S+)"
)
FINAL_ERROR_RE = re.compile(r"^\s*Final Reconstruction Error\s+([0-9.eE+-]+)")


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
    return f"{n // 1024}k" if n % 1024 == 0 else str(n)


def parse(path):
    rows = []
    current = None
    with open(path, "r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            match = CASE_RE.search(line)
            if match:
                if current:
                    rows.append(current)
                method, nodes, mpi_ranks, m_rows, n_cols, repeat, ngpus = match.groups()
                current = {
                    "method": method,
                    "nodes": int(nodes),
                    "mpi_ranks": int(mpi_ranks),
                    "m": int(m_rows),
                    "n": int(n_cols),
                    "repeat": int(repeat),
                    "ngpus": int(ngpus),
                    "total_ms": "",
                    "host_gpu_mib": "",
                    "nvlink_mib": "",
                    "ib_mib": "",
                    "final_reconstruction_error": "",
                }
                continue
            if not current:
                continue
            match = MEAN_RE.match(line)
            if match and current["total_ms"] == "":
                current["total_ms"] = float(match.group(1))
                continue
            match = PAYLOAD_RE.match(line)
            if match:
                label, value, unit = match.groups()
                key = {
                    "Host-GPU Payload": "host_gpu_mib",
                    "NVLink Payload": "nvlink_mib",
                    "InfiniBand Payload": "ib_mib",
                }[label]
                current[key] = to_mib(value, unit)
                continue
            match = FINAL_ERROR_RE.match(line)
            if match:
                current["final_reconstruction_error"] = float(match.group(1))
    if current:
        rows.append(current)
    return rows


def ratio(numerator, denominator):
    if numerator == "" or denominator == "":
        return ""
    return f"{float(numerator) / float(denominator):.6g}"


def main():
    if len(sys.argv) != 2:
        print(
            "usage: summarize_n_grid_1node_scaling.py "
            "tq4_none_n_grid_1node_scaling_<jobid>.out",
            file=sys.stderr,
        )
        return 2

    rows = parse(sys.argv[1])
    none_by_n_gpu = {
        (row["n"], row["ngpus"]): row for row in rows if row["method"] == "none"
    }
    one_gpu_by_method_n = {
        (row["method"], row["n"]): row for row in rows if row["ngpus"] == 1
    }

    print(
        "method,nodes,mpi_ranks,gpu,matrix_size,m,n,repeat,total_ms,"
        "speedup_vs_none,scaling_vs_1gpu,host_gpu_mib,nvlink_mib,ib_mib,"
        "final_reconstruction_error"
    )
    for row in rows:
        none = none_by_n_gpu.get((row["n"], row["ngpus"]))
        one_gpu = one_gpu_by_method_n.get((row["method"], row["n"]))
        speedup_vs_none = ratio(none["total_ms"], row["total_ms"]) if none else ""
        scaling_vs_1gpu = ratio(one_gpu["total_ms"], row["total_ms"]) if one_gpu else ""
        print(
            f"{row['method']},{row['nodes']},{row['mpi_ranks']},{row['ngpus']},"
            f"{matrix_label(row['n'])},{row['m']},{row['n']},{row['repeat']},"
            f"{row['total_ms']},{speedup_vs_none},{scaling_vs_1gpu},"
            f"{row['host_gpu_mib']},{row['nvlink_mib']},{row['ib_mib']},"
            f"{row['final_reconstruction_error']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
