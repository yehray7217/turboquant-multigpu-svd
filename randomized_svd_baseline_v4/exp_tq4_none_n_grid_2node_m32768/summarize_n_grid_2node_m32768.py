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


def main():
    if len(sys.argv) != 2:
        print(
            "usage: summarize_n_grid_2node_m32768.py "
            "tq4_none_n_grid_2node_m32768_<jobid>.out",
            file=sys.stderr,
        )
        return 2

    rows = parse(sys.argv[1])
    none_by_n = {row["n"]: row for row in rows if row["method"] == "none"}
    print(
        "method,nodes,mpi_ranks,gpu,matrix_size,m,n,repeat,total_ms,"
        "speedup_vs_none,host_gpu_mib,nvlink_mib,ib_mib,final_reconstruction_error"
    )
    for row in rows:
        base = none_by_n.get(row["n"])
        speedup = ""
        if base and row["total_ms"] and base["total_ms"]:
            speedup = f"{float(base['total_ms']) / float(row['total_ms']):.6g}"
        print(
            f"{row['method']},{row['nodes']},{row['mpi_ranks']},{row['ngpus']},"
            f"{matrix_label(row['n'])},{row['m']},{row['n']},{row['repeat']},"
            f"{row['total_ms']},{speedup},{row['host_gpu_mib']},"
            f"{row['nvlink_mib']},{row['ib_mib']},{row['final_reconstruction_error']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
