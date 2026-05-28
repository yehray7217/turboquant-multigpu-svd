#!/bin/bash
set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <source.cpp> [program args...]"
    exit 1
fi

src="$1"
out="$(basename "$src" .cpp).out"

g++ -std=c++17 -O2 -Wall -Wextra -pedantic "$src" -o "$out"
"./$out" "${@:2}"
