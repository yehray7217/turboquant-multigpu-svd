#!/usr/bin/env python3
"""Download and decompress MNIST/Fashion-MNIST IDX files with stdlib only."""

import argparse
import gzip
import shutil
import urllib.request
from pathlib import Path


URLS = {
    "fashion-mnist": {
        "train-images-idx3-ubyte": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",
    },
    "mnist": {
        "train-images-idx3-ubyte": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
    },
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=sorted(URLS), default="fashion-mnist")
    p.add_argument("--data-dir", default="./data")
    args = p.parse_args()

    out_dir = Path(args.data_dir) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, url in URLS[args.dataset].items():
        raw = out_dir / name
        gz = out_dir / (name + ".gz")
        if raw.exists():
            print("exists", raw)
            continue
        if not gz.exists():
            print("download", url)
            urllib.request.urlretrieve(url, str(gz))
        print("decompress", gz)
        with gzip.open(str(gz), "rb") as src, raw.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    print("ready", out_dir)


if __name__ == "__main__":
    main()
