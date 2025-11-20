#!/usr/bin/env python3
"""Generate augmented point-history CSV from `data/point_history.csv`.

This script reads rows of the form: label, x1, y1, x2, y2, ..., xT, yT
and writes an augmented CSV with the same format. Augmentations include:
- random time shift (left/right) with zero padding
- random resample (simulate speed changes)
- time warp (local stretch/compress)
- random crop or pad
- gaussian coordinate noise

Usage example:
  python scripts/augment_point_history.py \
      --input data/point_history.csv \
      --output data/point_history_augmented.csv \
      --aug-factor 3 --seed 42 --keep-original

The script is intentionally dependency-light (only uses Python stdlib and numpy).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random
import sys
from typing import Tuple

import numpy as np


def read_row_to_array(row: list) -> Tuple[int, np.ndarray]:
    # row: list of strings, first is label
    label = int(row[0])
    vals = np.array([float(x) for x in row[1:]], dtype=np.float32)
    if vals.size % 2 != 0:
        raise ValueError("Feature length is not even (expected pairs of x,y)")
    return label, vals.reshape(-1, 2)


def flatten_array(arr: np.ndarray) -> list:
    return [f"{v:.6f}" for v in arr.reshape(-1).tolist()]


def random_time_shift(x: np.ndarray, max_shift: int = 8) -> np.ndarray:
    # shift right (>0) inserts zeros at start; shift left (<0) removes from start and pads zeros at end
    T = x.shape[0]
    shift = random.randint(-max_shift, max_shift)
    if shift == 0:
        return x.copy()
    if shift > 0:
        pad = np.zeros((shift, 2), dtype=x.dtype)
        out = np.vstack([pad, x])[:T]
    else:
        out = np.vstack([x[-shift:], np.zeros((-shift, 2), dtype=x.dtype)])
        out = out[:T]
    return out


def random_resample(x: np.ndarray, target_T: int = 16) -> np.ndarray:
    # linear interpolation to target_T
    T = x.shape[0]
    if T == target_T:
        return x.copy()
    xp = np.linspace(0.0, 1.0, T)
    xq = np.linspace(0.0, 1.0, target_T)
    x_new = np.vstack([
        np.interp(xq, xp, x[:, 0]),
        np.interp(xq, xp, x[:, 1]),
    ]).T.astype(np.float32)
    return x_new


def time_warp(x: np.ndarray, warp_strength: float = 0.2) -> np.ndarray:
    # Simple global stretch/compress by a random factor near 1.0, then resample back
    factor = 1.0 + random.uniform(-warp_strength, warp_strength)
    T = x.shape[0]
    stretched_T = max(2, int(round(T * factor)))
    xp = np.linspace(0.0, 1.0, T)
    xq = np.linspace(0.0, 1.0, stretched_T)
    stretched = np.vstack([
        np.interp(xq, xp, x[:, 0]),
        np.interp(xq, xp, x[:, 1]),
    ]).T.astype(np.float32)
    # resample back to T
    return random_resample(stretched, target_T=T)


def crop_or_pad(x: np.ndarray, target_T: int = 16) -> np.ndarray:
    T = x.shape[0]
    if T == target_T:
        return x.copy()
    if T > target_T:
        # random contiguous crop
        start = random.randint(0, T - target_T)
        return x[start : start + target_T].copy()
    else:
        # pad: choose pad at start or end randomly
        pad_len = target_T - T
        if random.random() < 0.5:
            pad = np.zeros((pad_len, 2), dtype=x.dtype)
            return np.vstack([pad, x]).astype(np.float32)
        else:
            pad = np.zeros((pad_len, 2), dtype=x.dtype)
            return np.vstack([x, pad]).astype(np.float32)


def add_noise(x: np.ndarray, sigma: float = 0.01) -> np.ndarray:
    noise = np.random.normal(scale=sigma, size=x.shape).astype(np.float32)
    return (x + noise)


def augment_one(x: np.ndarray, target_T: int = 16) -> np.ndarray:
    # pipeline: randomly apply a subset of transforms
    y = x.copy()
    # random resample (simulate speed change) with prob
    if random.random() < 0.7:
        # allow variable source length by randomly cropping/padding first
        if random.random() < 0.3:
            y = crop_or_pad(y, target_T=random.randint(max(8, target_T - 6), target_T + 6))
        y = random_resample(y, target_T=target_T)

    # time warp occasionally
    if random.random() < 0.3:
        y = time_warp(y, warp_strength=0.25)

    # random time shift
    if random.random() < 0.6:
        y = random_time_shift(y, max_shift=4)

    # add noise
    if random.random() < 0.9:
        y = add_noise(y, sigma=0.01)

    # final ensure shape
    if y.shape[0] != target_T:
        y = random_resample(y, target_T=target_T)
    return y


def process_file(input_path: Path, output_path: Path, aug_factor: int = 2, target_T: int = 16, keep_original: bool = True, seed: int = None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    input_rows = []
    with input_path.open('r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            input_rows.append(row)

    out_rows = []
    for row in input_rows:
        label, arr = read_row_to_array(row)
        # ensure arr length equals target_T initially by resample/crop/pad
        if arr.shape[0] != target_T:
            arr = random_resample(arr, target_T=target_T)

        if keep_original:
            out_rows.append([str(label), *flatten_array(arr)])

        for i in range(aug_factor):
            aug = augment_one(arr, target_T=target_T)
            out_rows.append([str(label), *flatten_array(aug)])

    # write out
    with output_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        for r in out_rows:
            writer.writerow(r)

    print(f"Wrote {len(out_rows)} rows to {output_path}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Augment point_history dataset")
    p.add_argument('--input', '-i', type=Path, default=Path('data/point_history.csv'))
    p.add_argument('--output', '-o', type=Path, default=Path('data/point_history_augmented.csv'))
    p.add_argument('--aug-factor', type=int, default=2, help='How many augmented samples per original')
    p.add_argument('--target-t', type=int, default=16, help='Target time steps (default 16)')
    p.add_argument('--keep-original', action='store_true', help='Include original samples in output')
    p.add_argument('--seed', type=int, default=None)
    args = p.parse_args(argv)

    if not args.input.exists():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        sys.exit(2)

    process_file(args.input, args.output, aug_factor=args.aug_factor, target_T=args.target_t, keep_original=args.keep_original, seed=args.seed)


if __name__ == '__main__':
    main()
