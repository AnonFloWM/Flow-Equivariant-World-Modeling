import os
import sys
import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Set

import torch


def _bytes_of(value) -> bytes:
    try:
        import numpy as np  # noqa: F401
    except Exception:
        pass

    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
        # Include shape and dtype to avoid collisions across different shapes
        return (str(arr.shape) + str(arr.dtype)).encode() + arr.tobytes()
    elif isinstance(value, (list, tuple)):
        return b"(" + b"".join(_bytes_of(x) for x in value) + b")"
    elif isinstance(value, dict):
        return b"{" + b"".join(k.encode() + b":" + _bytes_of(value[k]) for k in sorted(value)) + b"}"
    elif value is None:
        return b"None"
    else:
        try:
            return str(value).encode()
        except Exception:
            return bytes(str(type(value)), "utf-8")


def _hash_blob(parts: List[bytes]) -> str:
    return hashlib.sha256(b"||".join(parts)).hexdigest()


def compute_hashes_for_pt(pt_path: str) -> Dict[str, str]:
    """Return multiple hash signatures extracted from a saved .pt payload.

    We support both static and dynamic payloads. Keys we look for:
      - common:  actions
      - static:  digit_labels, digit_positions
      - dynamic: digit_labels, digit_init_positions, digit_velocities,
                 digit_positions_over_time, camera_centers_input, camera_centers_target

    Returned hash keys:
      - actions_hash: based only on actions
      - world_init_hash: based on (digit_labels + initial positions)
      - dynamic_traj_hash (only if dynamic keys present): based on digit init + velocities + traj + camera path
    """
    payload = torch.load(pt_path, map_location="cpu")

    hashes: Dict[str, str] = {}

    # 1) actions hash (common to both static + dynamic)
    if "actions" in payload:
        hashes["actions_hash"] = _hash_blob([_bytes_of(payload["actions"])])

    # 2) world init hash: use labels + positions (static: digit_positions; dynamic: digit_init_positions)
    labels = payload.get("digit_labels", None)
    pos0 = payload.get("digit_positions", None)
    if pos0 is None:
        pos0 = payload.get("digit_init_positions", None)
    if labels is not None and pos0 is not None:
        hashes["world_init_hash"] = _hash_blob([_bytes_of(labels), _bytes_of(pos0)])

    # 3) dynamic trajectory hash: only if dynamic-specific fields exist
    dyn_fields = (
        payload.get("digit_init_positions", None),
        payload.get("digit_velocities", None),
        payload.get("digit_positions_over_time", None),
        payload.get("camera_centers_input", None),
        payload.get("camera_centers_target", None),
    )
    if all(x is not None for x in dyn_fields):
        hashes["dynamic_traj_hash"] = _hash_blob([_bytes_of(x) for x in dyn_fields])

    return hashes


def scan_dir(pt_dir: str) -> Dict[str, Dict[str, List[str]]]:
    """Scan a directory recursively for .pt files and compute all hash types.

    Returns a dict: { hash_type -> { hash_value -> [pt_paths...] } }.
    """
    result: Dict[str, Dict[str, List[str]]] = {}
    for root, _, files in os.walk(pt_dir):
        for fname in files:
            if not fname.endswith('.pt'):
                continue
            p = os.path.join(root, fname)
            try:
                hashes = compute_hashes_for_pt(p)
            except Exception as e:
                print(f"[WARN] Failed to load {p}: {e}", file=sys.stderr)
                continue
            for htype, hval in hashes.items():
                bucket = result.setdefault(htype, {})
                bucket.setdefault(hval, []).append(p)
    return result


def intersections(a: Dict[str, Dict[str, List[str]]], b: Dict[str, Dict[str, List[str]]]) -> Dict[str, Set[str]]:
    """Return set intersections per hash_type between two scanned directories."""
    inter: Dict[str, Set[str]] = {}
    for htype in set(a.keys()) | set(b.keys()):
        aset = set(a.get(htype, {}).keys())
        bset = set(b.get(htype, {}).keys())
        inter[htype] = aset & bset
    return inter


def print_summary(name_a: str, a: Dict[str, Dict[str, List[str]]], name_b: str, b: Dict[str, Dict[str, List[str]]], max_show: int = 5) -> None:
    inter = intersections(a, b)
    print(f"\n=== {name_a} vs {name_b} ===")
    for htype, hv in inter.items():
        total_a = sum(len(v) for v in a.get(htype, {}).values())
        total_b = sum(len(v) for v in b.get(htype, {}).values())
        uniq_a = len(a.get(htype, {}))
        uniq_b = len(b.get(htype, {}))
        print(f"[{htype}] total_a={total_a} total_b={total_b} uniq_a={uniq_a} uniq_b={uniq_b} intersections={len(hv)}")
        if hv:
            print(f"  examples (up to {max_show}):")
            for i, h in enumerate(list(hv)[:max_show]):
                print("  HASH:", h)
                print("    A:")
                for p in a[htype][h][:max_show]:
                    print("      ", p)
                print("    B:")
                for p in b[htype][h][:max_show]:
                    print("      ", p)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check for overlaps between static/dynamic train/val MNIST-World datasets (.pt metadata files)")
    parser.add_argument("--root", default="./data/mnist_world/", help="Root directory containing dynamic_* and static_* subdirs")
    parser.add_argument("--max_show", type=int, default=5, help="Max examples to print per intersecting hash")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)

    dynamic_train = str(root / "dynamic_training")
    dynamic_val = str(root / "dynamic_validation")
    static_train = str(root / "static_training")
    static_val = str(root / "static_validation")

    print("Scanning directories:")
    for d in [dynamic_train, dynamic_val, static_train, static_val]:
        print(" -", d)

    dyn_tr = scan_dir(dynamic_train) if os.path.isdir(dynamic_train) else {}
    dyn_va = scan_dir(dynamic_val) if os.path.isdir(dynamic_val) else {}
    sta_tr = scan_dir(static_train) if os.path.isdir(static_train) else {}
    sta_va = scan_dir(static_val) if os.path.isdir(static_val) else {}

    # Primary checks: within each regime
    print_summary("dynamic_training", dyn_tr, "dynamic_validation", dyn_va, max_show=args.max_show)
    print_summary("static_training", sta_tr, "static_validation", sta_va, max_show=args.max_show)

    # Optional cross checks (can reveal identical action/camera paths across regimes)
    print_summary("dynamic_training", dyn_tr, "static_validation", sta_va, max_show=args.max_show)
    print_summary("static_training", sta_tr, "dynamic_validation", dyn_va, max_show=args.max_show)


if __name__ == "__main__":
    main()


