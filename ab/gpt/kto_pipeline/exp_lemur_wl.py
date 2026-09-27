#!/usr/bin/env python3
"""WL-graph diversity of the LEMUR corpus as a human-authored reference (TEVC revision).

Loads unique LEMUR architectures from the ab.nn dataset, canonicalizes each to its
Weisfeiler-Lehman graph hash with the SAME settings as the novelty gate used in the
runs (aten tracer, topology granularity, wl_rounds=3, sizes 32/64/96/224/256), and
reports the number of distinct WL structures, the WL-collision rate among the
code-unique architectures, and the Shannon entropy of the hash distribution. This is
the "human reference" diversity the paper compares its generated corpora against.

CPU only (torch.fx tracing); the cluster launcher still allocates a GPU to satisfy
policy. Writes Experiment_Metrics/lemur_wl_diversity.json.

Run in-container:
  python -m ab.gpt.kto_pipeline.exp_lemur_wl --dataset cifar-10
  python -m ab.gpt.kto_pipeline.exp_lemur_wl --dataset "" --out Experiment_Metrics/lemur_wl_all.json
"""
from __future__ import annotations

import argparse
import json
import math
import tempfile
from collections import Counter
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="WL-graph diversity of the LEMUR corpus")
    ap.add_argument("--task", default="img-classification")
    ap.add_argument("--dataset", default="cifar-10",
                    help='LEMUR dataset filter; pass "" for all datasets under the task')
    ap.add_argument("--sizes", default="32,64,96,224,256")
    ap.add_argument("--in_ch", type=int, default=3)
    ap.add_argument("--n_cls", type=int, default=10)
    ap.add_argument("--granularity", default="topology", choices=["topology", "typed"])
    ap.add_argument("--wl_rounds", type=int, default=3)
    ap.add_argument("--out", default="Experiment_Metrics/lemur_wl_diversity.json")
    args = ap.parse_args()

    from ab.nn.api import data
    from ab.gpt.kto_pipeline.graph_novelty_audit import graph_hash_for_file

    df = data(task=args.task, dataset=(args.dataset or None), unique_nn=True)
    codes = [c for c in df["nn_code"].tolist() if isinstance(c, str) and c.strip()] \
        if "nn_code" in getattr(df, "columns", []) else []
    print(f"LEMUR architectures (unique code): {len(codes)}  task={args.task} dataset={args.dataset or 'ALL'}")
    if not codes:
        raise SystemExit("No LEMUR nn_code rows returned; check ab.nn.api.data availability.")

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    hashes, ok, failed = [], 0, Counter()
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        for i, code in enumerate(codes):
            f = tdp / f"lemur_{i}.py"
            f.write_text(code, encoding="utf-8")
            try:
                h, status = graph_hash_for_file(f, sizes, args.in_ch, args.n_cls,
                                                "aten", args.granularity, args.wl_rounds)
            except Exception as e:  # noqa: BLE001
                h, status = None, f"raise:{type(e).__name__}"
            if h:
                hashes.append(h); ok += 1
            else:
                failed[status] += 1
            if (i + 1) % 50 == 0:
                print(f"  traced {i+1}/{len(codes)}  ok={ok}")

    distinct = len(set(hashes))
    counts = list(Counter(hashes).values())
    tot = sum(counts)
    H = -sum((c / tot) * math.log2(c / tot) for c in counts) if tot else 0.0
    Hnorm = H / math.log2(distinct) if distinct > 1 else 0.0
    collision = 1.0 - distinct / ok if ok else 0.0

    result = {
        "task": args.task, "dataset": args.dataset or "ALL",
        "n_code_unique": len(codes), "n_traced": ok, "trace_failures": dict(failed),
        "distinct_wl": distinct, "wl_collision_rate": round(collision, 4),
        "entropy_bits": round(H, 3), "entropy_norm": round(Hnorm, 3),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
