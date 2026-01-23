from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from datasets import load_dataset


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _get_first_present(row: Dict[str, Any], keys: Iterable[str]) -> Optional[Any]:
    for k in keys:
        if k in row:
            return row[k]
    return None


@dataclass(frozen=True)
class SliceSpec:
    n: int
    seed: int
    method: str
    split: str
    revision: Optional[str]
    streaming: bool


def _infer_source_id(row: Dict[str, Any], idx: int) -> str:
    v = _get_first_present(row, ["id", "problem_id", "source_id", "uuid"])
    if v is None:
        return str(idx)
    return str(v)


def build_slice(
    *,
    n: int,
    seed: int,
    method: str,
    split: str,
    revision: Optional[str],
    streaming: bool,
) -> Dict[str, Any]:
    ds = load_dataset(
        "nvidia/OpenMathReasoning",
        split=split,
        revision=revision,
        streaming=streaming,
    )

    if method == "shuffle":
        if streaming:
            ds = ds.shuffle(seed=seed, buffer_size=10_000)
        else:
            ds = ds.shuffle(seed=seed)

    rows = []
    if streaming:
        for i, r in enumerate(ds):
            if i >= n:
                break
            rows.append(r)
    else:
        rows = [ds[i] for i in range(min(n, len(ds)))]

    items = []
    for i, r in enumerate(rows):
        r = dict(r)
        source_id = _infer_source_id(r, i)
        problem = _get_first_present(r, ["problem", "question", "prompt"])
        if problem is None:
            raise KeyError(f"Missing problem field in row keys={sorted(r.keys())}")

        final_answer = _get_first_present(r, ["final_answer", "expected_answer", "answer", "expected_output"])
        if final_answer is not None:
            final_answer = str(final_answer)

        generated_solution = _get_first_present(r, ["generated_solution", "solution", "generated_answer"])
        if generated_solution is not None:
            generated_solution = str(generated_solution)

        problem_source = _get_first_present(r, ["problem_source"])
        generation_model = _get_first_present(r, ["generation_model"])

        item_id = f"omr:{split}:{source_id}"
        content = {
            "problem": str(problem),
            "final_answer": final_answer,
        }
        item = {
            "item_id": item_id,
            "source": {
                "dataset": "nvidia/OpenMathReasoning",
                "split": split,
                "source_id": source_id,
            },
            "content": content,
            "trace": {
                "steps": [],
                "max_steps": 30,
            },
            "provenance": {
                "generated_solution": generated_solution,
            },
            "hashes": {
                "content_sha256": _sha256_str(_canonical_json(content)),
            },
        }
        if problem_source is not None:
            item["source"]["problem_source"] = str(problem_source)
        if generation_model is not None:
            item["source"]["generation_model"] = str(generation_model)

        items.append(item)

    features = list(getattr(ds, "features", {}).keys())
    meta = {
        "dataset_fingerprint": getattr(ds, "_fingerprint", None),
        "num_rows_total": (None if streaming else len(ds)),
        "features": features,
        "streaming": streaming,
    }
    return {"items": items, "meta": meta}


def write_slice(out_dir: Path, *, spec: SliceSpec) -> Path:
    slice_hash = _sha256_str(
        _canonical_json(
            {
                "dataset": "nvidia/OpenMathReasoning",
                "split": spec.split,
                "revision": spec.revision,
                "method": spec.method,
                "seed": spec.seed,
                "n": spec.n,
                "streaming": spec.streaming,
            }
        )
    )[:12]
    slice_id = f"slice_omr_{spec.split}_{spec.method}_n{spec.n}_s{spec.seed}_{slice_hash}"

    slice_dir = out_dir / slice_id

    if slice_dir.exists():
        manifest_path = slice_dir / "manifest.json"
        items_path = slice_dir / "items.jsonl"
        if manifest_path.exists() and items_path.exists():
            return slice_dir
        raise FileExistsError(
            f"Slice dir already exists but is missing expected files: {slice_dir}"
        )

    slice_dir.mkdir(parents=True, exist_ok=False)

    built = build_slice(
        n=spec.n,
        seed=spec.seed,
        method=spec.method,
        split=spec.split,
        revision=spec.revision,
        streaming=spec.streaming,
    )
    items = built["items"]
    meta = built["meta"]

    created_at_utc = datetime.now(timezone.utc).isoformat()
    fingerprint = meta.get("dataset_fingerprint") or "unknown"

    items_path = slice_dir / "items.jsonl"
    with items_path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(_canonical_json(it))

    items_sha256 = _sha256_bytes(items_path.read_bytes())

    manifest = {
        "schema_version": "candidate_pool_slice@0.1",
        "slice_id": slice_id,
        "created_at_utc": created_at_utc,
        "source": {
            "dataset": "nvidia/OpenMathReasoning",
            "split": spec.split,
            "revision": spec.revision,
            "method": spec.method,
            "seed": spec.seed,
            "n": spec.n,
            "streaming": spec.streaming,
            "dataset_fingerprint": fingerprint,
            "features": meta.get("features"),
        },
        "trace": {
            "steps_initially_blank": True,
            "max_steps_range": [8, 30],
            "step_type_vocab": [
                "parse",
                "rewrite",
                "simplify",
                "substitute",
                "expand",
                "factor",
                "differentiate",
                "integrate",
                "apply_identity",
                "apply_theorem",
                "case_split",
                "introduce_variable",
                "bound_estimate",
                "compute",
                "verify",
                "conclude",
            ],
        },
        "counts": {"items": len(items)},
        "files": {
            "items_jsonl": {"path": "items.jsonl", "sha256": items_sha256},
        },
        "hashing": {
            "canonicalization": "json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False) + '\\n'",
            "item_content_hash_input": "content object only",
            "items_file_hash_input": "exact bytes of items.jsonl",
        },
    }

    manifest_path = slice_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return slice_dir


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--method", choices=["shuffle", "first"], default="shuffle")
    p.add_argument("--split", choices=["cot", "tir", "genselect", "additional_problems"], default="cot")
    p.add_argument("--revision", default=None)
    p.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--out", default="candidate_pool/slices")
    args = p.parse_args()

    out_dir = Path(args.out)
    spec = SliceSpec(
        n=int(args.n),
        seed=int(args.seed),
        method=str(args.method),
        split=str(args.split),
        revision=(str(args.revision) if args.revision is not None else None),
        streaming=bool(args.streaming),
    )
    slice_dir = write_slice(out_dir, spec=spec)
    print(str(slice_dir))
    sys.stdout.flush()
    sys.stderr.flush()
    if spec.streaming:
        os._exit(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
