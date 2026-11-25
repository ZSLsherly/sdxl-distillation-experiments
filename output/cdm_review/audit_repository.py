"""Reproduce configuration, data-overlap, and provenance checks without model dependencies."""
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "research_repos/CDM"
OUT = Path(__file__).resolve().parent
SHA = "bacc57f60c4366cb9374b59860fb094e4edb419f"

class ConfigDict(dict):
    def __getattr__(self, key):
        return self[key]
    def __setattr__(self, key, value):
        self[key] = value

tree = ast.parse((REPO/"config/config.py").read_text(encoding="utf8"))
tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
ns = {"os": os, "Path": Path, "__file__": str(REPO/"config/config.py"),
      "ml_collections": types.SimpleNamespace(ConfigDict=ConfigDict)}
exec(compile(tree, "config.py", "exec"), ns)
result = {"commit": SHA, "method": "Original config assignments executed with a lightweight ConfigDict substitute; no model imported.",
          "configs": {}, "datasets": {}, "overlap": {}}
for name in ["sd3", "longcat"]:
    c = ns[name]()
    ft = student = 0
    for epoch in range(c.num_epochs):
        for batch in range(c.sample.num_batches_per_epoch):
            ft += 1
            if ft % c.train.student.update_ratio == 0:
                student += 1
    c["audit_count"] = {"student_updates": student, "fake_updates": ft,
                        "global_batch_8gpu": c.train.batch_size*8,
                        "global_batch_16gpu": c.train.batch_size*16}
    result["configs"][name] = c

def read_lines(path):
    with path.open(encoding="utf8") as f:
        return [line.strip() for line in f.readlines()]

for d in (REPO/"dataset").iterdir():
    for f in d.glob("*.txt"):
        lines = read_lines(f)
        result["datasets"][f.relative_to(REPO).as_posix()] = {
            "rows": len(lines), "nonempty": sum(bool(x) for x in lines),
            "unique": len(set(lines)), "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
train = read_lines(REPO/"dataset/mixed_200k/train.txt")
positions = {}
for i, line in enumerate(train, 1):
    positions.setdefault(line, []).append(i)
with (OUT/"overlap_evidence.csv").open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["evaluation_file", "test_line_1based", "training_lines_1based", "prompt_sha256"])
    for name in ["mixed_200k", "pickscore", "ocr", "coco2014val_10k"]:
        rel = f"dataset/{name}/test.txt"
        test = read_lines(REPO/rel)
        matched = [x for x in test if x in positions]
        result["overlap"][name] = {"test_rows": len(test), "matching_test_rows": len(matched),
                                   "matching_unique_prompts": len(set(matched)),
                                   "fraction": len(matched)/len(test)}
        for i, line in enumerate(test, 1):
            if line in positions:
                w.writerow([rel, i, ";".join(map(str, positions[line])),
                            hashlib.sha256(line.encode()).hexdigest()])
with (REPO/"dataset/dpgbench/dpg_bench.csv").open(encoding="utf8") as f:
    rows = list(csv.DictReader(f))
result["datasets"]["dataset/dpgbench/dpg_bench.csv"] = {
    "rows": len(rows), "unique_item_ids": len({x["item_id"] for x in rows})}
ours_path = ROOT/"output/sdxl-crossdistill/coco10k_g160_20261020/manifest.json"
if ours_path.exists():
    ours = json.loads(ours_path.read_text(encoding="utf8"))
    a = {r["prompt"] for r in ours["samples"]}
    b = set(read_lines(REPO/"dataset/coco2014val_10k/test.txt"))
    result["coco_prompt_comparison"] = {"our_rows": len(ours["samples"]),
        "our_unique": len(a), "cdm_unique": len(b), "exact_shared_unique_prompts": len(a & b),
        "same_prompt_set": a == b}
try:
    random.Random((0, 0))
    result["tuple_seed"] = {"python": sys.version, "works": True}
except TypeError as error:
    result["tuple_seed"] = {"python": sys.version, "error": str(error)}
result["files"] = {p.relative_to(REPO).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in REPO.rglob("*.py") if ".git" not in p.parts}
(OUT/"repository_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
print(json.dumps({"overlap": result["overlap"], "coco_prompt_comparison": result.get("coco_prompt_comparison")}, indent=2))
