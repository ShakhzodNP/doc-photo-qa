"""Run the labeled test set through the running service and compute quality metrics.

Usage:
    python eval/evaluate.py [--url http://localhost:5000/api/check] [--data data/test_set]

Reports precision and recall per check and per reason, the share of good photos rejected,
the share of bad photos accepted, mean and max response time, and the most frequent errors.
Writes eval/results/predictions.csv and eval/results/report.json.
"""
import argparse
import csv
import json
import os
import statistics
import time
from collections import Counter

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKS = {
    "document": {"NO_DOCUMENT", "SCREENSHOT", "BLANK_PAGE"},
    "framing": {"DOCUMENT_CUT"},
    "geometry": {"TOO_SMALL", "TOO_TILTED"},
    "sharpness": {"BLURRY"},
    "lighting": {"TOO_DARK", "OVEREXPOSED", "GLARE"},
}


def precision_recall(rows, codes):
    tp = sum(1 for r in rows if r["expected"] & codes and r["predicted"] & codes)
    fp = sum(1 for r in rows if not r["expected"] & codes and r["predicted"] & codes)
    fn = sum(1 for r in rows if r["expected"] & codes and not r["predicted"] & codes)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {"precision": precision, "recall": recall, "positives": tp + fn, "tp": tp, "fp": fp, "fn": fn}


def fmt(value):
    return "  n/a" if value is None else f"{value:5.1%}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:5000/api/check")
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "test_set"))
    args = ap.parse_args()

    labels = list(csv.DictReader(open(os.path.join(args.data, "labels.csv"), encoding="utf-8")))
    rows = []
    for label in labels:
        path = os.path.join(args.data, "images", label["file"])
        start = time.perf_counter()
        with open(path, "rb") as f:
            response = requests.post(args.url, files={"file": (label["file"], f, "image/jpeg")}, timeout=30)
        elapsed = time.perf_counter() - start
        body = response.json()
        rows.append({
            "file": label["file"], "kind": label["kind"],
            "expected_ok": label["ok"] == "1", "predicted_ok": bool(body["ok"]),
            "expected": set(filter(None, label["reasons"].split(";"))),
            "predicted": {r["code"] for r in body["reasons"]},
            "status": response.status_code, "seconds": elapsed,
        })

    good = [r for r in rows if r["expected_ok"]]
    bad = [r for r in rows if not r["expected_ok"]]
    times = [r["seconds"] for r in rows]
    report = {
        "images": len(rows), "good": len(good), "bad": len(bad),
        "good_rejected_share": sum(not r["predicted_ok"] for r in good) / len(good),
        "bad_accepted_share": sum(r["predicted_ok"] for r in bad) / len(bad),
        "mean_response_s": statistics.mean(times), "max_response_s": max(times),
        "checks": {name: precision_recall(rows, codes) for name, codes in CHECKS.items()},
        "reasons": {code: precision_recall(rows, {code}) for codes in CHECKS.values() for code in sorted(codes)},
    }

    print(f"images: {report['images']} (good {report['good']}, bad {report['bad']})")
    print(f"good photos rejected: {report['good_rejected_share']:.1%}")
    print(f"bad photos accepted:  {report['bad_accepted_share']:.1%}")
    print(f"response time: mean {report['mean_response_s'] * 1000:.0f} ms, max {report['max_response_s'] * 1000:.0f} ms")
    for title, table in (("check", report["checks"]), ("reason", report["reasons"])):
        print(f"\n{title:12s} precision  recall  positives")
        for name, m in table.items():
            print(f"{name:12s} {fmt(m['precision'])}    {fmt(m['recall'])}  {m['positives']:5d}")

    errors = Counter()
    for r in rows:
        for code in r["expected"] - r["predicted"]:
            errors[f"missed {code}"] += 1
        for code in r["predicted"] - r["expected"]:
            errors[f"extra {code} on {'good' if r['expected_ok'] else 'bad'} photo"] += 1
    report["top_errors"] = errors.most_common(10)
    print("\nmost frequent errors:")
    for error, count in report["top_errors"]:
        print(f"  {count:3d}  {error}")

    out_dir = os.path.join(ROOT, "eval", "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "predictions.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "kind", "expected", "predicted", "status", "ms"])
        for r in rows:
            writer.writerow([r["file"], r["kind"], ";".join(sorted(r["expected"])), ";".join(sorted(r["predicted"])),
                             r["status"], round(r["seconds"] * 1000)])
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
