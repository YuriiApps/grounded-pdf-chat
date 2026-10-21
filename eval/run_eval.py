"""Run the test questions against the API and write eval/report.md.

    python eval/run_eval.py [--api http://localhost:8000]

Questions with "source": null are NOT in the documents: the correct behaviour is to say "not found".
Metrics:
  - retrieval: the expected document + page is among the cited sources
  - answer:    the answer contains the expected key fact (simple substring check)
  - refusal:   out-of-scope questions are answered with "not found"
"""
import argparse
import json
from pathlib import Path

import requests

HERE = Path(__file__).parent


def evaluate(api: str) -> tuple[list[dict], dict]:
    cases = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))
    rows = []
    for case in cases:
        r = requests.post(f"{api}/chat", json={"question": case["question"]}, timeout=120)
        r.raise_for_status()
        res = r.json()
        in_scope = case["source"] is not None
        cited = {(s["source"], s["page"]) for s in res["sources"]}
        rows.append(
            {
                "question": case["question"],
                "in_scope": in_scope,
                "retrieval_ok": (case["source"], case["page"]) in cited if in_scope else None,
                "answer_ok": (case["expect"].lower() in res["answer"].lower()) if in_scope else None,
                "refusal_ok": None if in_scope else not res["found"],
                "answer": res["answer"],
                "cited": ", ".join(f"{s} p.{p}" for s, p in sorted(cited)) or "-",
            }
        )

    def rate(key: str) -> str:
        vals = [row[key] for row in rows if row[key] is not None]
        return f"{sum(vals)}/{len(vals)}" if vals else "-"

    summary = {"retrieval": rate("retrieval_ok"), "answer": rate("answer_ok"), "refusal": rate("refusal_ok")}
    return rows, summary


def write_report(rows: list[dict], summary: dict) -> Path:
    mark = {True: "✅", False: "❌", None: ""}
    lines = [
        "# Evaluation report",
        "",
        f"- Correct source cited (document + page): **{summary['retrieval']}**",
        f"- Answer contains the expected fact: **{summary['answer']}**",
        f"- Out-of-scope questions answered with \"not found\": **{summary['refusal']}**",
        "",
        "| Question | Source | Fact | Not found | Cited |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['question']} | {mark[row['retrieval_ok']]} | {mark[row['answer_ok']]} "
            f"| {mark[row['refusal_ok']]} | {row['cited']} |"
        )
    path = HERE / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://localhost:8000")
    args = parser.parse_args()
    rows, summary = evaluate(args.api)
    print(json.dumps(summary, indent=2))
    print(f"Report: {write_report(rows, summary)}")
