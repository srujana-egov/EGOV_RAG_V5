#!/usr/bin/env python3
"""
Offline evaluation script for DIGIT Studio RAG assistant.

Runs the golden question set through the full RAG pipeline and measures:
  - Retrieval hit rate: was the expected section retrieved?
  - Answer quality: does the answer contain expected keywords?
  - Out-of-domain accuracy: were OOD questions correctly rejected?
  - Latency: per-phase ms

Usage:
    cd /path/to/project
    python eval/run_eval.py [--use-case digit_studio|hcm|all] [--verbose]

Output:
    Prints a summary table + saves results to eval/results_YYYYMMDD_HHMMSS.json
"""
import sys
import os
import json
import time
import argparse
import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


def load_golden_set(use_case: str = "all") -> list:
    golden_path = os.path.join(os.path.dirname(__file__), "golden_set.json")
    with open(golden_path) as f:
        data = json.load(f)
    cases = []
    for uc_name, questions in data["use_cases"].items():
        if use_case == "all" or use_case == uc_name:
            cases.extend(questions)
    return cases


def run_eval(use_case: str = "all", verbose: bool = False):
    from retrieval import hybrid_retrieve_pg, detect_section_hint, multi_query_retrieve
    from generator import stream_rag_pipeline, OUT_OF_DOMAIN_MSG, generate_query_variants

    questions = load_golden_set(use_case)
    results = []
    total = len(questions)

    print(f"\n{'='*60}")
    print(f"DIGIT Studio RAG — Evaluation Run")
    print(f"Use case: {use_case} | Questions: {total}")
    print(f"{'='*60}\n")

    retrieval_hits = 0
    answer_hits = 0
    ood_correct = 0
    ood_total = 0
    latencies = []

    for q_data in questions:
        qid = q_data["id"]
        question = q_data["question"]
        expected_contains = q_data.get("expected_answer_contains", [])
        expected_section = q_data.get("expected_section")
        is_ood = q_data.get("expected_out_of_domain", False)

        if verbose:
            print(f"\n[{qid}] {question}")

        t_start = time.perf_counter()
        timings = {}
        sources = []

        answer_chunks = []
        try:
            for chunk in stream_rag_pipeline(
                query=question,
                hybrid_retrieve_pg=lambda q, k: multi_query_retrieve(
                    generate_query_variants(q), top_k=k,
                    section_hint=detect_section_hint(q)
                ),
                top_k=8,
                model="gpt-4o",
                collected_sources=sources,
                timings=timings,
            ):
                answer_chunks.append(chunk)
        except Exception as e:
            answer_chunks = [f"ERROR: {e}"]

        total_ms = int((time.perf_counter() - t_start) * 1000)
        latencies.append(total_ms)
        answer = "".join(answer_chunks)

        # ── Score ──
        is_ood_response = OUT_OF_DOMAIN_MSG[:40] in answer

        if is_ood:
            ood_total += 1
            ood_ok = is_ood_response
            if ood_ok:
                ood_correct += 1
            status = "✅ OOD" if ood_ok else "❌ OOD_MISS"
        else:
            # Retrieval hit: was expected section in retrieved sources?
            retrieved_sections = [s.get("section", "").lower() for s in sources]
            retrieval_ok = expected_section is None or any(
                expected_section.lower() in sec for sec in retrieved_sections
            )
            if retrieval_ok:
                retrieval_hits += 1

            # Answer quality: do expected keywords appear?
            answer_lower = answer.lower()
            answer_ok = all(kw.lower() in answer_lower for kw in expected_contains) if expected_contains else True
            if answer_ok:
                answer_hits += 1

            status = ("✅" if retrieval_ok and answer_ok
                      else "⚠️ PARTIAL" if retrieval_ok or answer_ok
                      else "❌ MISS")

        result = {
            "id": qid,
            "question": question,
            "status": status,
            "retrieval_ok": retrieval_ok if not is_ood else None,
            "answer_ok": answer_ok if not is_ood else None,
            "ood_correct": ood_ok if is_ood else None,
            "retrieved_sections": retrieved_sections if not is_ood else [],
            "expected_section": expected_section,
            "latency_ms": total_ms,
            "timings": timings,
            "answer_preview": answer[:200],
        }
        results.append(result)

        if verbose:
            print(f"  Status: {status} | Latency: {total_ms}ms")
            print(f"  Retrieved: {retrieved_sections}")
            print(f"  Answer: {answer[:120]}...")

    # ── Summary ──
    non_ood = total - ood_total
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total questions:     {total}")
    print(f"Retrieval hit rate:  {retrieval_hits}/{non_ood} = {retrieval_hits/max(non_ood,1)*100:.0f}%")
    print(f"Answer quality:      {answer_hits}/{non_ood} = {answer_hits/max(non_ood,1)*100:.0f}%")
    if ood_total:
        print(f"OOD accuracy:        {ood_correct}/{ood_total} = {ood_correct/ood_total*100:.0f}%")
    print(f"Avg latency:         {sum(latencies)/len(latencies):.0f}ms")
    print(f"P95 latency:         {sorted(latencies)[int(len(latencies)*0.95)]:.0f}ms")
    print(f"{'='*60}\n")

    # Failed questions
    failed = [r for r in results if "❌" in r["status"] or "⚠️" in r["status"]]
    if failed:
        print(f"⚠️  {len(failed)} questions need attention:")
        for r in failed:
            print(f"  [{r['id']}] {r['status']} — {r['question'][:70]}")
    else:
        print("✅ All questions passed!")

    # Save results
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(os.path.dirname(__file__), f"results_{ts}.json")
    with open(out_path, "w") as f:
        json.dump({
            "run_at": ts,
            "use_case": use_case,
            "summary": {
                "total": total,
                "retrieval_hit_rate": retrieval_hits / max(non_ood, 1),
                "answer_quality_rate": answer_hits / max(non_ood, 1),
                "ood_accuracy": ood_correct / max(ood_total, 1) if ood_total else None,
                "avg_latency_ms": sum(latencies) / len(latencies),
            },
            "results": results,
        }, f, indent=2, default=str)
    print(f"Results saved to: {out_path}\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument("--use-case", default="all", choices=["digit_studio", "hcm", "all"])
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run_eval(use_case=args.use_case, verbose=args.verbose)
