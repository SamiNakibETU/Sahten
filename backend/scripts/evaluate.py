#!/usr/bin/env python
"""
Sahten Evaluation Script
========================

Runs the golden dataset and computes quality metrics:
- Constraint pass rate
- Intent accuracy
- Precision@k
- Recall@k
- Keyword hit rate

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --output results.json
    python scripts/evaluate.py --verbose
"""

from __future__ import annotations

import asyncio
import json
import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_golden_dataset() -> dict:
    """Load the golden evaluation dataset."""
    p = Path(__file__).parent.parent / "tests" / "golden_dataset.json"
    if not p.exists():
        # Fallback to test_matrix.json
        p = Path(__file__).parent.parent / "tests" / "test_matrix.json"
    return json.loads(p.read_text(encoding="utf-8"))


def extract_urls_from_response(response) -> List[str]:
    """Extract recipe URLs from SahtenResponse."""
    urls = []
    for recipe in response.recipes:
        if recipe.url:
            urls.append(recipe.url)
    return urls


def extract_categories_from_response(response) -> List[str]:
    """Extract recipe categories from SahtenResponse."""
    categories = []
    for recipe in response.recipes:
        if recipe.category:
            categories.append(recipe.category.lower())
    return categories


def check_constraints(case: dict, response, urls: List[str], categories: List[str]) -> Dict[str, bool]:
    """Check all constraints for a test case."""
    constraints = case.get("constraints", {})
    results = {}
    
    # Response type constraint
    if "response_type" in constraints:
        results["response_type"] = response.response_type == constraints["response_type"]
    
    # Min recipes
    if "min_recipes" in constraints:
        results["min_recipes"] = response.recipe_count >= int(constraints["min_recipes"])
    
    # Domain constraint
    if "must_include_domain" in constraints:
        dom = constraints["must_include_domain"]
        results["must_include_domain"] = any(dom in u for u in urls)
    
    # No duplicate URLs
    if constraints.get("no_duplicate_urls"):
        olj_urls = [u for u in urls if "lorientlejour.com" in u]
        results["no_duplicate_urls"] = len(olj_urls) == len(set(olj_urls))
    
    # Category constraints
    if "all_categories_include" in constraints:
        needle = str(constraints["all_categories_include"]).lower()
        if categories:
            results["all_categories_include"] = all(needle in c for c in categories)
        else:
            results["all_categories_include"] = False
    
    # Must have certain categories (for menu)
    if "must_have_categories" in constraints:
        expected = set(constraints["must_have_categories"])
        actual = set(categories)
        results["must_have_categories"] = bool(expected & actual)
    
    return results


def compute_keyword_hits(case: dict, response) -> float:
    """
    Compute keyword hit rate.
    Returns percentage of expected keywords found in response content.
    """
    keywords = case.get("relevant_keywords", [])
    if not keywords:
        return 1.0  # No keywords to check = pass
    
    # Build searchable text from response
    search_text = ""
    for recipe in response.recipes:
        search_text += f" {recipe.title or ''} "
        if recipe.cited_passage:
            search_text += f" {recipe.cited_passage} "
    
    search_text = search_text.lower()
    
    hits = sum(1 for kw in keywords if kw.lower() in search_text)
    return hits / len(keywords)


def compute_precision_at_k(expected_urls: List[str], actual_urls: List[str], k: int) -> float:
    """
    Compute Precision@K.
    Fraction of retrieved documents in top-k that are relevant.
    """
    if not expected_urls or not actual_urls:
        return 0.0 if expected_urls else 1.0  # If no expected, consider it a pass
    
    actual_top_k = actual_urls[:k]
    relevant = sum(1 for url in actual_top_k if url in expected_urls)
    return relevant / len(actual_top_k) if actual_top_k else 0.0


def compute_recall_at_k(expected_urls: List[str], actual_urls: List[str], k: int) -> float:
    """
    Compute Recall@K.
    Fraction of relevant documents that appear in top-k.
    """
    if not expected_urls:
        return 1.0  # If no expected, consider it a pass
    
    actual_top_k = set(actual_urls[:k])
    relevant = sum(1 for url in expected_urls if url in actual_top_k)
    return relevant / len(expected_urls)


async def evaluate_case(bot, case: dict, verbose: bool = False) -> Dict[str, Any]:
    """Evaluate a single test case."""
    query = case["query"]
    case_id = case["id"]
    
    try:
        response, _ = await bot.chat(query, debug=True)
        
        urls = extract_urls_from_response(response)
        categories = extract_categories_from_response(response)
        
        # Check constraints
        constraint_results = check_constraints(case, response, urls, categories)
        all_passed = all(constraint_results.values()) if constraint_results else True
        
        # Compute metrics
        expected_urls = case.get("expected_urls", [])
        precision_1 = compute_precision_at_k(expected_urls, urls, 1)
        precision_3 = compute_precision_at_k(expected_urls, urls, 3)
        recall_3 = compute_recall_at_k(expected_urls, urls, 3)
        keyword_hit_rate = compute_keyword_hits(case, response)
        
        # Intent accuracy
        expected_intent = case.get("intent")
        intent_match = response.intent_detected == expected_intent if expected_intent else True
        
        result = {
            "case_id": case_id,
            "query": query,
            "passed": all_passed,
            "response_type": response.response_type,
            "intent_detected": response.intent_detected,
            "intent_match": intent_match,
            "recipe_count": response.recipe_count,
            "constraint_results": constraint_results,
            "metrics": {
                "precision@1": precision_1,
                "precision@3": precision_3,
                "recall@3": recall_3,
                "keyword_hit_rate": keyword_hit_rate,
            },
            "urls": urls,
            "categories": categories,
            "model_used": response.model_used,
        }
        
        if verbose:
            status = "✓ PASS" if all_passed else "✗ FAIL"
            print(f"  [{status}] {case_id}: {query[:40]}...")
            if not all_passed:
                for k, v in constraint_results.items():
                    if not v:
                        print(f"       ↳ {k}: FAILED")
        
        return result
        
    except Exception as e:
        if verbose:
            print(f"  [✗ ERROR] {case_id}: {str(e)[:60]}...")
        
        return {
            "case_id": case_id,
            "query": query,
            "passed": False,
            "error": str(e),
            "metrics": {
                "precision@1": 0.0,
                "precision@3": 0.0,
                "recall@3": 0.0,
                "keyword_hit_rate": 0.0,
            },
        }


async def run_evaluation(verbose: bool = False) -> Dict[str, Any]:
    """Run full evaluation suite."""
    from app.bot import get_bot
    
    print("=" * 60)
    print("SAHTEN EVALUATION")
    print("=" * 60)
    print()
    
    # Load dataset
    dataset = load_golden_dataset()
    cases = dataset.get("cases", [])
    
    print(f"Dataset version: {dataset.get('version', 'unknown')}")
    print(f"Test cases: {len(cases)}")
    print()
    
    # Initialize bot
    print("Initializing bot...")
    bot = get_bot()
    print(f"Bot ready. Model: {bot.default_model}")
    print()
    
    # Run evaluations
    print("Running evaluations:")
    results = []
    
    for case in cases:
        result = await evaluate_case(bot, case, verbose=verbose)
        results.append(result)
    
    print()
    
    # Aggregate metrics
    total = len(results)
    passed = sum(1 for r in results if r.get("passed", False))
    failed = total - passed
    
    # Intent accuracy
    intent_matches = [r for r in results if r.get("intent_match") is not None]
    intent_accuracy = sum(1 for r in intent_matches if r["intent_match"]) / len(intent_matches) if intent_matches else 0
    
    # Average metrics
    avg_precision_1 = sum(r["metrics"]["precision@1"] for r in results) / total if total > 0 else 0
    avg_precision_3 = sum(r["metrics"]["precision@3"] for r in results) / total if total > 0 else 0
    avg_recall_3 = sum(r["metrics"]["recall@3"] for r in results) / total if total > 0 else 0
    avg_keyword_hit = sum(r["metrics"]["keyword_hit_rate"] for r in results) / total if total > 0 else 0
    
    summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "dataset_version": dataset.get("version", "unknown"),
        "total_cases": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / total * 100, 1) if total > 0 else 0,
        "intent_accuracy": round(intent_accuracy * 100, 1),
        "avg_metrics": {
            "precision@1": round(avg_precision_1, 3),
            "precision@3": round(avg_precision_3, 3),
            "recall@3": round(avg_recall_3, 3),
            "keyword_hit_rate": round(avg_keyword_hit, 3),
        },
        "results": results,
    }
    
    # Print summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Pass rate: {passed}/{total} ({summary['pass_rate']}%)")
    print(f"Intent accuracy: {summary['intent_accuracy']}%")
    print(f"Avg Precision@1: {summary['avg_metrics']['precision@1']}")
    print(f"Avg Precision@3: {summary['avg_metrics']['precision@3']}")
    print(f"Avg Recall@3: {summary['avg_metrics']['recall@3']}")
    print(f"Avg Keyword Hit Rate: {summary['avg_metrics']['keyword_hit_rate']}")
    print()
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run Sahten evaluation suite")
    parser.add_argument("--output", "-o", help="Output JSON file for results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    # Run evaluation
    summary = asyncio.run(run_evaluation(verbose=args.verbose))
    
    # Save results if output specified
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Results saved to: {output_path}")
    
    # Exit code based on pass rate
    if summary["pass_rate"] < 80:
        print("\n⚠️  Evaluation FAILED (pass rate < 80%)")
        sys.exit(1)
    else:
        print("\n✓ Evaluation PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
