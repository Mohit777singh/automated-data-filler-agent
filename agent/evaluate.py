"""
Evaluates the extraction agent against labeled test data.

Run from the project root:
    python -m agent.evaluate

What it measures:
1. Field-level accuracy on `synthetic` test rows (full ground truth) --
   for each field, did the model extract exactly the right value?
2. Correct null-handling on `sales_conversations` / `samsum` test rows --
   did the model correctly avoid hallucinating company/contact/deal info
   that genuinely wasn't in the (mostly non-business) text?

Prints a summary report at the end. Also saves full per-example results
to eval_results.json for closer inspection.
"""

import json
import sys
from pathlib import Path

from agent.extractor_ollama import extract_crm_data

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_PATH = PROJECT_ROOT / "data" / "merged_test.jsonl"

# How many examples to test from each category. Set to cover the full
# labeled test set (42 synthetic rows) plus a solid sparse sample for the
# hallucination check. Local inference is slow -- this run will take a
# while (dozens of extractions), but gives real, reportable accuracy
# numbers rather than a tiny/anecdotal sample.
N_LABELED = 42   # all synthetic (fully labeled) test rows
N_SPARSE = 40    # sparse (sales_conversations/samsum) test rows

FLAT_FIELDS = [
    ("Contact", "FirstName"), ("Contact", "LastName"), ("Contact", "Title"),
    ("Contact", "Department"), ("Contact", "Email"),
    ("Account", "Name"), ("Account", "Industry"), ("Account", "NumberOfEmployees"),
    ("Opportunity", "Name"), ("Opportunity", "Amount"), ("Opportunity", "StageName"),
    ("Opportunity", "Competitor__c"),
]

# Fields NOT guaranteed to be literally stated in the synthetic generator's
# text (e.g. Email is fabricated but never mentioned in the summary itself;
# NumberOfEmployees ground truth is one specific number picked from a range
# that's all the text actually states; StageName is assigned independent of
# the text; Department is inferred from Title, not stated directly).
# A correct, non-hallucinating agent should score LOW on these -- it's
# supposed to say null when the source text doesn't contain the answer.
# Reported separately so raw accuracy isn't misleadingly dragged down by
# a dataset-generation artifact rather than a real extraction failure.
NOT_RELIABLY_STATED_IN_TEXT = {("Contact", "Email"), ("Contact", "Department"),
                                 ("Account", "NumberOfEmployees"), ("Opportunity", "StageName")}


def normalize(v):
    if isinstance(v, str):
        return v.strip().lower()
    return v


def load_test_rows():
    with open(TEST_PATH) as f:
        rows = [json.loads(l) for l in f]
    labeled = [r for r in rows if r["source"] == "synthetic"][:N_LABELED]
    sparse = [r for r in rows if r["source"] in ("sales_conversations", "samsum")][:N_SPARSE]
    return labeled, sparse


def evaluate_labeled(rows):
    """Field-by-field exact-match accuracy against full ground truth."""
    field_correct = {f: 0 for f in FLAT_FIELDS}
    field_total = {f: 0 for f in FLAT_FIELDS}
    results = []

    for i, row in enumerate(rows, 1):
        print(f"  [{i}/{len(rows)}] extracting...", file=sys.stderr)
        try:
            pred = extract_crm_data(row["meeting_summary"])
        except RuntimeError as e:
            print(f"  extraction failed: {e}", file=sys.stderr)
            continue

        truth = row["salesforce_record"]
        row_result = {"summary": row["meeting_summary"], "predicted": pred, "truth": truth, "matches": {}}

        for obj, field in FLAT_FIELDS:
            truth_val = normalize(truth.get(obj, {}).get(field))
            pred_val = normalize(pred.get(obj, {}).get(field))
            match = truth_val == pred_val
            row_result["matches"][f"{obj}.{field}"] = match
            field_total[(obj, field)] += 1
            if match:
                field_correct[(obj, field)] += 1

        results.append(row_result)

    return field_correct, field_total, results


def evaluate_sparse(rows):
    """
    Checks correct null-handling: for sparse/non-business text, did the
    model avoid inventing a company name, contact name, or deal amount?
    """
    correct_nulls = 0
    total = 0
    results = []

    for i, row in enumerate(rows, 1):
        print(f"  [{i}/{len(rows)}] extracting (sparse)...", file=sys.stderr)
        try:
            pred = extract_crm_data(row["meeting_summary"])
        except RuntimeError as e:
            print(f"  extraction failed: {e}", file=sys.stderr)
            continue

        # These fields should almost always be null for sparse/general dialogue
        no_hallucination = (
            pred.get("Account", {}).get("Name") is None
            and pred.get("Opportunity", {}).get("Amount") is None
        )
        total += 1
        if no_hallucination:
            correct_nulls += 1

        results.append({
            "summary": row["meeting_summary"][:150],
            "predicted_account_name": pred.get("Account", {}).get("Name"),
            "predicted_amount": pred.get("Opportunity", {}).get("Amount"),
            "correctly_avoided_hallucination": no_hallucination,
        })

    return correct_nulls, total, results


def main():
    labeled_rows, sparse_rows = load_test_rows()
    print(f"Evaluating on {len(labeled_rows)} labeled + {len(sparse_rows)} sparse examples...\n")

    print("=== Part 1: Field accuracy on labeled (synthetic) examples ===")
    field_correct, field_total, labeled_results = evaluate_labeled(labeled_rows)

    print("\nField-by-field accuracy:")
    total_correct, total_fields = 0, 0
    adjusted_correct, adjusted_fields = 0, 0
    for (obj, field), total in field_total.items():
        correct = field_correct[(obj, field)]
        total_correct += correct
        total_fields += total
        flag = ""
        if (obj, field) in NOT_RELIABLY_STATED_IN_TEXT:
            flag = "  [not reliably stated in source text -- see note below]"
        else:
            adjusted_correct += correct
            adjusted_fields += total
        pct = (correct / total * 100) if total else 0
        print(f"  {obj}.{field:<20} {correct}/{total}  ({pct:.0f}%){flag}")

    overall_pct = (total_correct / total_fields * 100) if total_fields else 0
    adjusted_pct = (adjusted_correct / adjusted_fields * 100) if adjusted_fields else 0
    print(f"\nRAW OVERALL ACCURACY (all fields): {total_correct}/{total_fields} ({overall_pct:.1f}%)")
    print(f"ADJUSTED ACCURACY (fields actually stated in text): "
          f"{adjusted_correct}/{adjusted_fields} ({adjusted_pct:.1f}%)")
    print("\nNote: Email, Department, NumberOfEmployees (exact), and StageName are")
    print("often not literally present in the source meeting summary text (the")
    print("synthetic generator sets these independent of what the text states).")
    print("A correct, non-hallucinating agent should score LOW here by returning")
    print("null rather than guessing -- so the RAW score understates real quality.")
    print("The ADJUSTED score reflects accuracy on fields genuinely extractable")
    print("from the input text, which is the fairer measure of extraction quality.")

    print("\n=== Part 2: Null-handling on sparse (non-labeled) examples ===")
    correct_nulls, total_sparse, sparse_results = evaluate_sparse(sparse_rows)
    sparse_pct = (correct_nulls / total_sparse * 100) if total_sparse else 0
    print(f"Correctly avoided hallucination: {correct_nulls}/{total_sparse} ({sparse_pct:.1f}%)")

    # Save detailed results for inspection
    out_path = PROJECT_ROOT / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "field_accuracy": {f"{o}.{fld}": {"correct": field_correct[(o, fld)], "total": field_total[(o, fld)]}
                                 for o, fld in FLAT_FIELDS},
            "overall_field_accuracy_pct": overall_pct,
            "adjusted_field_accuracy_pct": adjusted_pct,
            "null_handling_accuracy_pct": sparse_pct,
            "labeled_results": labeled_results,
            "sparse_results": sparse_results,
        }, f, indent=2)
    print(f"\nDetailed results saved to {out_path}")

    print("\n=== Copy-paste summary for README / slides ===")
    print(f"Field accuracy (fields stated in text): {adjusted_pct:.1f}% ({adjusted_correct}/{adjusted_fields})")
    print(f"Field accuracy (all fields, incl. unstated): {overall_pct:.1f}% ({total_correct}/{total_fields})")
    print(f"Hallucination-avoidance rate: {sparse_pct:.1f}% ({correct_nulls}/{total_sparse} sparse examples)")


if __name__ == "__main__":
    main()
