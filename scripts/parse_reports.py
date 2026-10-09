"""CLI utility to parse existing JUnit XML and/or plain-text test logs without rerunning tests."""

import argparse
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.log_parser import RecordExporter, correlate_and_normalize


def main():
    parser = argparse.ArgumentParser(
        description="Parse existing test execution artifacts (JUnit XML and/or text logs) into normalized failure records."
    )
    parser.add_argument("--xml", "-x", type=str, default=None, help="Path to JUnit XML report file.")
    parser.add_argument("--log", "-l", type=str, default=None, help="Path to plain-text test/application log file.")
    parser.add_argument("--output-dir", "-o", type=str, default="data/normalized", help="Output directory for normalized data.")
    parser.add_argument("--base-name", "-b", type=str, default="normalized_results", help="Base file name for exports.")
    parser.add_argument("--no-append", action="store_true", help="Overwrite historical file instead of appending.")
    parser.add_argument("--context-before", type=int, default=3, help="Context lines before an error.")
    parser.add_argument("--context-after", type=int, default=3, help="Context lines after an error.")

    args = parser.parse_args()

    if not args.xml and not args.log:
        print("Error: At least one of --xml or --log must be provided.")
        sys.exit(1)

    print(f"Parsing test artifacts:")
    if args.xml:
        print(f"  - JUnit XML: {args.xml}")
    if args.log:
        print(f"  - Log file:  {args.log}")

    records = correlate_and_normalize(
        junit_path=args.xml,
        log_path=args.log,
        context_lines_before=args.context_before,
        context_lines_after=args.context_after,
    )

    if not records:
        print("Warning: No test records were parsed from the provided input files.")
        sys.exit(0)

    exporter = RecordExporter()
    append_mode = not args.no_append
    outputs = exporter.export_all(
        records=records,
        output_dir=args.output_dir,
        base_name=args.base_name,
        append=append_mode,
    )

    passed_count = sum(1 for r in records if r.status == "passed")
    failed_count = sum(1 for r in records if r.status in ["failed", "error"])
    skipped_count = sum(1 for r in records if r.status == "skipped")

    print("\nNormalized Parsing Summary:")
    print(f"  Total Parsed Records: {len(records)}")
    print(f"  Passed:  {passed_count}")
    print(f"  Failed:  {failed_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Historical Append Mode: {append_mode}")
    print(f"  Exported JSON: {outputs['json']}")
    print(f"  Exported CSV:  {outputs['csv']}")


if __name__ == "__main__":
    main()
