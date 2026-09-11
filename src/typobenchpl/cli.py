from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from typobenchpl.corpus import CorpusError, score_stream, verify_cases
from typobenchpl.gate import analyze
from typobenchpl.runner import run_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="typobenchpl",
        description="Deterministic surface-quality benchmark for Polish text.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check", help="return 1 or 0 for one text")
    check_parser.add_argument("input", nargs="?", help="UTF-8 file path, or - for stdin")
    check_parser.add_argument("--text", help="text supplied directly on the command line")
    check_parser.add_argument("--json", action="store_true", help="print diagnostic JSON")

    score_parser = subparsers.add_parser("score", help="score inference texts from JSONL")
    score_parser.add_argument("input", nargs="?", default="-", help="JSONL file, or - for stdin")
    score_parser.add_argument("-n", "--runs", type=int, default=1000)
    score_parser.add_argument("--json", action="store_true", help="print the full summary as JSON")

    verify_parser = subparsers.add_parser("verify", help="verify the benchmark test corpus")
    verify_parser.add_argument(
        "--cases-dir", type=Path, help="directory containing case JSONL files"
    )
    verify_parser.add_argument("--json", action="store_true", help="print the full summary as JSON")

    run_parser = subparsers.add_parser("run", help="generate and score texts with a causal LM")
    run_parser.add_argument("--model", required=True, help="Hugging Face model ID or local path")
    run_parser.add_argument("--suite", default="polish-prose-v1", help="suite name or directory")
    run_parser.add_argument("-n", "--runs", type=int, default=1000)
    run_parser.add_argument("--revision", help="Hugging Face model revision")
    run_parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:N, or mps")
    run_parser.add_argument("--output-dir", type=Path, help="new directory for run artifacts")
    run_parser.add_argument("--json", action="store_true", help="print the full summary as JSON")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        if args.command == "check":
            return _check(args)
        if args.command == "score":
            return _score(args)
        if args.command == "verify":
            return _verify(args)
        return _run_model(args)
    except (CorpusError, OSError, UnicodeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def entrypoint() -> int:
    return main()


def _check(args: argparse.Namespace) -> int:
    if args.text is not None and args.input is not None:
        raise ValueError("use either --text or an input file, not both")

    if args.text is not None:
        value: str | bytes = args.text
    elif args.input is None or args.input == "-":
        value = sys.stdin.buffer.read()
    else:
        value = Path(args.input).read_bytes()

    result = analyze(value)
    if args.json:
        print(json.dumps(result.as_dict(), ensure_ascii=False))
    else:
        print(result.value)
    return 0 if result.passed else 1


def _score(args: argparse.Namespace) -> int:
    if args.input == "-":
        summary = score_stream(sys.stdin, args.runs)
    else:
        with Path(args.input).open("r", encoding="utf-8") as stream:
            summary = score_stream(stream, args.runs)

    if args.json:
        print(json.dumps(summary.as_dict(), ensure_ascii=False, sort_keys=True))
    else:
        print(f"{summary.score:.2f}")
    return 0


def _verify(args: argparse.Namespace) -> int:
    summary = verify_cases(args.cases_dir)
    if args.json:
        print(json.dumps(summary.as_dict(), ensure_ascii=False, sort_keys=True))
    else:
        print(f"{summary.passed}/{summary.total} tests passed")
        for failure in summary.failures:
            print(f"FAIL {failure}", file=sys.stderr)
    return 0 if summary.successful else 1


def _run_model(args: argparse.Namespace) -> int:
    summary = run_benchmark(
        model_reference=args.model,
        suite_reference=args.suite,
        runs=args.runs,
        output_directory=args.output_dir,
        revision=args.revision,
        device=args.device,
    )
    if args.json:
        print(json.dumps(summary.as_dict(), ensure_ascii=False, sort_keys=True))
    else:
        print(f"{summary.score.score:.2f}")
        print(f"results: {summary.output_directory}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(entrypoint())
