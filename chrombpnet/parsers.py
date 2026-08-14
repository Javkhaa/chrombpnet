import argparse
from argparse import RawTextHelpFormatter


desc = """ChromBPNet preprocessing helpers for the PyTorch nucleosome fork."""


def read_parser():
    parser = argparse.ArgumentParser(description=desc, formatter_class=RawTextHelpFormatter)
    subparsers = parser.add_subparsers(
        help="Supported commands: prep", required=True, dest="cmd"
    )

    prep_parser_full = subparsers.add_parser(
        "prep", help="Generate preprocessing data for PyTorch training"
    )
    prep_parser_sub = prep_parser_full.add_subparsers(
        help="Supported prep commands: nonpeaks, splits", required=True, dest="cmd_prep"
    )

    nonpeaks_parser = prep_parser_sub.add_parser(
        "nonpeaks", help="Generate GC-matched non-peak background regions"
    )
    nonpeaks_parser._action_groups.pop()
    required_nonpeaks_parser = nonpeaks_parser.add_argument_group("required arguments")
    optional_nonpeaks_parser = nonpeaks_parser.add_argument_group("optional arguments")

    required_nonpeaks_parser.add_argument("-g", "--genome", required=True, help="reference genome file")
    required_nonpeaks_parser.add_argument(
        "-o",
        "--output-prefix",
        required=True,
        help="output BED prefix; suffixes are appended by the code",
    )
    required_nonpeaks_parser.add_argument(
        "-p",
        "--peaks",
        type=str,
        required=True,
        help="10-column BED/narrowPeak file of peaks",
    )
    required_nonpeaks_parser.add_argument(
        "-c", "--chrom-sizes", type=str, required=True, help="chrom sizes file"
    )
    required_nonpeaks_parser.add_argument(
        "-fl",
        "--chr-fold-path",
        type=str,
        required=True,
        help="fold JSON with test, valid, and train chromosome lists",
    )
    optional_nonpeaks_parser.add_argument(
        "-il", "--inputlen", type=int, default=2114, help="input length for bins"
    )
    optional_nonpeaks_parser.add_argument(
        "-st", "--stride", type=int, default=1000, help="genomewide bin stride"
    )
    optional_nonpeaks_parser.add_argument(
        "-npr",
        "--neg-to-pos-ratio-train",
        type=int,
        default=2,
        help="negative:positive ratio for training chromosomes",
    )
    optional_nonpeaks_parser.add_argument(
        "-br",
        "--blacklist-regions",
        type=str,
        required=False,
        default=None,
        help="optional 3-column BED/TSV blacklist",
    )
    optional_nonpeaks_parser.add_argument(
        "-s", "--seed", type=int, default=1234, help="seed for sampling nonpeaks"
    )

    splits_parser = prep_parser_sub.add_parser("splits", help="Generate chromosome splits")
    splits_parser._action_groups.pop()
    required_splits_parser = splits_parser.add_argument_group("required arguments")
    required_splits_parser.add_argument(
        "-op",
        "--output_prefix",
        type=str,
        required=True,
        help="path prefix for the fold JSON output",
    )
    required_splits_parser.add_argument(
        "-c",
        "--chrom-sizes",
        type=str,
        required=True,
        help="TSV file with chromosome sizes",
    )
    required_splits_parser.add_argument(
        "-tcr", "--test-chroms", nargs="*", type=str, required=True
    )
    required_splits_parser.add_argument(
        "-vcr", "--valid-chroms", nargs="*", type=str, required=True
    )

    return parser.parse_args()
