import copy
import os

import pandas as pd

import chrombpnet.parsers as parsers


def main():
    args = parsers.read_parser()

    if args.cmd != "prep":
        raise SystemExit("Only `chrombpnet prep` is supported in the PyTorch nucleosome fork.")

    if args.cmd_prep == "nonpeaks":
        assert args.inputlen % 2 == 0
        os.makedirs(args.output_prefix + "_auxiliary/", exist_ok=False)

        from chrombpnet.helpers.make_gc_matched_negatives.get_genomewide_gc_buckets.get_genomewide_gc_bins import get_genomewide_gc
        get_genomewide_gc(
            args.genome,
            args.output_prefix + "_auxiliary/genomewide_gc.bed",
            args.inputlen,
            args.stride,
        )

        import chrombpnet.helpers.make_gc_matched_negatives.get_gc_content as get_gc_content
        args_copy = copy.deepcopy(args)
        args_copy.input_bed = args_copy.peaks
        args_copy.output_prefix = args.output_prefix + "_auxiliary/foreground.gc"
        get_gc_content.main(args_copy)

        os.system(
            "bedtools slop -i {peaks} -g {chrom_sizes} -b {flank_size} > {output}".format(
                peaks=args.peaks,
                chrom_sizes=args.chrom_sizes,
                flank_size=args.inputlen // 2,
                output=args.output_prefix + "_auxiliary/peaks_slop.bed",
            )
        )
        exclude_bed = pd.read_csv(
            args.output_prefix + "_auxiliary/peaks_slop.bed",
            sep="\t",
            header=None,
            usecols=[0, 1, 2],
        )

        if args.blacklist_regions:
            os.system(
                "bedtools slop -i {blacklist} -g {chrom_sizes} -b {flank_size} > {output}".format(
                    blacklist=args.blacklist_regions,
                    chrom_sizes=args.chrom_sizes,
                    flank_size=args.inputlen // 2,
                    output=args.output_prefix + "_auxiliary/blacklist_slop.bed",
                )
            )
            exclude_bed = pd.concat([
                exclude_bed,
                pd.read_csv(
                    args.output_prefix + "_auxiliary/blacklist_slop.bed",
                    sep="\t",
                    header=None,
                    usecols=[0, 1, 2],
                ),
            ])

        exclude_bed.to_csv(
            args.output_prefix + "_auxiliary/exclude_unmerged.bed",
            sep="\t",
            header=False,
            index=False,
        )
        os.system(
            "bedtools sort -i {inputb} | bedtools merge -i stdin > {output}".format(
                inputb=args.output_prefix + "_auxiliary/exclude_unmerged.bed",
                output=args.output_prefix + "_auxiliary/exclude.bed",
            )
        )
        os.system(
            "bedtools intersect -v -a {genomewide_gc} -b {exclude_bed} > {candidate_bed}".format(
                genomewide_gc=args.output_prefix + "_auxiliary/genomewide_gc.bed",
                exclude_bed=args.output_prefix + "_auxiliary/exclude.bed",
                candidate_bed=args.output_prefix + "_auxiliary/candidates.bed",
            )
        )

        import chrombpnet.helpers.make_gc_matched_negatives.get_gc_matched_negatives as get_gc_matched_negatives
        args_copy = copy.deepcopy(args)
        args_copy.candidate_negatives = args.output_prefix + "_auxiliary/candidates.bed"
        args_copy.foreground_gc_bed = args.output_prefix + "_auxiliary/foreground.gc.bed"
        args_copy.output_prefix = args.output_prefix + "_auxiliary/negatives"
        get_gc_matched_negatives.main(args_copy)

        negatives = pd.read_csv(args.output_prefix + "_auxiliary/negatives.bed", sep="\t", header=None)
        negatives[3] = "."
        negatives[4] = "."
        negatives[5] = "."
        negatives[6] = "."
        negatives[7] = "."
        negatives[8] = "."
        negatives[9] = args.inputlen // 2
        negatives.to_csv(args.output_prefix + "_negatives.bed", sep="\t", header=False, index=False)
    elif args.cmd_prep == "splits":
        import chrombpnet.helpers.make_chr_splits.splits as splits
        splits.main(args)
    else:
        raise SystemExit(f"Unsupported prep command: {args.cmd_prep}")


if __name__ == "__main__":
    main()
