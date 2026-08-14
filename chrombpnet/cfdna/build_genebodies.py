"""Build a protein-coding gene-body annotation for block B (genebody_wps).

Input: a GENCODE/Ensembl GTF (gzipped ok). Output: TSV `chrom start end strand name`,
chr-prefixed, protein-coding genes on main chromosomes only. Gene bodies are full
TSS->TES; genebody_wps filters to those >= its window (default 4kb).
"""
import argparse, gzip, re

MAIN = set(f"chr{c}" for c in list(range(1, 23)) + ["X", "Y"])
PAT_NAME = re.compile(r'gene_name "([^"]+)"')
PAT_BT = re.compile(r'gene_(?:bio)?type "([^"]+)"')  # Ensembl gene_biotype / GENCODE gene_type


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gtf", required=True, help="GENCODE/Ensembl GTF (.gtf or .gtf.gz)")
    ap.add_argument("--out", required=True, help="output TSV")
    args = ap.parse_args()

    opn = gzip.open if args.gtf.endswith(".gz") else open
    n = 0
    with opn(args.gtf, "rt") as fh, open(args.out, "w") as w:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t")
            if len(f) < 9 or f[2] != "gene":
                continue
            bt = PAT_BT.search(f[8])
            if not bt or bt.group(1) != "protein_coding":
                continue
            chrom = f[0] if f[0].startswith("chr") else "chr" + f[0]
            if chrom not in MAIN:
                continue
            nm = PAT_NAME.search(f[8])
            name = nm.group(1) if nm else "."
            w.write(f"{chrom}\t{int(f[3])}\t{int(f[4])}\t{f[6]}\t{name}\n")
            n += 1
    print(f"wrote {n} protein-coding genes -> {args.out}")


if __name__ == "__main__":
    main()
