"""Corpus I/O and CLI for adult differential nucleosome-positioning calls."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import polars as pl
import pyBigWig
from tqdm import tqdm

from chrombpnet.multitask.differential_nucleosomes import (
    CallConfig,
    _require_columns,
    normalize_profiles,
    score_profiles,
    validate_adult_cohort,
)


@dataclass(frozen=True)
class CallArtifacts:
    """Paths and counts produced by a completed corpus call."""

    calls_path: Path
    summary_path: Path
    profiles_path: Path
    beds_dir: Path
    n_loci: int
    n_calls: int


def read_manifest(path: str | Path, root: str | Path | None = None) -> pl.DataFrame:
    """Read the five-column multicell manifest and resolve every artifact path."""

    path = Path(path).resolve()
    base = Path(root).resolve() if root is not None else path.parent
    rows: list[dict[str, str]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) != 5:
                raise ValueError(
                    f"manifest line {line_number} needs 5 tab-separated fields, got {len(fields)}"
                )
            cell_type, peaks, nonpeaks, acc_bw, nuc_bw = fields

            def resolve(value: str) -> str:
                candidate = Path(value)
                return str(
                    candidate
                    if candidate.is_absolute()
                    else (base / candidate).resolve()
                )

            rows.append(
                {
                    "cell_type": cell_type,
                    "peaks": resolve(peaks),
                    "nonpeaks": resolve(nonpeaks),
                    "acc_bw": resolve(acc_bw),
                    "nuc_bw": resolve(nuc_bw),
                }
            )
    if not rows:
        raise ValueError(f"manifest has no data rows: {path}")
    return pl.DataFrame(rows)


def read_metadata(path: str | Path) -> pl.DataFrame:
    """Read explicit cell-type cohort metadata from a tab-separated file."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"adult cohort metadata not found: {path}")
    return pl.read_csv(path, separator="\t", infer_schema_length=None)


def _read_peak_summits(path: str | Path, cell_type: str, limit: int) -> pl.DataFrame:
    frame = pl.read_csv(
        path,
        separator="\t",
        has_header=False,
        comment_prefix="#",
        truncate_ragged_lines=True,
        infer_schema_length=1_000,
        raise_if_empty=False,
    )
    if frame.width < 3 or frame.is_empty():
        return pl.DataFrame(
            schema={
                "chrom": pl.String,
                "center": pl.Int64,
                "peak_score": pl.Float64,
                "source_cell_type": pl.String,
            }
        )
    columns = frame.columns
    chrom = pl.col(columns[0]).cast(pl.String).alias("chrom")
    start = pl.col(columns[1]).cast(pl.Int64)
    end = pl.col(columns[2]).cast(pl.Int64)
    if frame.width >= 10:
        center = (start + pl.col(columns[9]).cast(pl.Int64)).alias("center")
    else:
        center = ((start + end) // 2).alias("center")
    score_column = columns[6] if frame.width >= 7 else None
    peak_score = (
        pl.col(score_column).cast(pl.Float64, strict=False).fill_null(0.0)
        if score_column is not None
        else pl.lit(0.0)
    ).alias("peak_score")
    result = frame.select(
        chrom,
        center,
        peak_score,
        pl.lit(cell_type).alias("source_cell_type"),
    ).sort("peak_score", descending=True)
    return result.head(limit) if limit > 0 else result


def build_candidate_loci(
    cohort: pl.DataFrame,
    *,
    chroms: set[str] | None = None,
    per_cell_limit: int = 50,
    max_loci: int = 5_000,
    merge_bp: int = 100,
) -> pl.DataFrame:
    """Build a bounded, support-ranked union of adult peak summits."""

    _require_columns(cohort, {"cell_type", "peaks"}, "cohort")
    if merge_bp <= 0:
        raise ValueError("merge_bp must be positive")
    candidates: list[pl.DataFrame] = []
    for row in cohort.select("cell_type", "peaks").iter_rows(named=True):
        peak_path = Path(row["peaks"])
        if not peak_path.is_file():
            raise FileNotFoundError(
                f"peak file not found for {row['cell_type']}: {peak_path}"
            )
        candidates.append(
            _read_peak_summits(peak_path, row["cell_type"], per_cell_limit)
        )
    if not candidates:
        raise ValueError(
            "no adult peak files were available for candidate construction"
        )
    merged = pl.concat(candidates)
    if chroms:
        merged = merged.filter(pl.col("chrom").is_in(sorted(chroms)))
    if merged.is_empty():
        raise ValueError("no candidate peak summits remain after chromosome filtering")
    loci = (
        merged.with_columns((pl.col("center") // merge_bp).alias("_bin"))
        .group_by("chrom", "_bin")
        .agg(
            pl.col("center").median().round().cast(pl.Int64).alias("center"),
            pl.col("source_cell_type").n_unique().alias("source_cell_types"),
            pl.col("peak_score").max().alias("peak_score"),
        )
        .drop("_bin")
        .sort(
            ["source_cell_types", "peak_score", "chrom", "center"],
            descending=[True, True, False, False],
        )
    )
    return loci.head(max_loci) if max_loci > 0 else loci


def read_region_loci(path: str | Path, chroms: set[str] | None = None) -> pl.DataFrame:
    """Read a BED-like candidate set, using interval midpoints as analysis centers."""

    frame = pl.read_csv(
        path,
        separator="\t",
        has_header=False,
        comment_prefix="#",
        truncate_ragged_lines=True,
        infer_schema_length=None,
    )
    if frame.width < 3:
        raise ValueError(f"regions file needs at least 3 BED columns: {path}")
    columns = frame.columns
    loci = frame.select(
        pl.col(columns[0]).cast(pl.String).alias("chrom"),
        (
            (pl.col(columns[1]).cast(pl.Int64) + pl.col(columns[2]).cast(pl.Int64)) // 2
        ).alias("center"),
        (pl.col(columns[3]).cast(pl.String) if frame.width >= 4 else pl.lit("")).alias(
            "source_name"
        ),
        pl.lit(0).cast(pl.UInt32).alias("source_cell_types"),
        pl.lit(float("nan")).alias("peak_score"),
    ).unique(subset=["chrom", "center"], keep="first", maintain_order=True)
    if chroms:
        loci = loci.filter(pl.col("chrom").is_in(sorted(chroms)))
    if loci.is_empty():
        raise ValueError("regions file has no loci after chromosome filtering")
    return loci


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "unnamed"


def _read_profile_cube(
    handles: list,
    loci: pl.DataFrame,
    window: int,
) -> np.ndarray:
    half = window // 2
    raw = np.zeros((loci.height, len(handles), window), dtype=np.float32)
    rows = list(loci.select("chrom", "center").iter_rows())
    for cell_index, bigwig in enumerate(handles):
        for locus_index, (chrom, center) in enumerate(rows):
            start = int(center) - half
            end = start + window
            values = bigwig.values(chrom, start, end, numpy=True)
            raw[locus_index, cell_index] = np.nan_to_num(values, nan=0.0).astype(
                np.float32, copy=False
            )
    return raw


def _filter_bigwig_bounds(
    loci: pl.DataFrame,
    chrom_sizes: dict[str, int],
    window: int,
) -> pl.DataFrame:
    half = window // 2
    rows = [
        row
        for row in loci.iter_rows(named=True)
        if row["chrom"] in chrom_sizes
        and int(row["center"]) - half >= 0
        and int(row["center"]) - half + window <= chrom_sizes[row["chrom"]]
    ]
    if not rows:
        raise ValueError("no candidate loci fit inside the nucleosome BigWig bounds")
    return pl.DataFrame(rows, schema=loci.schema)


def _atomic_parquet(frame: pl.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.write_parquet(temporary)
    temporary.replace(path)


def _atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_beds(calls: pl.DataFrame, cell_types: list[str], beds_dir: Path) -> None:
    beds_dir.mkdir(parents=True, exist_ok=True)
    selected = calls.filter(pl.col("is_differential"))
    for cell_type in cell_types:
        path = beds_dir / f"{_safe_filename(cell_type)}.diffnuc.bed"
        rows = selected.filter(pl.col("cell_type") == cell_type).sort(
            ["chrom", "center"]
        )
        with path.open("w") as handle:
            for row in rows.iter_rows(named=True):
                handle.write(
                    "\t".join(
                        [
                            row["chrom"],
                            str(max(0, int(row["center"]) - 1)),
                            str(int(row["center"]) + 1),
                            cell_type,
                            row["lineage"],
                            f"{row['lineage_effect']:.6g}",
                            f"{row['lineage_robust_z']:.6g}",
                            str(row["lineage_shift_bp"]),
                            f"{row['split_half_noise']:.6g}",
                        ]
                    )
                    + "\n"
                )


def _cache_profiles(
    path: Path,
    loci: pl.DataFrame,
    raw: np.ndarray,
    cell_types: list[str],
    lineages: list[str],
    sigma: float,
) -> None:
    normalized = np.stack(
        [normalize_profiles(locus, sigma)[1] for locus in raw], axis=0
    ).astype(np.float32)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(
        temporary,
        locus_ids=np.asarray(
            [
                f"{chrom}:{center}"
                for chrom, center in loci.select("chrom", "center").iter_rows()
            ]
        ),
        chroms=np.asarray(loci["chrom"].to_list()),
        centers=loci["center"].to_numpy().astype(np.int64),
        cell_types=np.asarray(cell_types),
        lineages=np.asarray(lineages),
        raw=raw.astype(np.float32),
        normalized=normalized,
    )
    temporary.replace(path)


def call_differential_nucleosomes(
    *,
    manifest_path: str | Path,
    metadata_path: str | Path,
    output_dir: str | Path,
    manifest_root: str | Path | None = None,
    regions_path: str | Path | None = None,
    chroms: set[str] | None = None,
    window: int = 501,
    config: CallConfig | None = None,
    per_cell_peak_limit: int = 50,
    max_loci: int = 5_000,
    merge_bp: int = 100,
    chunk_size: int = 32,
    cache_loci: int = 100,
) -> CallArtifacts:
    """Run adult differential-positioning calls and serialize notebook artifacts."""

    if window < 3 or window % 2 == 0:
        raise ValueError("window must be an odd integer of at least 3 bp")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    config = config or CallConfig()
    if config.max_shift >= window - 1:
        raise ValueError("max_shift must be smaller than the profile window")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(manifest_path, manifest_root)
    metadata = read_metadata(metadata_path)
    cohort = validate_adult_cohort(manifest, metadata)
    for row in cohort.select("cell_type", "nuc_bw").iter_rows(named=True):
        if not Path(row["nuc_bw"]).is_file():
            raise FileNotFoundError(
                f"nucleosome BigWig not found for {row['cell_type']}: {row['nuc_bw']}"
            )

    if regions_path is not None:
        loci = read_region_loci(regions_path, chroms)
        candidate_source = str(Path(regions_path).resolve())
    else:
        loci = build_candidate_loci(
            cohort,
            chroms=chroms,
            per_cell_limit=per_cell_peak_limit,
            max_loci=max_loci,
            merge_bp=merge_bp,
        )
        candidate_source = "adult_peak_union"

    cell_types = cohort["cell_type"].to_list()
    lineages = cohort["lineage"].to_list()
    handles = [pyBigWig.open(path) for path in cohort["nuc_bw"].to_list()]
    try:
        chrom_sizes = handles[0].chroms()
        loci = _filter_bigwig_bounds(loci, chrom_sizes, window)
        call_frames: list[pl.DataFrame] = []
        for offset in tqdm(
            range(0, loci.height, chunk_size),
            desc="adult differential nucleosomes",
            unit="chunk",
        ):
            chunk = loci.slice(offset, chunk_size)
            raw_chunk = _read_profile_cube(handles, chunk, window)
            for local_index, locus in enumerate(chunk.iter_rows(named=True)):
                calls, _ = score_profiles(
                    raw_chunk[local_index],
                    cell_types,
                    lineages,
                    config,
                    chrom=locus["chrom"],
                    center=int(locus["center"]),
                )
                call_frames.append(
                    calls.with_columns(
                        pl.lit(f"{locus['chrom']}:{locus['center']}").alias("locus_id"),
                        pl.lit(locus.get("source_name", "")).alias("source_name"),
                    )
                )
        calls = pl.concat(call_frames)
        calls_path = output_dir / "differential_nucleosomes.parquet"
        _atomic_parquet(calls, calls_path)

        ranked_loci = (
            calls.group_by("locus_id", "chrom", "center")
            .agg(
                pl.col("is_differential").any().alias("has_lineage_call"),
                pl.col("is_global_differential").any().alias("has_global_call"),
                pl.col("lineage_robust_z").max().fill_nan(-1.0).alias("max_lineage_z"),
                pl.col("global_robust_z").max().fill_nan(-1.0).alias("max_global_z"),
            )
            .sort(
                [
                    "has_lineage_call",
                    "has_global_call",
                    "max_lineage_z",
                    "max_global_z",
                ],
                descending=True,
            )
            .head(cache_loci)
        )
        cache_keys = set(ranked_loci["locus_id"].to_list())
        cached_loci = loci.filter(
            pl.concat_str([pl.col("chrom"), pl.lit(":"), pl.col("center")]).is_in(
                cache_keys
            )
        )
        rank_order = {
            value: index
            for index, value in enumerate(ranked_loci["locus_id"].to_list())
        }
        cached_loci = (
            cached_loci.with_columns(
                pl.concat_str([pl.col("chrom"), pl.lit(":"), pl.col("center")])
                .replace_strict(rank_order)
                .alias("_rank")
            )
            .sort("_rank")
            .drop("_rank")
        )
        cached_raw = _read_profile_cube(handles, cached_loci, window)
        profiles_path = output_dir / "profiles.npz"
        _cache_profiles(
            profiles_path,
            cached_loci,
            cached_raw,
            cell_types,
            lineages,
            config.sigma,
        )
    finally:
        for handle in handles:
            handle.close()

    beds_dir = output_dir / "beds"
    _write_beds(calls, cell_types, beds_dir)
    lineage_counts = cohort.group_by("lineage").len().sort("lineage")
    summary = {
        "analysis": "adult_lineage_aware_differential_nucleosome_positioning",
        "manifest": str(Path(manifest_path).resolve()),
        "metadata": str(Path(metadata_path).resolve()),
        "chromosomes": sorted(loci["chrom"].unique().to_list()),
        "candidate_source": candidate_source,
        "cohort": {
            "cell_types": cohort.height,
            "lineages": {
                lineage: int(count) for lineage, count in lineage_counts.iter_rows()
            },
        },
        "parameters": {**asdict(config), "window": window},
        "loci_tested": loci.height,
        "target_locus_tests": calls.height,
        "lineage_calls": int(calls["is_differential"].sum()),
        "global_calls": int(calls["is_global_differential"].sum()),
        "cached_loci": cached_loci.height,
    }
    summary_path = output_dir / "summary.json"
    _atomic_json(summary, summary_path)
    return CallArtifacts(
        calls_path=calls_path,
        summary_path=summary_path,
        profiles_path=profiles_path,
        beds_dir=beds_dir,
        n_loci=loci.height,
        n_calls=summary["lineage_calls"],
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--manifest-root", type=Path)
    parser.add_argument(
        "--regions",
        type=Path,
        help="optional BED candidate set; use the adult peak union when omitted",
    )
    parser.add_argument("--chroms", nargs="+", default=None)
    parser.add_argument("--window", type=int, default=501)
    parser.add_argument("--smooth-sigma", type=float, default=20.0)
    parser.add_argument("--max-shift", type=int, default=100)
    parser.add_argument("--min-counts", type=float, default=20.0)
    parser.add_argument("--min-effect", type=float, default=0.05)
    parser.add_argument("--min-robust-z", type=float, default=3.0)
    parser.add_argument("--noise-multiplier", type=float, default=1.0)
    parser.add_argument("--min-lineage-comparators", type=int, default=3)
    parser.add_argument("--per-cell-peak-limit", type=int, default=50)
    parser.add_argument("--max-loci", type=int, default=5_000)
    parser.add_argument("--merge-bp", type=int, default=100)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--cache-loci", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = CallConfig(
        sigma=args.smooth_sigma,
        max_shift=args.max_shift,
        min_counts=args.min_counts,
        min_effect=args.min_effect,
        min_robust_z=args.min_robust_z,
        noise_multiplier=args.noise_multiplier,
        min_lineage_comparators=args.min_lineage_comparators,
        seed=args.seed,
    )
    artifacts = call_differential_nucleosomes(
        manifest_path=args.manifest,
        metadata_path=args.metadata,
        output_dir=args.out_dir,
        manifest_root=args.manifest_root,
        regions_path=args.regions,
        chroms=set(args.chroms) if args.chroms else None,
        window=args.window,
        config=config,
        per_cell_peak_limit=args.per_cell_peak_limit,
        max_loci=args.max_loci,
        merge_bp=args.merge_bp,
        chunk_size=args.chunk_size,
        cache_loci=args.cache_loci,
    )
    print(
        f"adult differential nucleosomes: loci={artifacts.n_loci} "
        f"lineage_calls={artifacts.n_calls} results={artifacts.calls_path}"
    )


if __name__ == "__main__":
    main()
