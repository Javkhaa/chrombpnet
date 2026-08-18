"""Leakage-safe paired locus-context data for differential fine-tuning."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import pyBigWig
import pyfaidx
import torch
from torch.utils.data import Dataset, Sampler

from chrombpnet.training.utils import one_hot


@dataclass(frozen=True)
class ContextUnit:
    locus_id: str
    chrom: str
    center: int
    tissue_context: str
    cell_types: tuple[str, ...]
    checkpoint_indices: tuple[int, ...]
    observed_calls: tuple[bool, ...]


def _require_columns(frame: pl.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {', '.join(missing)}")


def validate_artifact_chromosomes(
    calls: pl.DataFrame,
    *,
    allowed: set[str],
    forbidden: set[str],
    label: str,
) -> None:
    _require_columns(calls, {"chrom"}, f"{label} differential artifact")
    observed = set(calls["chrom"].unique().to_list())
    blocked = observed & forbidden
    outside = observed - allowed
    if blocked or outside:
        raise ValueError(
            f"{label} differential artifact has forbidden={sorted(blocked)} "
            f"outside_allowed={sorted(outside)}"
        )


def map_adult_checkpoint_indices(
    metadata: pl.DataFrame, checkpoint_cell_types: list[str]
) -> dict[str, int]:
    """Map adult, non-cell-line metadata rows to the checkpoint's fixed order."""

    _require_columns(
        metadata,
        {"cell_type", "lineage", "is_adult", "is_cell_line"},
        "metadata",
    )
    if metadata["cell_type"].n_unique() != metadata.height:
        raise ValueError("metadata contains duplicate cell_type rows")
    if len(set(checkpoint_cell_types)) != len(checkpoint_cell_types):
        raise ValueError("checkpoint contains duplicate cell_type entries")

    checkpoint_index = {name: index for index, name in enumerate(checkpoint_cell_types)}
    adults = metadata.filter(pl.col("is_adult") & ~pl.col("is_cell_line"))
    missing = sorted(set(adults["cell_type"].to_list()) - set(checkpoint_index))
    if missing:
        raise ValueError(f"adult metadata cell types absent from checkpoint: {missing}")
    return {
        cell_type: checkpoint_index[cell_type]
        for cell_type in adults["cell_type"].to_list()
    }


def build_context_units(
    calls: pl.DataFrame,
    checkpoint_index: dict[str, int],
    min_members: int = 4,
) -> list[ContextUnit]:
    """Build depth-qualified, checkpoint-ordered locus tissue-context units."""

    if min_members < 1:
        raise ValueError("min_members must be positive")
    _require_columns(
        calls,
        {
            "locus_id",
            "chrom",
            "center",
            "lineage",
            "cell_type",
            "passes_depth",
            "is_differential",
        },
        "differential calls",
    )
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    qualified = calls.filter(pl.col("passes_depth"))
    for row in qualified.iter_rows(named=True):
        key = (str(row["locus_id"]), str(row["lineage"]))
        groups.setdefault(key, []).append(row)

    units: list[ContextUnit] = []
    for (locus_id, tissue_context), rows in groups.items():
        missing = sorted(
            {str(row["cell_type"]) for row in rows}.difference(checkpoint_index)
        )
        if missing:
            raise ValueError(f"depth-qualified calls absent from checkpoint: {missing}")
        rows.sort(key=lambda row: checkpoint_index[str(row["cell_type"])])
        if len(rows) < min_members:
            continue
        chroms = {str(row["chrom"]) for row in rows}
        centers = {int(row["center"]) for row in rows}
        if len(chroms) != 1 or len(centers) != 1:
            raise ValueError(
                f"locus-context {locus_id!r}, {tissue_context!r} is inconsistent"
            )
        units.append(
            ContextUnit(
                locus_id=locus_id,
                chrom=chroms.pop(),
                center=centers.pop(),
                tissue_context=tissue_context,
                cell_types=tuple(str(row["cell_type"]) for row in rows),
                checkpoint_indices=tuple(
                    checkpoint_index[str(row["cell_type"])] for row in rows
                ),
                observed_calls=tuple(bool(row["is_differential"]) for row in rows),
            )
        )
    return units


@dataclass(frozen=True)
class PairedContextExample:
    sequence: torch.Tensor
    raw_profiles: torch.Tensor
    checkpoint_indices: torch.Tensor
    observed_calls: torch.Tensor
    deviation_slice: slice
    locus_id: str
    tissue_context: str


class PairedContextDataset(Dataset):
    """Load one sequence and all depth-qualified adult profiles for a context."""

    def __init__(
        self,
        units: Sequence[ContextUnit],
        genome_fasta: str | Path,
        nuc_bw_files: Mapping[str, str | Path],
        checkpoint_inputlen: int,
        checkpoint_outputlen: int,
    ) -> None:
        if checkpoint_inputlen <= 0:
            raise ValueError("checkpoint_inputlen must be positive")
        if checkpoint_outputlen < 501:
            raise ValueError("checkpoint_outputlen must be at least 501")
        self.units = tuple(units)
        self.genome_fasta = str(genome_fasta)
        self.nuc_bw_files = {name: str(path) for name, path in nuc_bw_files.items()}
        self.checkpoint_inputlen = checkpoint_inputlen
        self.checkpoint_outputlen = checkpoint_outputlen
        self._genome = None
        self._nuc_bws = None

    def __len__(self) -> int:
        return len(self.units)

    def _handles(self):
        if self._genome is None:
            self._genome = pyfaidx.Fasta(self.genome_fasta)
        if self._nuc_bws is None:
            needed = {cell for unit in self.units for cell in unit.cell_types}
            missing = sorted(needed.difference(self.nuc_bw_files))
            if missing:
                raise ValueError(
                    f"missing nucleosome BigWigs for cell types: {missing}"
                )
            self._nuc_bws = {
                cell: pyBigWig.open(self.nuc_bw_files[cell]) for cell in needed
            }
        return self._genome, self._nuc_bws

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_genome"] = None
        state["_nuc_bws"] = None
        return state

    @staticmethod
    def _window_start(center: int, width: int) -> int:
        return center - width // 2

    def __getitem__(self, index: int) -> PairedContextExample:
        unit = self.units[index]
        genome, nuc_bws = self._handles()
        sequence_start = self._window_start(unit.center, self.checkpoint_inputlen)
        sequence_end = sequence_start + self.checkpoint_inputlen
        if sequence_start < 0 or sequence_end > len(genome[unit.chrom]):
            raise ValueError(
                f"sequence window exceeds genome bounds for {unit.locus_id}"
            )
        sequence = str(genome[unit.chrom][sequence_start:sequence_end]).upper()
        sequence_array = one_hot.dna_to_one_hot([sequence])[0].astype(np.float32)

        profile_start = self._window_start(unit.center, self.checkpoint_outputlen)
        profile_end = profile_start + self.checkpoint_outputlen
        profiles = []
        for cell_type in unit.cell_types:
            bigwig = nuc_bws[cell_type]
            chrom_size = bigwig.chroms().get(unit.chrom)
            if chrom_size is None or profile_start < 0 or profile_end > chrom_size:
                raise ValueError(
                    f"profile window exceeds BigWig bounds for {unit.locus_id}"
                )
            values = bigwig.values(unit.chrom, profile_start, profile_end)
            profiles.append(np.nan_to_num(np.asarray(values, dtype=np.float32)))
        start = self.checkpoint_outputlen // 2 - 501 // 2
        return PairedContextExample(
            sequence=torch.from_numpy(sequence_array),
            raw_profiles=torch.from_numpy(np.stack(profiles)),
            checkpoint_indices=torch.tensor(unit.checkpoint_indices, dtype=torch.long),
            observed_calls=torch.tensor(unit.observed_calls, dtype=torch.bool),
            deviation_slice=slice(start, start + 501),
            locus_id=unit.locus_id,
            tissue_context=unit.tissue_context,
        )


class ContextBalancedSampler(Sampler[int]):
    """Sample tissue contexts uniformly, then units within each context uniformly."""

    def __init__(
        self,
        dataset_or_units: PairedContextDataset | Sequence[ContextUnit],
        *,
        num_samples: int | None = None,
        seed: int = 0,
    ) -> None:
        units = getattr(dataset_or_units, "units", dataset_or_units)
        self.units = tuple(units)
        if not self.units:
            raise ValueError("context-balanced sampling requires at least one unit")
        self.num_samples = len(self.units) if num_samples is None else num_samples
        if self.num_samples < 0:
            raise ValueError("num_samples must be non-negative")
        self.seed = seed
        by_context: dict[str, list[int]] = {}
        for index, unit in enumerate(self.units):
            by_context.setdefault(unit.tissue_context, []).append(index)
        self._context_indices = tuple(by_context.values())

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        for _ in range(self.num_samples):
            context = self._context_indices[rng.integers(len(self._context_indices))]
            yield context[rng.integers(len(context))]

    def __len__(self) -> int:
        return self.num_samples
