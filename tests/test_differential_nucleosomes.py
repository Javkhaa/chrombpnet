from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl
import pyBigWig

from chrombpnet.multitask.differential_nucleosomes import (
    CallConfig,
    estimate_shift,
    normalize_profiles,
    score_profiles,
    validate_adult_cohort,
)
from chrombpnet.multitask.differential_nucleosomes_runner import (
    build_candidate_loci,
    call_differential_nucleosomes,
    read_manifest,
)


class AdultCohortTests(unittest.TestCase):
    def setUp(self):
        self.manifest = pl.DataFrame(
            {
                "cell_type": ["adult_a", "fetal_a", "cell_line", "adult_b"],
                "peaks": ["a.bed", "f.bed", "c.bed", "b.bed"],
                "nonpeaks": ["an.bed", "fn.bed", "cn.bed", "bn.bed"],
                "acc_bw": ["a.acc.bw", "f.acc.bw", "c.acc.bw", "b.acc.bw"],
                "nuc_bw": ["a.nuc.bw", "f.nuc.bw", "c.nuc.bw", "b.nuc.bw"],
            }
        )
        self.metadata = pl.DataFrame(
            {
                "cell_type": ["adult_a", "fetal_a", "cell_line", "adult_b"],
                "lineage": ["immune", "neural", "cell_line", "immune"],
                "is_adult": [True, False, True, True],
                "is_cell_line": [False, False, True, False],
            }
        )

    def test_filters_nonadult_and_cell_line_rows(self):
        cohort = validate_adult_cohort(self.manifest, self.metadata)

        self.assertEqual(cohort["cell_type"].to_list(), ["adult_a", "adult_b"])
        self.assertEqual(cohort["lineage"].to_list(), ["immune", "immune"])

    def test_rejects_manifest_rows_without_metadata(self):
        metadata = self.metadata.filter(pl.col("cell_type") != "adult_b")

        with self.assertRaisesRegex(ValueError, "missing metadata"):
            validate_adult_cohort(self.manifest, metadata)

    def test_rejects_duplicate_metadata_cell_types(self):
        metadata = pl.concat([self.metadata, self.metadata.head(1)])

        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_adult_cohort(self.manifest, metadata)


class ProfileScoringTests(unittest.TestCase):
    def test_normalization_is_sum_preserving_and_handles_empty_profiles(self):
        raw = np.zeros((2, 51), dtype=np.float32)
        raw[0, 25] = 10

        smoothed, normalized = normalize_profiles(raw, sigma=2.0)

        self.assertAlmostEqual(float(smoothed[0].sum()), 10.0, places=5)
        self.assertAlmostEqual(float(normalized[0].sum()), 1.0, places=6)
        self.assertEqual(float(smoothed[1].sum()), 0.0)
        self.assertEqual(float(normalized[1].sum()), 0.0)

    def test_estimate_shift_is_positive_when_target_moves_right(self):
        consensus = np.zeros(51, dtype=np.float32)
        target = np.zeros(51, dtype=np.float32)
        consensus[25] = 1
        target[30] = 1

        self.assertEqual(estimate_shift(target, consensus, max_shift=10), 5)

    def test_lineage_outlier_is_called_with_known_shift(self):
        cell_types = ["a1", "a2", "a3", "a4", "b1"]
        lineages = ["A", "A", "A", "A", "B"]
        raw = np.zeros((5, 51), dtype=np.float32)
        raw[0, 30] = 100
        raw[1:4, 25] = 100
        raw[4, 12] = 100
        config = CallConfig(
            sigma=2.0,
            max_shift=10,
            min_counts=10,
            min_effect=0.05,
            min_robust_z=1.0,
            noise_multiplier=1.0,
            min_lineage_comparators=3,
            seed=7,
        )

        calls, profiles = score_profiles(
            raw,
            cell_types,
            lineages,
            config,
            chrom="chr1",
            center=1_000,
        )

        a1 = calls.filter(pl.col("cell_type") == "a1").row(0, named=True)
        b1 = calls.filter(pl.col("cell_type") == "b1").row(0, named=True)
        self.assertTrue(a1["is_differential"])
        self.assertEqual(a1["lineage_shift_bp"], 5)
        self.assertEqual(a1["lineage_comparators"], 3)
        self.assertFalse(b1["is_differential"])
        self.assertEqual(b1["lineage_comparators"], 0)
        self.assertEqual(profiles["normalized"].shape, raw.shape)

    def test_shallow_comparators_are_excluded_from_lineage_consensus(self):
        raw = np.zeros((5, 51), dtype=np.float32)
        raw[0, 30] = 100
        raw[1:4, 25] = 100
        config = CallConfig(
            sigma=2.0,
            min_counts=10,
            min_effect=0.01,
            min_robust_z=0.0,
            noise_multiplier=0.0,
            min_lineage_comparators=3,
            seed=5,
        )

        calls, _ = score_profiles(
            raw, ["a", "b", "c", "d", "shallow"], ["L"] * 5, config
        )

        self.assertEqual(calls.row(0, named=True)["lineage_comparators"], 3)
        self.assertFalse(
            calls.filter(pl.col("cell_type") == "shallow").row(0, named=True)[
                "passes_depth"
            ]
        )

    def test_shallow_profile_fails_depth_gate_and_noise_is_deterministic(self):
        raw = np.zeros((4, 51), dtype=np.float32)
        raw[0, 30] = 1
        raw[1:, 25] = 100
        config = CallConfig(
            sigma=2.0,
            min_counts=10,
            min_effect=0.01,
            min_robust_z=0.0,
            noise_multiplier=0.0,
            min_lineage_comparators=3,
            seed=11,
        )

        first, _ = score_profiles(raw, ["a", "b", "c", "d"], ["L"] * 4, config)
        second, _ = score_profiles(raw, ["a", "b", "c", "d"], ["L"] * 4, config)

        self.assertFalse(first.row(0, named=True)["passes_depth"])
        self.assertFalse(first.row(0, named=True)["is_differential"])
        np.testing.assert_allclose(
            first["split_half_noise"].to_numpy(),
            second["split_half_noise"].to_numpy(),
        )


def _write_bigwig(path: Path, position: int) -> None:
    with pyBigWig.open(str(path), "w") as bigwig:
        bigwig.addHeader([("chr1", 2_000)])
        bigwig.addEntries(["chr1"], [position], ends=[position + 1], values=[100.0])


def _write_peak(path: Path, center: int, score: float = 10.0) -> None:
    start = center - 50
    path.write_text(
        f"chr1\t{start}\t{start + 100}\tpeak\t1000\t.\t{score}\t10\t10\t50\n"
    )


class ArtifactRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self):
        self.tempdir.cleanup()

    def _write_cohort_inputs(self) -> tuple[Path, Path, Path]:
        rows = []
        metadata = []
        positions = {
            "a1": 1_005,
            "a2": 1_000,
            "a3": 1_000,
            "a4": 1_000,
            "fetal": 980,
            "line": 1_020,
        }
        for cell_type, position in positions.items():
            peak = self.root / f"{cell_type}.peaks.bed"
            acc = self.root / f"{cell_type}.acc.bw"
            nuc = self.root / f"{cell_type}.nuc.bw"
            _write_peak(peak, 1_000)
            _write_bigwig(acc, position)
            _write_bigwig(nuc, position)
            rows.append([cell_type, peak.name, peak.name, acc.name, nuc.name])
            metadata.append(
                {
                    "cell_type": cell_type,
                    "lineage": "A" if cell_type.startswith("a") else "excluded",
                    "is_adult": cell_type != "fetal",
                    "is_cell_line": cell_type == "line",
                }
            )

        manifest_path = self.root / "manifest.tsv"
        manifest_path.write_text("\n".join("\t".join(row) for row in rows) + "\n")
        metadata_path = self.root / "metadata.tsv"
        pl.DataFrame(metadata).write_csv(metadata_path, separator="\t")
        regions_path = self.root / "regions.bed"
        regions_path.write_text("chr1\t975\t1026\tpilot_site\n")
        return manifest_path, metadata_path, regions_path

    def test_manifest_paths_are_resolved_against_root(self):
        manifest_path, _, _ = self._write_cohort_inputs()

        manifest = read_manifest(manifest_path)

        self.assertTrue(Path(manifest.row(0, named=True)["nuc_bw"]).is_absolute())
        self.assertEqual(Path(manifest.row(0, named=True)["nuc_bw"]).parent, self.root)

    def test_candidate_loci_merge_nearby_peak_summits(self):
        first = self.root / "first.bed"
        second = self.root / "second.bed"
        _write_peak(first, 1_000, score=20)
        _write_peak(second, 1_004, score=10)
        cohort = pl.DataFrame(
            {
                "cell_type": ["a", "b"],
                "peaks": [str(first), str(second)],
            }
        )

        loci = build_candidate_loci(
            cohort,
            chroms={"chr1"},
            per_cell_limit=10,
            max_loci=10,
            merge_bp=10,
        )

        self.assertEqual(loci.height, 1)
        self.assertEqual(loci.row(0, named=True)["source_cell_types"], 2)
        self.assertEqual(loci.row(0, named=True)["center"], 1_002)

    def test_end_to_end_runner_writes_adult_only_artifacts(self):
        manifest_path, metadata_path, regions_path = self._write_cohort_inputs()
        output_dir = self.root / "result"
        config = CallConfig(
            sigma=2,
            max_shift=10,
            min_counts=10,
            min_effect=0.05,
            min_robust_z=1,
            noise_multiplier=1,
            min_lineage_comparators=3,
            seed=13,
        )

        artifacts = call_differential_nucleosomes(
            manifest_path=manifest_path,
            metadata_path=metadata_path,
            output_dir=output_dir,
            regions_path=regions_path,
            window=51,
            config=config,
            cache_loci=5,
        )

        calls = pl.read_parquet(artifacts.calls_path)
        summary = json.loads(artifacts.summary_path.read_text())
        cached = np.load(artifacts.profiles_path)
        self.assertEqual(summary["chromosomes"], ["chr1"])
        self.assertEqual(Path(summary["manifest"]), manifest_path.resolve())
        self.assertEqual(Path(summary["metadata"]), metadata_path.resolve())
        self.assertEqual(set(calls["cell_type"].to_list()), {"a1", "a2", "a3", "a4"})
        self.assertEqual(calls.height, 4)
        self.assertTrue(
            calls.filter(pl.col("cell_type") == "a1").row(0, named=True)[
                "is_differential"
            ]
        )
        self.assertEqual(summary["cohort"]["cell_types"], 4)
        self.assertEqual(summary["cohort"]["lineages"], {"A": 4})
        self.assertEqual(cached["raw"].shape, (1, 4, 51))
        self.assertEqual(cached["normalized"].shape, (1, 4, 51))
        self.assertTrue((artifacts.beds_dir / "a1.diffnuc.bed").exists())

    def test_chrom_filter_limits_every_artifact_call_to_requested_chromosomes(self):
        manifest_path, metadata_path, regions_path = self._write_cohort_inputs()
        output_dir = self.root / "filtered-result"

        artifacts = call_differential_nucleosomes(
            manifest_path=manifest_path,
            metadata_path=metadata_path,
            output_dir=output_dir,
            regions_path=regions_path,
            chroms={"chr1"},
            window=51,
            config=CallConfig(
                sigma=2,
                max_shift=10,
                min_counts=10,
                min_effect=0.05,
                min_robust_z=1,
                noise_multiplier=1,
                min_lineage_comparators=3,
                seed=13,
            ),
        )

        calls = pl.read_parquet(artifacts.calls_path)
        self.assertEqual(set(calls["chrom"].to_list()), {"chr1"})


if __name__ == "__main__":
    unittest.main()
