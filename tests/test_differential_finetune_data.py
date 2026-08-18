from __future__ import annotations

import unittest

import polars as pl

from chrombpnet.multitask.differential_finetune_data import (
    ContextUnit,
    build_context_units,
    map_adult_checkpoint_indices,
    validate_artifact_chromosomes,
)


class ContextUnitTests(unittest.TestCase):
    def test_rejects_test_chromosome_in_training_artifact(self):
        calls = pl.DataFrame({"chrom": ["chr2", "chr1"]})

        with self.assertRaisesRegex(ValueError, "forbidden.*chr1"):
            validate_artifact_chromosomes(
                calls,
                allowed={"chr2", "chr4"},
                forbidden={"chr1", "chr3", "chr6"},
                label="train",
            )

    def test_maps_adults_to_checkpoint_order(self):
        metadata = pl.DataFrame(
            {
                "cell_type": ["adult_b", "adult_a"],
                "lineage": ["brain", "brain"],
                "is_adult": [True, True],
                "is_cell_line": [False, False],
            }
        )

        self.assertEqual(
            map_adult_checkpoint_indices(metadata, ["fetal", "adult_a", "adult_b"]),
            {"adult_b": 2, "adult_a": 1},
        )

    def test_build_context_units_requires_four_depth_qualified_members(self):
        calls = pl.DataFrame(
            {
                "locus_id": ["kept"] * 4 + ["dropped"] * 3,
                "chrom": ["chr2"] * 7,
                "center": [1_000] * 7,
                "lineage": ["brain"] * 7,
                "cell_type": ["a", "b", "c", "d", "a", "b", "c"],
                "passes_depth": [True] * 7,
                "is_differential": [True, False, True, False, True, True, False],
            }
        )

        units = build_context_units(calls, {"a": 4, "b": 1, "c": 3, "d": 2})

        self.assertEqual(len(units), 1)
        self.assertEqual(units[0].locus_id, "kept")
        self.assertEqual(units[0].cell_types, ("b", "d", "c", "a"))
        self.assertEqual(units[0].checkpoint_indices, (1, 2, 3, 4))
        self.assertEqual(units[0].observed_calls, (False, False, True, True))


class PairedContextDatasetTests(unittest.TestCase):
    def test_returns_one_sequence_and_raw_profiles_for_a_context_unit(self):
        import tempfile
        from pathlib import Path

        import pyBigWig

        from chrombpnet.multitask.differential_finetune_data import (
            PairedContextDataset,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            genome = root / "genome.fa"
            genome.write_text(">chr2\n" + "ACGT" * 1_000 + "\n")
            tracks = {}
            for index, cell_type in enumerate(("a", "b", "c", "d"), start=1):
                path = root / f"{cell_type}.bw"
                with pyBigWig.open(str(path), "w") as bigwig:
                    bigwig.addHeader([("chr2", 4_000)])
                    bigwig.addEntries(
                        ["chr2"], [0], ends=[4_000], values=[float(index)]
                    )
                tracks[cell_type] = path
            unit = ContextUnit(
                "chr2:2000",
                "chr2",
                2_000,
                "brain",
                ("a", "b", "c", "d"),
                (1, 2, 3, 4),
                (True, False, True, False),
            )
            dataset = PairedContextDataset([unit], genome, tracks, 512, 503)
            self.assertIsNone(dataset._genome)
            self.assertIsNone(dataset._nuc_bws)
            example = dataset[0]

        self.assertEqual(example.sequence.shape, (512, 4))
        self.assertEqual(example.raw_profiles.shape, (4, 503))
        self.assertEqual(example.checkpoint_indices.tolist(), [1, 2, 3, 4])
        self.assertEqual(
            example.deviation_slice.stop - example.deviation_slice.start, 501
        )
        self.assertEqual(example.sequence.ndim, 2)
        self.assertEqual(example.raw_profiles[:, 0].tolist(), [1.0, 2.0, 3.0, 4.0])

    def test_drops_open_handles_when_pickled_for_workers(self):
        from chrombpnet.multitask.differential_finetune_data import PairedContextDataset

        dataset = PairedContextDataset([], "not-read.fa", {}, 512, 503)
        dataset._genome = object()
        dataset._nuc_bws = object()

        state = dataset.__getstate__()

        self.assertIsNone(state["_genome"])

        self.assertIsNone(state["_nuc_bws"])


class ContextBalancedSamplerTests(unittest.TestCase):
    def test_samples_contexts_uniformly_and_repeats_for_equal_seeds(self):
        import numpy as np

        from chrombpnet.multitask.differential_finetune_data import (
            ContextBalancedSampler,
        )

        units = [ContextUnit("0", "chr2", 0, "brain", (), (), ())] + [
            ContextUnit(str(index), "chr2", index, "immune", (), (), ())
            for index in range(1, 5)
        ]
        first = list(ContextBalancedSampler(units, num_samples=6_000, seed=7))
        second = list(ContextBalancedSampler(units, num_samples=6_000, seed=7))
        counts = np.bincount(
            [0 if units[index].tissue_context == "brain" else 1 for index in first],
            minlength=2,
        )

        self.assertEqual(first, second)
        self.assertLess(abs(int(counts[0]) - 3_000), 250)
        self.assertLess(abs(int(counts[1]) - 3_000), 250)
