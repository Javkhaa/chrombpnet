from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl
import pyBigWig

from chrombpnet.multitask.differential_finetune_data import (
    ContextBalancedSampler,
    ContextUnit,
    PairedContextDataset,
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

    def test_maps_adults_to_noncontiguous_checkpoint_indices(self):
        metadata = pl.DataFrame(
            {
                "cell_type": ["adult_b", "adult_a"],
                "lineage": ["brain", "brain"],
                "is_adult": [True, True],
                "is_cell_line": [False, False],
            }
        )

        self.assertEqual(
            map_adult_checkpoint_indices(
                metadata,
                ["fetal_first", "adult_a", "fetal_middle", "adult_b", "line"],
            ),
            {"adult_b": 3, "adult_a": 1},
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

    def test_rejects_duplicate_cell_types_within_a_locus_context(self):
        calls = pl.DataFrame(
            {
                "locus_id": ["duplicate"] * 4,
                "chrom": ["chr2"] * 4,
                "center": [1_000] * 4,
                "lineage": ["brain"] * 4,
                "cell_type": ["a", "a", "b", "c"],
                "passes_depth": [True] * 4,
                "is_differential": [True, False, False, True],
            }
        )

        with self.assertRaisesRegex(ValueError, "duplicate.*a"):
            build_context_units(calls, {"a": 1, "b": 2, "c": 3})


class PairedContextDatasetTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.genome = self.root / "genome.fa"
        self.genome.write_text(">chr2\n" + "ACGT" * 1_000 + "\n")
        self.tracks = {}
        for index, cell_type in enumerate(("a", "b", "c", "d"), start=1):
            path = self.root / f"{cell_type}.bw"
            with pyBigWig.open(str(path), "w") as bigwig:
                bigwig.addHeader([("chr2", 4_000)])
                bigwig.addEntries(["chr2"], [0], ends=[4_000], values=[float(index)])
            self.tracks[cell_type] = path
        self.unit = ContextUnit(
            "chr2:2000",
            "chr2",
            2_000,
            "brain",
            ("a", "b", "c", "d"),
            (1, 2, 3, 4),
            (True, False, True, False),
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def _dataset(self):
        return PairedContextDataset([self.unit], self.genome, self.tracks, 512, 503)

    def test_returns_one_sequence_and_raw_profiles_for_a_context_unit(self):
        dataset = self._dataset()
        self.assertIsNone(dataset._genome)
        self.assertIsNone(dataset._nuc_bws)

        example = dataset[0]

        self.assertEqual(example.sequence.shape, (512, 4))
        self.assertEqual(example.raw_profiles.shape, (4, 503))
        self.assertEqual(example.checkpoint_indices.tolist(), [1, 2, 3, 4])
        self.assertEqual(example.deviation_slice.start, 1)
        self.assertEqual(example.deviation_slice.stop, 502)
        self.assertEqual(example.sequence.ndim, 2)
        self.assertEqual(example.raw_profiles[:, 0].tolist(), [1.0, 2.0, 3.0, 4.0])

    def test_pickle_round_trip_reopens_handles_and_reads(self):
        dataset = self._dataset()
        dataset[0]

        restored = pickle.loads(pickle.dumps(dataset))
        self.assertIsNone(restored._genome)
        self.assertIsNone(restored._nuc_bws)

        example = restored[0]

        self.assertEqual(example.raw_profiles[:, 0].tolist(), [1.0, 2.0, 3.0, 4.0])
        self.assertIsNotNone(restored._genome)
        self.assertIsNotNone(restored._nuc_bws)

    def test_close_calls_close_on_all_open_handles(self):
        class Handle:
            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        dataset = self._dataset()
        genome = Handle()
        bigwigs = {"a": Handle(), "b": Handle()}
        dataset._genome = genome
        dataset._nuc_bws = bigwigs

        dataset.close()
        dataset.close()

        self.assertTrue(genome.closed)
        self.assertTrue(all(bigwig.closed for bigwig in bigwigs.values()))
        self.assertIsNone(dataset._genome)
        self.assertIsNone(dataset._nuc_bws)

    def test_close_clears_handles_and_later_access_reopens_them(self):
        dataset = self._dataset()
        dataset[0]
        old_handles = dataset._nuc_bws

        dataset.close()
        dataset.close()

        self.assertIsNone(dataset._genome)
        self.assertIsNone(dataset._nuc_bws)

        example = dataset[0]

        self.assertEqual(example.raw_profiles[:, 0].tolist(), [1.0, 2.0, 3.0, 4.0])
        self.assertIsNot(dataset._nuc_bws, old_handles)
        self.assertIsNotNone(dataset._genome)
        self.assertIsNotNone(dataset._nuc_bws)


class ContextBalancedSamplerTests(unittest.TestCase):
    def test_samples_contexts_uniformly_and_repeats_for_equal_seeds(self):
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
