from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np


CODE = pathlib.Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(CODE))

from caer import (  # noqa: E402
    CONTEXT_FEATURE_NAMES,
    boundary_feature_names,
    build_boundary_features,
    combine_boundary_scores,
    extract_boundary_window,
    offline_keep_labels,
    rerank_boundary,
    selected_keep_labels,
)


class BoundaryExtractionTests(unittest.TestCase):
    def test_w2_returns_two_candidates_on_each_side(self) -> None:
        order = np.array([4, 2, 0, 5, 3, 1])
        got = extract_boundary_window(order, keep_count=3, radius=2)
        np.testing.assert_array_equal(got, [2, 0, 5, 3])

    def test_incomplete_symmetric_window_is_ineligible(self) -> None:
        self.assertIsNone(extract_boundary_window(np.arange(5), keep_count=1, radius=2))
        self.assertIsNone(extract_boundary_window(np.arange(5), keep_count=4, radius=2))

    def test_duplicate_order_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            extract_boundary_window(np.array([0, 1, 1, 2]), keep_count=2, radius=2)


class OfflineLabelTests(unittest.TestCase):
    def test_future_label_keeps_exactly_two_nearest_reuses(self) -> None:
        labels = offline_keep_labels(
            np.array([10, 11, 12, 13]),
            np.array([9, 2, 7, 1]),
            np.array([4, 3, 2, 1]),
        )
        np.testing.assert_array_equal(labels, [0, 1, 0, 1])

    def test_capped_ties_follow_recency_then_expert_id(self) -> None:
        labels = offline_keep_labels(
            np.array([8, 3, 6, 2]),
            np.array([100, 100, 100, 100]),
            np.array([4, 9, 9, 9]),
            keep_count=2,
            distance_cap=33,
        )
        # Experts 2, 3, and 6 tie on recency; ids 2 and 3 win.
        np.testing.assert_array_equal(labels, [0, 1, 0, 1])


class DeployableFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.block = np.vstack((np.arange(8), np.arange(8) * 10.0))
        self.scores = np.array([0.4, 0.1, 0.7, 0.2, 0.9, 0.6, 0.3, 0.8])
        self.candidates = np.array([0, 1, 2, 3, 4, 5])
        self.order = np.array([4, 2, 5, 0, 3, 1])

    def test_features_use_only_causal_block_and_global_context(self) -> None:
        result = build_boundary_features(
            self.block,
            self.scores,
            self.candidates,
            self.order,
            keep_count=3,
            capacity=24,
            radius=2,
            num_experts=8,
        )
        self.assertIsNotNone(result)
        rows, window = result
        np.testing.assert_array_equal(window, [2, 5, 0, 3])
        self.assertEqual(rows.shape, (4, 2 + len(CONTEXT_FEATURE_NAMES)))
        np.testing.assert_array_equal(rows[:, :2], self.block[:, [2, 5, 0, 3]].T)
        self.assertEqual(
            len(boundary_feature_names(("f0", "f1"))),
            rows.shape[1],
        )

    def test_future_labels_cannot_change_deployable_features(self) -> None:
        before, _ = build_boundary_features(
            self.block, self.scores, self.candidates, self.order, 3, 24, num_experts=8
        )
        offline_keep_labels(
            np.array([2, 5, 0, 3]),
            np.array([1, 100, 2, 99]),
            np.zeros(4, dtype=np.int64),
        )
        after, _ = build_boundary_features(
            self.block, self.scores, self.candidates, self.order, 3, 24, num_experts=8
        )
        np.testing.assert_array_equal(before, after)


class WindowRadiusTests(unittest.TestCase):
    """Boundary windows, labels, features and reranking for radii other than 2."""

    def test_window_width_is_two_times_radius(self) -> None:
        order = np.arange(20)
        for radius in (1, 2, 4, 8):
            window = extract_boundary_window(order, keep_count=10, radius=radius)
            self.assertEqual(window.size, 2 * radius)
            np.testing.assert_array_equal(window, order[10 - radius : 10 + radius])

    def test_wide_radius_is_ineligible_near_the_ends(self) -> None:
        order = np.arange(20)
        self.assertIsNone(extract_boundary_window(order, keep_count=3, radius=4))
        self.assertIsNone(extract_boundary_window(order, keep_count=17, radius=4))
        self.assertIsNotNone(extract_boundary_window(order, keep_count=4, radius=4))

    def test_labels_keep_radius_candidates(self) -> None:
        for radius in (1, 2, 4):
            experts = np.arange(2 * radius)
            distances = np.arange(2 * radius)[::-1].astype(np.int64)
            labels = offline_keep_labels(
                experts,
                distances,
                np.zeros(2 * radius, dtype=np.int64),
                keep_count=radius,
            )
            self.assertEqual(int(labels.sum()), radius)
            np.testing.assert_array_equal(labels[-radius:], np.ones(radius))

    def test_feature_rows_scale_with_radius(self) -> None:
        block = np.vstack((np.arange(16.0), np.arange(16.0) * 2.0))
        scores = np.linspace(1.0, 0.0, 16)
        candidates = np.arange(12)
        order = np.arange(12)
        for radius in (1, 2, 4):
            rows, window = build_boundary_features(
                block,
                scores,
                candidates,
                order,
                keep_count=6,
                capacity=24,
                radius=radius,
                num_experts=16,
            )
            self.assertEqual(rows.shape[0], 2 * radius)
            self.assertEqual(window.size, 2 * radius)
            self.assertEqual(rows.shape[1], 2 + len(CONTEXT_FEATURE_NAMES))

    def test_reranking_preserves_positions_outside_a_wide_window(self) -> None:
        order = np.arange(20)
        window = order[6:14].copy()
        scores = np.linspace(0.0, 1.0, 8)
        got = rerank_boundary(order, window, scores)
        np.testing.assert_array_equal(got[:6], order[:6])
        np.testing.assert_array_equal(got[14:], order[14:])
        np.testing.assert_array_equal(got[6:14], window[::-1])


class BoundaryRerankingTests(unittest.TestCase):
    def test_replacement_changes_only_the_window(self) -> None:
        order = np.array([7, 6, 5, 4, 3, 2, 1, 0])
        window = order[2:6].copy()
        got = rerank_boundary(order, window, np.array([0.1, 0.9, 0.8, 0.2]))
        np.testing.assert_array_equal(got, [7, 6, 4, 3, 2, 5, 1, 0])
        np.testing.assert_array_equal(got[:2], order[:2])
        np.testing.assert_array_equal(got[6:], order[6:])

    def test_equal_boundary_scores_preserve_base_window_order(self) -> None:
        order = np.arange(8)
        window = order[2:6]
        np.testing.assert_array_equal(rerank_boundary(order, window, np.ones(4)), order)

    def test_selected_labels_follow_final_cutoff(self) -> None:
        order = np.array([0, 1, 3, 2, 4, 5])
        window = np.array([1, 2, 3, 4])
        np.testing.assert_array_equal(selected_keep_labels(window, order, 3), [1, 0, 1, 0])

    def test_residual_and_replacement_score_modes(self) -> None:
        base = np.array([0.2, 0.4])
        correction = np.array([0.5, -0.5])
        np.testing.assert_array_equal(
            combine_boundary_scores(base, correction, mode="replacement"), correction
        )
        np.testing.assert_allclose(
            combine_boundary_scores(base, correction, mode="residual", strength=0.25),
            [0.325, 0.275],
        )


if __name__ == "__main__":
    unittest.main()
