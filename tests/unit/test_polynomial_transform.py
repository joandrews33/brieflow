"""Tests for polynomial_transform module."""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import Delaunay

# Add workflow directory to path so lib imports work
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "workflow"))

from lib.merge.polynomial_transform import (
    PolynomialTransformModel,
    compute_jacobian_determinant,
    evaluate_match_polynomial,
    fit_polynomial_ransac,
)
from lib.merge.hash import find_triangles


def _make_grid_points(n=20, spacing=10.0):
    """Create a regular grid of 2D points."""
    xs = np.arange(n) * spacing
    ys = np.arange(n) * spacing
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack([xx.ravel(), yy.ravel()])


def _apply_known_polynomial(pts, a=0.001, b=0.0005):
    """Apply a known polynomial distortion: x' = x + a*x^2 + b*x*y."""
    x, y = pts[:, 0], pts[:, 1]
    x_new = x + a * x**2 + b * x * y
    y_new = y + b * y**2 + a * x * y
    return np.column_stack([x_new, y_new])


class TestFitPolynomialRansac:
    def test_recovers_known_polynomial(self):
        src = _make_grid_points(15, spacing=5.0)
        dst = _apply_known_polynomial(src)
        model, inliers = fit_polynomial_ransac(src, dst, degree=2)
        predicted = model.predict(src)
        residuals = np.linalg.norm(predicted - dst, axis=1)
        assert np.mean(residuals) < 0.1, f"Mean residual {np.mean(residuals):.4f} too high"
        assert np.all(inliers), "All points should be inliers for clean data"

    def test_rejects_outliers(self):
        rng = np.random.default_rng(123)
        src = _make_grid_points(15, spacing=5.0)
        dst = _apply_known_polynomial(src)
        # Corrupt 20% of destination points
        n_corrupt = len(src) // 5
        corrupt_idx = rng.choice(len(src), n_corrupt, replace=False)
        dst[corrupt_idx] += rng.normal(0, 50, (n_corrupt, 2))

        model, inliers = fit_polynomial_ransac(
            src, dst, degree=2, residual_threshold=2.0
        )
        # Most corrupted points should be outliers
        outlier_rate = (~inliers[corrupt_idx]).mean()
        assert outlier_rate > 0.5, f"Only {outlier_rate:.0%} of corrupted points were rejected"

    def test_degree_1_matches_linear(self):
        rng = np.random.default_rng(42)
        src = rng.uniform(0, 100, (50, 2))
        # Pure linear transform: rotation + translation
        angle = np.radians(5)
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        t = np.array([10.0, -5.0])
        dst = src @ R.T + t

        model, inliers = fit_polynomial_ransac(src, dst, degree=1)
        predicted = model.predict(src)
        residuals = np.linalg.norm(predicted - dst, axis=1)
        assert np.mean(residuals) < 0.01

    def test_returns_correct_types(self):
        src = _make_grid_points(10, spacing=5.0)
        dst = _apply_known_polynomial(src)
        model, inliers = fit_polynomial_ransac(src, dst)
        assert isinstance(model, PolynomialTransformModel)
        assert isinstance(inliers, np.ndarray)
        assert inliers.dtype == bool
        assert len(inliers) == len(src)


class TestPolynomialTransformModel:
    def test_predict_shape(self):
        src = _make_grid_points(10, spacing=5.0)
        dst = _apply_known_polynomial(src)
        model, _ = fit_polynomial_ransac(src, dst)
        result = model.predict(src)
        assert result.shape == src.shape

    def test_serialization_roundtrip(self):
        src = _make_grid_points(10, spacing=5.0)
        dst = _apply_known_polynomial(src)
        model, _ = fit_polynomial_ransac(src, dst)

        params = model.to_dict()
        restored = PolynomialTransformModel.from_dict(params)

        original_pred = model.predict(src)
        restored_pred = restored.predict(src)
        np.testing.assert_allclose(original_pred, restored_pred, atol=1e-10)

    def test_serialization_from_json_string(self):
        import json

        src = _make_grid_points(10, spacing=5.0)
        dst = _apply_known_polynomial(src)
        model, _ = fit_polynomial_ransac(src, dst)

        json_str = json.dumps(model.to_dict())
        restored = PolynomialTransformModel.from_dict(json_str)
        np.testing.assert_allclose(
            model.predict(src), restored.predict(src), atol=1e-10
        )

    def test_to_dict_contents(self):
        src = _make_grid_points(10, spacing=5.0)
        dst = _apply_known_polynomial(src)
        model, _ = fit_polynomial_ransac(src, dst)
        d = model.to_dict()
        assert "degree" in d
        assert "coef_i" in d
        assert "coef_j" in d
        assert "intercept_i" in d
        assert "intercept_j" in d
        assert d["degree"] == 2
        # degree=2 with 2 inputs: [1, x, y, x², xy, y²] = 6 features
        assert len(d["coef_i"]) == 6


class TestJacobianDeterminant:
    def test_identity_like_transform(self):
        src = _make_grid_points(10, spacing=5.0)
        # Near-identity polynomial (small distortion)
        dst = src + 0.0001 * src**2
        model, _ = fit_polynomial_ransac(src, dst)
        det = compute_jacobian_determinant(model, np.array([25.0, 25.0]))
        # Should be close to 1 near center
        assert 0.9 < det < 1.2, f"Jacobian det {det} not near 1.0"

    def test_scaled_transform(self):
        src = _make_grid_points(10, spacing=5.0)
        dst = src * 2.0  # 2x scaling
        model, _ = fit_polynomial_ransac(src, dst, degree=1)
        det = compute_jacobian_determinant(model, np.array([25.0, 25.0]))
        assert abs(det - 4.0) < 0.1, f"Expected det≈4 for 2x scaling, got {det}"


class TestEvaluateMatchPolynomial:
    def _make_triangle_hash_df(self, points):
        """Generate triangle hash DataFrame from points."""
        import pandas as pd

        points_df = pd.DataFrame(points, columns=["i", "j"])
        points_df["well"] = "A1"
        points_df["tile"] = 1
        result = find_triangles(points_df)
        result["tile"] = 1
        return result

    def test_matching_point_clouds(self):
        rng = np.random.default_rng(42)
        src_pts = rng.uniform(0, 100, (100, 2))
        dst_pts = _apply_known_polynomial(src_pts, a=0.0005, b=0.0003)

        src_hash = self._make_triangle_hash_df(src_pts)
        dst_hash = self._make_triangle_hash_df(dst_pts)

        if len(src_hash) < 10 or len(dst_hash) < 10:
            pytest.skip("Not enough triangles generated for test")

        model, score, det = evaluate_match_polynomial(src_hash, dst_hash)
        assert model is not None, "Model should not be None for matching clouds"
        assert score > 0, f"Score should be positive, got {score}"

    def test_insufficient_triangles_returns_none(self):
        import pandas as pd

        # Create empty DataFrames mimicking triangle hash output with no matches
        cols_v = [f"V_{i}" for i in range(18)]
        cols_c = ["c_0", "c_1"]
        empty = pd.DataFrame(columns=cols_v + cols_c + ["magnitude", "tile"])

        model, score, det = evaluate_match_polynomial(empty, empty)
        assert model is None
        assert score == -1
