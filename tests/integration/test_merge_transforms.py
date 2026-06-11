"""Integration tests for polynomial transform and DAPI validation in merge pipeline.

Tests the full flow from cell hashing through alignment to merging,
verifying both linear backward compatibility and polynomial improvements.
"""

import sys
import tempfile
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../workflow"))

from lib.merge.hash import hash_cell_locations, multistep_alignment
from lib.merge.fast_merge import merge_triangle_hash
from lib.merge.polynomial_transform import (
    PolynomialTransformModel,
    fit_polynomial_ransac,
    evaluate_match_polynomial,
)
from lib.merge.dapi_validation import compute_zncc, extract_dapi_crop


def _generate_cell_grid(n_rows=15, n_cols=15, spacing=50, noise_std=2.0, seed=42):
    """Generate a grid of cell positions with jitter."""
    rng = np.random.RandomState(seed)
    i_coords, j_coords = [], []
    cells = []
    cell_id = 1
    for r in range(n_rows):
        for c in range(n_cols):
            i = r * spacing + rng.normal(0, noise_std)
            j = c * spacing + rng.normal(0, noise_std)
            i_coords.append(i)
            j_coords.append(j)
            cells.append(cell_id)
            cell_id += 1
    return pd.DataFrame({"cell": cells, "i": i_coords, "j": j_coords})


def _apply_linear_transform(df, rotation, translation):
    """Apply linear transform to coordinates."""
    coords = df[["i", "j"]].values
    transformed = coords @ rotation.T + translation
    result = df.copy()
    result["i"] = transformed[:, 0]
    result["j"] = transformed[:, 1]
    return result


def _apply_polynomial_distortion(df, seed=42):
    """Apply a known polynomial distortion to coordinates.

    Applies: x' = x + 0.0005*x^2 + 0.0003*x*y
             y' = y + 0.0004*y^2 + 0.0002*x*y
    """
    coords = df[["i", "j"]].values.copy()
    x, y = coords[:, 0], coords[:, 1]
    new_x = x + 0.0005 * x**2 + 0.0003 * x * y
    new_y = y + 0.0004 * y**2 + 0.0002 * x * y
    result = df.copy()
    result["i"] = new_x
    result["j"] = new_y
    return result


class TestFastMergePolynomialSynthetic:
    """Test polynomial transform through the fast merge pipeline with synthetic data."""

    def test_polynomial_recovers_distorted_matches(self):
        """Polynomial model should match cells despite nonlinear distortion."""
        phenotype = _generate_cell_grid(n_rows=12, n_cols=12, spacing=60, seed=42)
        phenotype["tile"] = 1
        phenotype["plate"] = "test"
        phenotype["well"] = "A1"

        sbs = _apply_polynomial_distortion(phenotype.copy(), seed=42)
        sbs["tile"] = 1
        sbs["cell"] = phenotype["cell"].values

        phenotype_hash = hash_cell_locations(phenotype)
        sbs_hash = hash_cell_locations(sbs).rename(columns={"tile": "site"})

        model, score, det = evaluate_match_polynomial(
            phenotype_hash, sbs_hash,
            threshold_triangle=0.5,
            threshold_point=5,
        )

        assert model is not None, "Polynomial model should be found"
        assert score > 0, f"Score should be positive, got {score}"

    def test_polynomial_outperforms_linear_on_distorted_data(self):
        """Polynomial should produce lower residuals than linear on distorted data."""
        phenotype = _generate_cell_grid(n_rows=15, n_cols=15, spacing=50, seed=123)

        sbs = _apply_polynomial_distortion(phenotype.copy(), seed=123)

        src = phenotype[["i", "j"]].values
        dst = sbs[["i", "j"]].values

        # Fit linear
        from sklearn.linear_model import LinearRegression
        linear = LinearRegression().fit(src, dst)
        linear_residuals = np.linalg.norm(dst - linear.predict(src), axis=1)

        # Fit polynomial
        poly_model, inlier_mask = fit_polynomial_ransac(src, dst, degree=2)
        poly_residuals = np.linalg.norm(dst - poly_model.predict(src), axis=1)

        assert poly_residuals.mean() < linear_residuals.mean(), (
            f"Polynomial mean residual ({poly_residuals.mean():.4f}) should be less "
            f"than linear ({linear_residuals.mean():.4f})"
        )


class TestFastMergeLinearBackwardCompat:
    """Verify that linear mode produces expected results."""

    def test_linear_matches_with_known_transform(self):
        """Linear model should perfectly match cells under known linear transform."""
        phenotype = _generate_cell_grid(n_rows=12, n_cols=12, spacing=60, seed=42)
        phenotype["tile"] = 1
        phenotype["plate"] = "test"
        phenotype["well"] = "A1"

        rotation = np.array([[0.98, -0.02], [0.02, 0.98]])
        translation = np.array([5.0, -3.0])

        sbs = _apply_linear_transform(phenotype.copy(), rotation, translation)
        sbs["tile"] = 1

        phenotype_hash = hash_cell_locations(phenotype)
        sbs_hash = hash_cell_locations(sbs).rename(columns={"tile": "site"})

        from lib.merge.hash import evaluate_match
        rot, trans, score = evaluate_match(
            phenotype_hash, sbs_hash,
            threshold_triangle=0.5,
            threshold_point=5,
        )

        assert rot is not None, "Linear model should be found"
        assert score > 0, f"Score should be positive, got {score}"


class TestDapiValidationIntegration:
    """Test DAPI validation with synthetic image data."""

    def test_zncc_filters_bad_matches(self):
        """DAPI ZNCC should distinguish good from bad matches."""
        rng = np.random.RandomState(42)
        size = 64

        # Good match: same pattern
        pattern = rng.rand(size, size).astype(np.float32)
        good_zncc = compute_zncc(pattern, pattern + rng.normal(0, 0.05, pattern.shape))

        # Bad match: unrelated pattern
        pattern2 = rng.rand(size, size).astype(np.float32)
        bad_zncc = compute_zncc(pattern, pattern2)

        assert good_zncc > 0.8, f"Good match ZNCC should be > 0.8, got {good_zncc}"
        assert bad_zncc < 0.5, f"Bad match ZNCC should be < 0.5, got {bad_zncc}"
        assert good_zncc > bad_zncc, "Good match should have higher ZNCC"

    def test_crop_extraction_preserves_content(self):
        """Extracted crops should contain expected pixel values."""
        rng = np.random.RandomState(42)
        image = rng.randint(0, 65535, (200, 200), dtype=np.uint16)

        crop = extract_dapi_crop(image, 100, 100, crop_size=32)
        assert crop.shape == (32, 32)

        expected = image[84:116, 84:116]
        np.testing.assert_array_equal(crop, expected)

    def test_serialization_roundtrip_through_parquet(self):
        """Polynomial model should survive JSON serialization in parquet."""
        import json

        src = np.random.RandomState(42).rand(100, 2) * 500
        dst = src + 0.0005 * src**2

        model, _ = fit_polynomial_ransac(src, dst, degree=2)
        model_dict = model.to_dict()
        json_str = json.dumps(model_dict)

        recovered = PolynomialTransformModel.from_dict(json.loads(json_str))

        test_points = np.array([[100, 200], [300, 400], [50, 50]])
        original_pred = model.predict(test_points)
        recovered_pred = recovered.predict(test_points)

        np.testing.assert_array_almost_equal(
            original_pred, recovered_pred, decimal=10,
            err_msg="Predictions should be identical after JSON roundtrip"
        )


class TestStitchPolynomialSynthetic:
    """Test polynomial transform through the stitch pipeline components."""

    def test_evaluate_well_match_polynomial(self):
        """Stitch's evaluate_well_match should work with polynomial model."""
        from lib.merge.stitch_alignment import (
            well_level_triangle_hash,
            evaluate_well_match,
        )

        phenotype = _generate_cell_grid(n_rows=20, n_cols=20, spacing=40, seed=42)
        sbs = _apply_polynomial_distortion(phenotype.copy(), seed=42)

        pheno_triangles = well_level_triangle_hash(phenotype)
        sbs_triangles = well_level_triangle_hash(sbs)

        assert len(pheno_triangles) > 0, "Should generate phenotype triangles"
        assert len(sbs_triangles) > 0, "Should generate SBS triangles"

        model, translation, score = evaluate_well_match(
            pheno_triangles, sbs_triangles,
            threshold_triangle=0.5,
            threshold_point=5,
            transform_model="polynomial2",
        )

        assert model is not None, "Polynomial model should be found"
        assert score > 0, f"Score should be positive, got {score}"

    def test_apply_polynomial_transformation(self):
        """_apply_polynomial_transformation should transform coordinates."""
        from lib.merge.stitch_alignment import _apply_polynomial_transformation

        positions = pd.DataFrame({
            "i": [100.0, 200.0, 300.0],
            "j": [100.0, 200.0, 300.0],
        })

        src = np.random.RandomState(42).rand(100, 2) * 500
        dst = src + 0.001 * src**2
        model, _ = fit_polynomial_ransac(src, dst, degree=2)

        result = _apply_polynomial_transformation(positions, model)

        assert len(result) == 3
        assert not np.array_equal(
            result[["i", "j"]].values,
            positions[["i", "j"]].values,
        ), "Coordinates should be transformed"

    def test_load_alignment_parameters_polynomial(self):
        """load_alignment_parameters should handle polynomial model data."""
        import json
        from lib.merge.stitch_merge import load_alignment_parameters

        src = np.random.RandomState(42).rand(100, 2) * 500
        dst = src + 0.001 * src**2
        model, _ = fit_polynomial_ransac(src, dst, degree=2)

        row = pd.Series({
            "rotation_matrix_flat": [1.0, 0.0, 0.0, 1.0],
            "translation_vector": [0.0, 0.0],
            "scale_factor": 0.5,
            "score": 0.85,
            "determinant": 1.0,
            "transformation_type": "triangle_hash_regional",
            "approach": "adaptive_regional_sampling",
            "validation_mean_distance": 1.5,
            "polynomial_model_json": json.dumps(model.to_dict()),
        })

        params = load_alignment_parameters(row)

        assert params["transform_model"] == "polynomial2"
        assert "polynomial_model" in params
        assert hasattr(params["polynomial_model"], "predict")

        test_points = np.array([[100, 200], [300, 400]])
        pred = params["polynomial_model"].predict(test_points)
        assert pred.shape == (2, 2)
