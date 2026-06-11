"""Tests for dapi_validation module."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "workflow"))

from lib.merge.dapi_validation import compute_zncc, extract_dapi_crop


class TestComputeZncc:
    def test_identical_images(self):
        img = np.random.default_rng(42).uniform(0, 1, (32, 32))
        assert abs(compute_zncc(img, img) - 1.0) < 1e-6

    def test_anticorrelated(self):
        img = np.random.default_rng(42).uniform(0.1, 0.9, (32, 32))
        assert abs(compute_zncc(img, 1.0 - img) - (-1.0)) < 1e-6

    def test_uncorrelated_noise(self):
        rng = np.random.default_rng(42)
        img1 = rng.uniform(0, 1, (64, 64))
        img2 = rng.uniform(0, 1, (64, 64))
        corr = compute_zncc(img1, img2)
        assert abs(corr) < 0.15, f"Random images should have low correlation, got {corr}"

    def test_zero_variance_returns_zero(self):
        constant = np.ones((32, 32)) * 5.0
        varied = np.random.default_rng(42).uniform(0, 1, (32, 32))
        assert compute_zncc(constant, varied) == 0.0
        assert compute_zncc(varied, constant) == 0.0
        assert compute_zncc(constant, constant) == 0.0

    def test_intensity_shift_invariant(self):
        img = np.random.default_rng(42).uniform(0, 1, (32, 32))
        shifted = img + 100.0
        assert abs(compute_zncc(img, shifted) - 1.0) < 1e-6

    def test_intensity_scale_invariant(self):
        img = np.random.default_rng(42).uniform(0, 1, (32, 32))
        scaled = img * 3.0
        assert abs(compute_zncc(img, scaled) - 1.0) < 1e-6

    def test_return_type(self):
        img = np.random.default_rng(42).uniform(0, 1, (16, 16))
        result = compute_zncc(img, img)
        assert isinstance(result, float)


class TestExtractDapiCrop:
    def test_center_crop(self):
        img = np.arange(100 * 100, dtype=float).reshape(100, 100)
        crop = extract_dapi_crop(img, 50, 50, crop_size=16)
        assert crop.shape == (16, 16)
        # Center value should be from the image
        assert crop[8, 8] == img[50, 50]

    def test_boundary_crop_top_left(self):
        img = np.ones((100, 100), dtype=float) * 42.0
        crop = extract_dapi_crop(img, 0, 0, crop_size=16)
        assert crop.shape == (16, 16)
        # Top-left quadrant should be zero (padding)
        assert crop[0, 0] == 0.0
        # Bottom-right should have image data
        assert crop[8, 8] == 42.0

    def test_boundary_crop_bottom_right(self):
        img = np.ones((100, 100), dtype=float) * 7.0
        crop = extract_dapi_crop(img, 99, 99, crop_size=16)
        assert crop.shape == (16, 16)
        # Some entries should be zero (padding beyond image)
        assert crop[-1, -1] == 0.0

    def test_crop_size_parameter(self):
        img = np.ones((200, 200))
        for size in [8, 16, 32, 64]:
            crop = extract_dapi_crop(img, 100, 100, crop_size=size)
            assert crop.shape == (size, size)

    def test_float_centroid_rounds(self):
        img = np.arange(50 * 50, dtype=float).reshape(50, 50)
        crop1 = extract_dapi_crop(img, 25.3, 25.7, crop_size=8)
        crop2 = extract_dapi_crop(img, 25, 26, crop_size=8)
        np.testing.assert_array_equal(crop1, crop2)

    def test_fully_outside_returns_zeros(self):
        img = np.ones((50, 50)) * 10.0
        crop = extract_dapi_crop(img, -100, -100, crop_size=8)
        assert crop.shape == (8, 8)
        assert np.all(crop == 0)


class TestValidateMatchesDapiUnit:
    """Unit-level tests that don't require real images on disk."""

    def test_empty_matches_returns_empty(self):
        from lib.merge.dapi_validation import validate_matches_dapi

        empty_df = pd.DataFrame(
            columns=[
                "tile", "site", "cell_0", "cell_1",
                "i_0", "j_0", "i_1", "j_1", "distance",
            ]
        )
        filtered, stats = validate_matches_dapi(
            empty_df,
            root_fp="/nonexistent",
            plate="1",
            well="A1",
            phenotype_dapi_index=0,
            sbs_dapi_index=0,
            sbs_dapi_cycle=1,
        )
        assert len(filtered) == 0
        assert len(stats) == 0
