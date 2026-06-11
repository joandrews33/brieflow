"""DAPI intensity correlation validation for merge matches.

After triangle-hash alignment finds candidate cell matches, this module
validates them by comparing DAPI intensity crops around matched centroids.
Matches with low DAPI correlation are likely false positives from
coincidental geometric similarity.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


def compute_zncc(crop1, crop2):
    """Compute zero-mean normalized cross-correlation between two image crops.

    Invariant to linear intensity changes (brightness/contrast).

    Args:
        crop1: (H, W) intensity array.
        crop2: (H, W) intensity array, same shape as crop1.

    Returns:
        Correlation coefficient in [-1, 1]. Returns 0.0 if either
        crop has zero variance (uniform intensity).
    """
    c1 = crop1.ravel().astype(float)
    c2 = crop2.ravel().astype(float)

    c1 = c1 - c1.mean()
    c2 = c2 - c2.mean()

    std1 = np.std(c1)
    std2 = np.std(c2)

    if std1 < 1e-10 or std2 < 1e-10:
        return 0.0

    correlation = np.dot(c1 / std1, c2 / std2) / len(c1)
    return float(np.clip(correlation, -1.0, 1.0))


def extract_dapi_crop(image, centroid_i, centroid_j, crop_size=32):
    """Extract a square crop around a centroid from a 2D image.

    Pads with zeros at image boundaries.

    Args:
        image: (H, W) 2D array.
        centroid_i: Row coordinate (float, will be rounded).
        centroid_j: Column coordinate (float, will be rounded).
        crop_size: Side length of the square crop in pixels.

    Returns:
        (crop_size, crop_size) array.
    """
    ci = int(round(centroid_i))
    cj = int(round(centroid_j))
    half = crop_size // 2
    h, w = image.shape

    # Compute source and destination slices
    src_i_start = max(0, ci - half)
    src_i_end = min(h, ci - half + crop_size)
    src_j_start = max(0, cj - half)
    src_j_end = min(w, cj - half + crop_size)

    dst_i_start = src_i_start - (ci - half)
    dst_j_start = src_j_start - (cj - half)

    crop = np.zeros((crop_size, crop_size), dtype=image.dtype)
    src_h = src_i_end - src_i_start
    src_w = src_j_end - src_j_start
    if src_h > 0 and src_w > 0:
        crop[dst_i_start : dst_i_start + src_h, dst_j_start : dst_j_start + src_w] = (
            image[src_i_start:src_i_end, src_j_start:src_j_end]
        )

    return crop


def _build_phenotype_image_path(root_fp, plate, well, tile):
    """Build path to a preprocessed phenotype image."""
    return str(
        Path(root_fp)
        / "preprocess"
        / "images"
        / "phenotype"
        / f"P-{plate}_W-{well}_T-{tile}__image.tiff"
    )


def _build_sbs_image_path(root_fp, plate, well, tile, cycle):
    """Build path to a preprocessed SBS image."""
    return str(
        Path(root_fp)
        / "preprocess"
        / "images"
        / "sbs"
        / f"P-{plate}_W-{well}_T-{tile}_C-{cycle}__image.tiff"
    )


def _load_dapi_channel(image_path, dapi_index):
    """Load DAPI channel from a multi-channel TIFF.

    Args:
        image_path: Path to TIFF file with shape (C, H, W).
        dapi_index: Channel index for DAPI.

    Returns:
        (H, W) array, or None if file not found.
    """
    import tifffile

    try:
        img = tifffile.imread(image_path)
    except FileNotFoundError:
        return None

    if img.ndim == 3:
        return img[dapi_index]
    return img


class _DapiCache:
    """LRU-style cache for loaded DAPI channel images."""

    def __init__(self, max_size=20):
        self._cache = {}
        self._order = []
        self._max_size = max_size

    def get(self, path, dapi_index):
        if path in self._cache:
            return self._cache[path]
        img = _load_dapi_channel(path, dapi_index)
        if img is not None:
            if len(self._cache) >= self._max_size:
                oldest = self._order.pop(0)
                self._cache.pop(oldest, None)
            self._cache[path] = img
            self._order.append(path)
        return img


def validate_matches_dapi(
    matches_df,
    root_fp,
    plate,
    well,
    phenotype_dapi_index,
    sbs_dapi_index,
    sbs_dapi_cycle,
    min_correlation=0.5,
    crop_size=32,
    max_pairs=200,
):
    """Validate cell matches using DAPI intensity correlation (fast approach).

    For each matched pair: loads DAPI tiles, extracts crops around cell
    centroids, resizes phenotype crop to SBS scale, computes ZNCC.
    Matches below min_correlation are filtered out.

    Args:
        matches_df: DataFrame with columns [tile, site, cell_0, cell_1,
            i_0, j_0, i_1, j_1, distance].
        root_fp: Root path to brieflow_output.
        plate: Plate identifier (str or int).
        well: Well identifier (str).
        phenotype_dapi_index: Channel index for DAPI in phenotype images.
        sbs_dapi_index: Channel index for DAPI in SBS images.
        sbs_dapi_cycle: SBS cycle number for DAPI image.
        min_correlation: Minimum ZNCC threshold to keep a match.
        crop_size: Side length of square crop in SBS pixels.
        max_pairs: Max pairs to validate (samples if more matches exist).

    Returns:
        filtered_df: matches_df filtered to pairs above threshold.
        stats_df: DataFrame with per-pair correlation scores.
    """
    from skimage.transform import resize

    if matches_df.empty:
        return matches_df.copy(), pd.DataFrame(
            columns=["tile", "site", "cell_0", "cell_1", "zncc"]
        )

    # Sample if too many pairs
    if len(matches_df) > max_pairs:
        sample_idx = np.random.default_rng(42).choice(
            len(matches_df), max_pairs, replace=False
        )
        validate_df = matches_df.iloc[sample_idx].copy()
    else:
        validate_df = matches_df.copy()

    ph_cache = _DapiCache(max_size=30)
    sbs_cache = _DapiCache(max_size=30)

    correlations = []
    for _, row in validate_df.iterrows():
        ph_tile = int(row["tile"])
        sbs_tile = int(row["site"])

        ph_path = _build_phenotype_image_path(root_fp, plate, well, ph_tile)
        sbs_path = _build_sbs_image_path(
            root_fp, plate, well, sbs_tile, sbs_dapi_cycle
        )

        ph_dapi = ph_cache.get(ph_path, phenotype_dapi_index)
        sbs_dapi = sbs_cache.get(sbs_path, sbs_dapi_index)

        if ph_dapi is None or sbs_dapi is None:
            correlations.append(np.nan)
            continue

        ph_crop = extract_dapi_crop(ph_dapi, row["i_0"], row["j_0"], crop_size * 2)
        sbs_crop = extract_dapi_crop(sbs_dapi, row["i_1"], row["j_1"], crop_size)

        # Resize phenotype crop to match SBS scale
        ph_crop_resized = resize(
            ph_crop.astype(float),
            (crop_size, crop_size),
            anti_aliasing=True,
            preserve_range=True,
        )

        zncc = compute_zncc(ph_crop_resized, sbs_crop.astype(float))
        correlations.append(zncc)

    stats_df = validate_df[["tile", "site", "cell_0", "cell_1"]].copy()
    stats_df["zncc"] = correlations

    # Determine which indices to keep
    valid_zncc = stats_df["zncc"].fillna(0)

    if len(matches_df) > max_pairs:
        # Only sampled a subset — use mean correlation as a well-level gate
        mean_corr = valid_zncc.mean()
        print(
            f"DAPI validation (sampled {max_pairs}/{len(matches_df)}): "
            f"mean ZNCC = {mean_corr:.3f}"
        )
        if mean_corr < min_correlation:
            print(
                f"  WARNING: mean DAPI correlation {mean_corr:.3f} < "
                f"threshold {min_correlation}. Consider reviewing alignment."
            )
        # Keep all matches but flag the well
        filtered_df = matches_df.copy()
    else:
        # Validated all pairs — filter individually
        keep_mask = valid_zncc >= min_correlation
        n_removed = (~keep_mask).sum()
        print(
            f"DAPI validation: {keep_mask.sum()}/{len(matches_df)} matches passed "
            f"(removed {n_removed}, threshold={min_correlation})"
        )
        filtered_df = matches_df[keep_mask.values].reset_index(drop=True)

    return filtered_df, stats_df


def validate_stitch_matches_dapi(
    matches_df,
    root_fp,
    plate,
    well,
    phenotype_dapi_index,
    sbs_dapi_index,
    sbs_dapi_cycle,
    min_correlation=0.5,
    crop_size=32,
    max_pairs=200,
):
    """Validate cell matches using DAPI correlation (stitch approach).

    Similar to validate_matches_dapi but handles stitch-specific column
    names (stitched_cell_id_0/1) and coordinate systems.

    Args:
        matches_df: DataFrame from stitch merge with columns including
            tile, site, i_0, j_0, i_1, j_1.
        root_fp: Root path to brieflow_output.
        plate, well: Identifiers.
        phenotype_dapi_index: DAPI channel in phenotype images.
        sbs_dapi_index: DAPI channel in SBS images.
        sbs_dapi_cycle: SBS cycle for DAPI.
        min_correlation: Minimum ZNCC threshold.
        crop_size: Crop size in SBS pixels.
        max_pairs: Max pairs to validate.

    Returns:
        filtered_df: Filtered matches.
        stats_df: Per-pair correlation scores.
    """
    cell_0_col = "stitched_cell_id_0" if "stitched_cell_id_0" in matches_df.columns else "cell_0"
    cell_1_col = "stitched_cell_id_1" if "stitched_cell_id_1" in matches_df.columns else "cell_1"

    # Rename to match fast approach interface
    renamed = matches_df.rename(
        columns={cell_0_col: "cell_0", cell_1_col: "cell_1"}
    )

    filtered_df, stats_df = validate_matches_dapi(
        renamed,
        root_fp=root_fp,
        plate=plate,
        well=well,
        phenotype_dapi_index=phenotype_dapi_index,
        sbs_dapi_index=sbs_dapi_index,
        sbs_dapi_cycle=sbs_dapi_cycle,
        min_correlation=min_correlation,
        crop_size=crop_size,
        max_pairs=max_pairs,
    )

    # Rename back
    filtered_df = filtered_df.rename(
        columns={"cell_0": cell_0_col, "cell_1": cell_1_col}
    )

    return filtered_df, stats_df
