"""Polynomial RANSAC transform for merge alignment.

Provides a polynomial (degree-2) transform model as an alternative to the
default linear (rotation + translation) model. Polynomial transforms capture
field-dependent optical distortion (barrel/pincushion) that linear models miss.

The PolynomialTransformModel class provides a .predict(X) interface matching
sklearn LinearRegression, so it can be used as a drop-in replacement in
merge_sbs_phenotype() and find_cell_matches().
"""

import json
import warnings

import numpy as np
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from scipy.spatial.distance import cdist

from lib.merge.hash import get_vc, nearest_neighbors


class PolynomialTransformModel:
    """Polynomial coordinate transform with predict() interface.

    Wraps two per-dimension linear models operating on polynomial features.
    For degree=2 and 2D input, features are [1, x, y, x², xy, y²].
    """

    def __init__(self, models, poly, degree):
        self._models = models
        self._poly = poly
        self.degree = degree

    def predict(self, X):
        """Transform points through the polynomial model.

        Args:
            X: (N, 2) array of coordinates.

        Returns:
            (N, 2) array of transformed coordinates.
        """
        X = np.asarray(X, dtype=float)
        X_poly = self._poly.transform(X)
        return np.column_stack([m.predict(X_poly) for m in self._models])

    def to_dict(self):
        """Serialize model parameters for parquet/JSON storage."""
        params = {"degree": self.degree, "n_features": self._poly.n_output_features_}
        for dim, name in enumerate(["i", "j"]):
            params[f"coef_{name}"] = self._models[dim].coef_.tolist()
            params[f"intercept_{name}"] = float(self._models[dim].intercept_)
        return params

    @classmethod
    def from_dict(cls, params):
        """Reconstruct model from serialized parameters."""
        if isinstance(params, str):
            params = json.loads(params)
        degree = params["degree"]
        poly = PolynomialFeatures(degree=degree)
        poly.fit(np.zeros((1, 2)))

        models = []
        for name in ["i", "j"]:
            m = LinearRegression()
            m.coef_ = np.array(params[f"coef_{name}"])
            m.intercept_ = params[f"intercept_{name}"]
            models.append(m)

        return cls(models, poly, degree)


def fit_polynomial_ransac(
    src_points,
    dst_points,
    degree=2,
    residual_threshold=2.0,
    max_trials=1000,
    random_state=42,
):
    """Fit polynomial transform using per-dimension RANSAC.

    Args:
        src_points: (N, 2) matched source points.
        dst_points: (N, 2) matched destination points.
        degree: Polynomial degree (2 = quadratic).
        residual_threshold: Max residual for RANSAC inlier classification.
        max_trials: Max RANSAC iterations.
        random_state: Random seed for reproducibility.

    Returns:
        model: PolynomialTransformModel.
        inlier_mask: (N,) boolean array of RANSAC inliers.
    """
    poly = PolynomialFeatures(degree=degree)
    src_poly = poly.fit_transform(src_points)

    ransac_models = []
    combined_inliers = None

    for dim in range(2):
        ransac = RANSACRegressor(
            residual_threshold=residual_threshold,
            max_trials=max_trials,
            min_samples=poly.n_output_features_,
            random_state=random_state,
        )
        ransac.fit(src_poly, dst_points[:, dim])
        ransac_models.append(ransac)

        if combined_inliers is None:
            combined_inliers = ransac.inlier_mask_.copy()
        else:
            combined_inliers &= ransac.inlier_mask_

    # Refit clean models on inliers only
    inlier_src_poly = src_poly[combined_inliers]
    inlier_dst = dst_points[combined_inliers]

    final_models = []
    for dim in range(2):
        m = LinearRegression()
        m.fit(inlier_src_poly, inlier_dst[:, dim])
        final_models.append(m)

    model = PolynomialTransformModel(final_models, poly, degree)
    return model, combined_inliers


def compute_jacobian_determinant(model, point):
    """Compute the Jacobian determinant of a polynomial transform at a point.

    This serves as a local magnification factor, analogous to the determinant
    of the rotation matrix used for gating in the linear pipeline.

    Args:
        model: PolynomialTransformModel.
        point: (2,) coordinate to evaluate at.

    Returns:
        Determinant of the 2x2 Jacobian matrix at the point.
    """
    point = np.asarray(point, dtype=float)
    eps = 0.5
    base = model.predict(point.reshape(1, -1))[0]

    jacobian = np.zeros((2, 2))
    for col in range(2):
        perturbed = point.copy()
        perturbed[col] += eps
        shifted = model.predict(perturbed.reshape(1, -1))[0]
        jacobian[:, col] = (shifted - base) / eps

    return np.linalg.det(jacobian)


def evaluate_match_polynomial(
    vec_centers_0,
    vec_centers_1,
    threshold_triangle=0.3,
    threshold_point=2,
    degree=2,
    max_trials=1000,
    random_state=42,
):
    """Evaluate alignment quality using polynomial RANSAC on triangle centers.

    Same triangle-hash matching logic as evaluate_match() in hash.py,
    but fits a polynomial transform instead of a linear one.

    Args:
        vec_centers_0: DataFrame with triangle hash vectors/centers for dataset 0.
        vec_centers_1: DataFrame with triangle hash vectors/centers for dataset 1.
        threshold_triangle: Max distance for triangle feature matching.
        threshold_point: Max distance for point correspondence validation.
        degree: Polynomial degree.
        max_trials: RANSAC iterations.
        random_state: Random seed.

    Returns:
        model: PolynomialTransformModel or None if failed.
        score: Alignment quality score (0-1), or -1 if failed.
        determinant: Jacobian determinant at data centroid, or None if failed.
    """
    if vec_centers_0.empty or vec_centers_1.empty:
        return None, -1, None

    V_0, c_0 = get_vc(vec_centers_0)
    V_1, c_1 = get_vc(vec_centers_1)

    if len(V_0) < 10 or len(V_1) < 10:
        return None, -1, None

    i0, i1, distances = nearest_neighbors(V_0, V_1)

    filt = distances < threshold_triangle
    X, Y = c_0[i0[filt]], c_1[i1[filt]]

    if sum(filt) < 10:
        return None, -1, None

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        model, inlier_mask = fit_polynomial_ransac(
            X,
            Y,
            degree=degree,
            residual_threshold=threshold_point,
            max_trials=max_trials,
            random_state=random_state,
        )

    # Score: fraction of c_0 centers that land within threshold_point of a c_1 center
    predicted = model.predict(c_0)
    distances_sq = cdist(predicted, c_1, metric="sqeuclidean")
    threshold_region = 50
    filt_score = np.sqrt(distances_sq.min(axis=0)) < threshold_region
    score = (np.sqrt(distances_sq.min(axis=0))[filt_score] < threshold_point).mean()

    centroid = c_0.mean(axis=0)
    determinant = compute_jacobian_determinant(model, centroid)

    return model, score, determinant
