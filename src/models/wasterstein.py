from __future__ import annotations

from functools import lru_cache
from itertools import permutations
from typing import Optional

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover - scikit-learn normally installs scipy.
    linear_sum_assignment = None


def wasserstein_distance(mu: np.ndarray, nu: np.ndarray, p: int = 1) -> float:
    """
    One-dimensional p-Wasserstein distance between empirical measures.

    The inputs must have the same number of atoms.
    """
    alpha = np.sort(np.asarray(mu, dtype=float).reshape(-1))
    beta = np.sort(np.asarray(nu, dtype=float).reshape(-1))
    if alpha.shape != beta.shape:
        raise ValueError("mu and nu must have the same shape")
    return float(np.mean(np.abs(alpha - beta) ** p) ** (1.0 / p))


def wasserstein_barycenter(cluster: np.ndarray, p: int = 1) -> np.ndarray:
    """
    One-dimensional Wasserstein barycenter for equal-size empirical measures.

    For p=1 the coordinate-wise median is used; for p=2 the mean is used.
    """
    sorted_cluster = np.sort(cluster, axis=1)
    if p == 1:
        return np.median(sorted_cluster, axis=0)
    return np.mean(sorted_cluster, axis=0)


def multivariate_wasserstein_distance(
    mu: np.ndarray,
    nu: np.ndarray,
    p: int = 1,
) -> float:
    """
    Empirical multivariate Wasserstein distance for equal-weight windows.

    This solves the optimal atom matching problem exactly for the window pair.
    """
    left = _as_multivariate_segment(mu)
    right = _as_multivariate_segment(nu)
    if left.shape != right.shape:
        raise ValueError("mu and nu must have the same shape")

    cost = np.linalg.norm(left[:, None, :] - right[None, :, :], axis=2) ** p
    if len(left) <= 8:
        perms = _assignment_permutations(len(left))
        row_idx = np.arange(len(left))[None, :]
        best = np.min(cost[row_idx, perms].sum(axis=1))
        return float((best / len(left)) ** (1.0 / p))

    if linear_sum_assignment is not None:
        rows, cols = linear_sum_assignment(cost)
        return float(np.mean(cost[rows, cols]) ** (1.0 / p))

    raise ImportError("scipy is required for full multivariate Wasserstein distance")


def sliced_wasserstein_distance(
    mu: np.ndarray,
    nu: np.ndarray,
    *,
    directions: np.ndarray,
    p: int = 1,
) -> float:
    """
    Sliced Wasserstein distance for multivariate empirical measures.

    The supplied directions must be unit vectors with shape
    (n_projections, n_features).
    """
    left = _as_multivariate_segment(mu)
    right = _as_multivariate_segment(nu)
    if left.shape != right.shape:
        raise ValueError("mu and nu must have the same shape")

    total = 0.0
    for direction in directions:
        total += wasserstein_distance(left @ direction, right @ direction, p=p)
    return total / len(directions)


class WKMeans:
    """
    One-dimensional Wasserstein K-means for market regime clustering.
    """

    def __init__(
        self,
        k: int = 3,
        p: int = 1,
        max_iter: int = 100,
        tol: float = 1e-6,
        random_state: int = 42,
    ):
        self.k = k
        self.p = p
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.centroids_: list[np.ndarray] = []
        self.labels_: Optional[np.ndarray] = None
        self.losses_: list[float] = []

    def fit(self, segments: np.ndarray) -> "WKMeans":
        segments = np.asarray(segments, dtype=float)
        if segments.ndim != 2:
            raise ValueError("WKMeans expects segments with shape (n_segments, h1)")

        rng = np.random.RandomState(self.random_state)
        init_idx = rng.choice(len(segments), self.k, replace=False)
        self.centroids_ = [np.sort(segments[i]) for i in init_idx]

        for iteration in range(self.max_iter):
            labels = np.array(
                [
                    np.argmin(
                        [wasserstein_distance(seg, c, self.p) for c in self.centroids_]
                    )
                    for seg in segments
                ]
            )

            new_centroids = []
            for state in range(self.k):
                members = segments[labels == state]
                if len(members) == 0:
                    new_centroids.append(self.centroids_[state])
                else:
                    new_centroids.append(wasserstein_barycenter(members, self.p))

            loss = sum(
                wasserstein_distance(self.centroids_[state], new_centroids[state], self.p)
                for state in range(self.k)
            )
            self.losses_.append(loss)
            self.centroids_ = new_centroids
            self.labels_ = labels

            print(f"Iter {iteration + 1:>3} | loss = {loss:.8f}")
            if loss < self.tol:
                print(f"Converged at iteration {iteration + 1}")
                break

        self._sort_clusters()
        return self

    def predict(self, segments: np.ndarray) -> np.ndarray:
        segments = np.asarray(segments, dtype=float)
        return np.array(
            [
                np.argmin(
                    [wasserstein_distance(seg, c, self.p) for c in self.centroids_]
                )
                for seg in segments
            ]
        )

    def _sort_clusters(self) -> None:
        cluster_vars = [np.var(centroid) for centroid in self.centroids_]
        order = np.argsort(cluster_vars)
        self.centroids_ = [self.centroids_[i] for i in order]
        if self.labels_ is not None:
            remap = {old: new for new, old in enumerate(order)}
            self.labels_ = np.array([remap[label] for label in self.labels_])


class MultivariateWKMeans:
    """
    WK-means over multivariate windows using full empirical Wasserstein distance.

    The assignment and loss steps use the full multivariate atom-matching
    distance. The centroid update uses a robust aligned median approximation so
    the method remains practical for walk-forward experiments.
    """

    def __init__(
        self,
        k: int = 3,
        p: int = 1,
        max_iter: int = 100,
        tol: float = 1e-6,
        random_state: int = 42,
    ):
        self.k = k
        self.p = p
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.centroids_: list[np.ndarray] = []
        self.labels_: Optional[np.ndarray] = None
        self.losses_: list[float] = []

    def fit(self, segments: np.ndarray) -> "MultivariateWKMeans":
        segments = _as_segment_collection(segments)
        rng = np.random.RandomState(self.random_state)
        init_idx = rng.choice(len(segments), self.k, replace=False)
        self.centroids_ = [segments[i].copy() for i in init_idx]

        for iteration in range(self.max_iter):
            labels = self._assign(segments)
            new_centroids = self._updated_centroids(segments, labels)

            loss = sum(
                self._distance(self.centroids_[state], new_centroids[state])
                for state in range(self.k)
            )
            self.losses_.append(loss)
            self.centroids_ = new_centroids
            self.labels_ = labels

            print(f"  Iter {iteration + 1:>3} | loss = {loss:.8f}")
            if loss < self.tol:
                print(f"  Converged at iteration {iteration + 1}")
                break

        self._sort_clusters()
        return self

    def predict(self, segments: np.ndarray) -> np.ndarray:
        return self._assign(_as_segment_collection(segments))

    def _distance(self, left: np.ndarray, right: np.ndarray) -> float:
        return multivariate_wasserstein_distance(left, right, p=self.p)

    def _assign(self, segments: np.ndarray) -> np.ndarray:
        return np.array(
            [
                np.argmin(
                    [
                        self._distance(seg, centroid) for centroid in self.centroids_
                    ]
                )
                for seg in segments
            ]
        )

    def _updated_centroids(
        self,
        segments: np.ndarray,
        labels: np.ndarray,
    ) -> list[np.ndarray]:
        centroids = []
        for state in range(self.k):
            members = segments[labels == state]
            if len(members) == 0:
                centroids.append(self.centroids_[state])
            else:
                centroids.append(self._barycenter(members))
        return centroids

    def _barycenter(self, cluster: np.ndarray) -> np.ndarray:
        aligned = np.array([segment[np.argsort(segment[:, 0])] for segment in cluster])
        return np.median(aligned, axis=0)

    def _sort_clusters(self) -> None:
        cluster_vars = [np.var(centroid[:, 0]) for centroid in self.centroids_]
        order = np.argsort(cluster_vars)
        self.centroids_ = [self.centroids_[i] for i in order]
        if self.labels_ is not None:
            remap = {old: new for new, old in enumerate(order)}
            self.labels_ = np.array([remap[label] for label in self.labels_])


class SlicedWKMeans(MultivariateWKMeans):
    """
    WK-means over multivariate windows using sliced Wasserstein distance.
    """

    def __init__(
        self,
        k: int = 3,
        p: int = 1,
        n_projections: int = 50,
        max_iter: int = 100,
        tol: float = 1e-6,
        random_state: int = 42,
    ):
        super().__init__(
            k=k,
            p=p,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
        )
        self.n_projections = n_projections
        self.directions_: Optional[np.ndarray] = None

    def fit(self, segments: np.ndarray) -> "SlicedWKMeans":
        segments = _as_segment_collection(segments)
        self.directions_ = self._make_directions(segments.shape[2])
        return super().fit(segments)

    def _assign(self, segments: np.ndarray) -> np.ndarray:
        if self.directions_ is None:
            self.directions_ = self._make_directions(segments.shape[2])

        return super()._assign(segments)

    def _distance(self, left: np.ndarray, right: np.ndarray) -> float:
        if self.directions_ is None:
            self.directions_ = self._make_directions(left.shape[1])
        return sliced_wasserstein_distance(
            left,
            right,
            directions=self.directions_,
            p=self.p,
        )

    def _barycenter(self, cluster: np.ndarray) -> np.ndarray:
        sorted_cluster = np.sort(cluster, axis=1)
        if self.p == 1:
            return np.median(sorted_cluster, axis=0)
        return np.mean(sorted_cluster, axis=0)

    def _make_directions(self, n_features: int) -> np.ndarray:
        rng = np.random.RandomState(self.random_state)
        directions = rng.randn(self.n_projections, n_features)
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return directions / norms


def _as_multivariate_segment(segment: np.ndarray) -> np.ndarray:
    values = np.asarray(segment, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2:
        raise ValueError("segment must have shape (h1, n_features)")
    if not np.all(np.isfinite(values)):
        raise ValueError("segment contains NaN or infinite values")
    return values


def _as_segment_collection(segments: np.ndarray) -> np.ndarray:
    values = np.asarray(segments, dtype=float)
    if values.ndim == 2:
        values = values[:, :, None]
    if values.ndim != 3:
        raise ValueError("segments must have shape (n_segments, h1, n_features)")
    if not np.all(np.isfinite(values)):
        raise ValueError("segments contains NaN or infinite values")
    return values


@lru_cache(maxsize=None)
def _assignment_permutations(n_atoms: int) -> np.ndarray:
    return np.array(list(permutations(range(n_atoms))), dtype=int)
