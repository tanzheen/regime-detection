from typing import Optional

import numpy as np

# =============================================================================
# STEP 2 — Wasserstein distance (Proposition 2.5, Equation 21)
# =============================================================================

def wasserstein_distance(mu: np.ndarray, nu: np.ndarray, p: int = 1) -> float:
    """
    p-Wasserstein distance between two empirical measures with equal atoms.

    W_p(mu, nu)^p = (1/N) * sum |alpha_i - beta_i|^p
    where alpha, beta are sorted atoms of mu, nu.

    O(N log N) due to sorting.
    """
    alpha = np.sort(mu)
    beta  = np.sort(nu)
    return float(np.mean(np.abs(alpha - beta) ** p) ** (1.0 / p))


# =============================================================================
# STEP 3 — Wasserstein barycenter (Proposition 2.6)
# =============================================================================

def wasserstein_barycenter(cluster: np.ndarray, p: int = 1) -> np.ndarray:
    """
    Wasserstein barycenter of a cluster of empirical measures.

    For p=1: barycenter atoms = coordinate-wise median  (Proposition 2.6)
    For p=2: barycenter atoms = coordinate-wise mean    (Remark C.2)

    Args:
        cluster: (M_cluster, h1) array — all windows assigned to this cluster
        p:       Wasserstein order

    Returns:
        (h1,) barycenter measure (sorted atoms)
    """
    sorted_cluster = np.sort(cluster, axis=1)
    if p == 1:
        return np.median(sorted_cluster, axis=0)
    else:
        return np.mean(sorted_cluster, axis=0)


# =============================================================================
# STEP 4 — WK-Means (Algorithm 1, Definition 2.7)
# =============================================================================

class WKMeans:
    """
    Wasserstein K-Means algorithm for market regime clustering.

    Reference: Horvath, Issa, Muguruza (2021)
    'Clustering Market Regimes Using the Wasserstein Distance'
    https://ssrn.com/abstract=3947905
    """

    def __init__(
        self,
        k: int = 2,
        p: int = 1,
        max_iter: int = 100,
        tol: float = 1e-6,
        random_state: int = 42,
    ):
        """
        Args:
            k:            Number of clusters (k=2 → bull/bear)
            p:            Wasserstein order (p=1 recommended)
            max_iter:     Maximum iterations
            tol:          Convergence tolerance (loss function threshold)
            random_state: Seed for centroid initialisation
        """
        self.k            = k
        self.p            = p
        self.max_iter     = max_iter
        self.tol          = tol
        self.random_state = random_state

        # Fitted attributes
        self.centroids_: list[np.ndarray] = []
        self.labels_: Optional[np.ndarray] = None
        self.losses_: list[float] = []

    def fit(self, segments: np.ndarray) -> "WKMeans":
        """
        Fit WK-means on an (M, h1) array of return windows.

        Args:
            segments: Output of stream_lift()

        Returns:
            self
        """
        rng = np.random.RandomState(self.random_state)
        M   = len(segments)

        # --- Initialise centroids by random sampling from segments ---
        init_idx       = rng.choice(M, self.k, replace=False)
        self.centroids_ = [np.sort(segments[i]) for i in init_idx]

        for iteration in range(self.max_iter):

            # --- Assign each segment to nearest centroid (Equation 7) ---
            labels = np.array([
                np.argmin([wasserstein_distance(seg, c, self.p) for c in self.centroids_])
                for seg in segments
            ])

            # --- Update centroids via Wasserstein barycenter ---
            new_centroids = []
            for l in range(self.k):
                members = segments[labels == l]
                if len(members) == 0:
                    new_centroids.append(self.centroids_[l])  # keep old if empty
                else:
                    new_centroids.append(wasserstein_barycenter(members, self.p))

            # --- Loss function (Equation 23) ---
            loss = sum(
                wasserstein_distance(self.centroids_[l], new_centroids[l], self.p)
                for l in range(self.k)
            )
            self.losses_.append(loss)
            self.centroids_ = new_centroids
            self.labels_    = labels

            print(f"Iter {iteration + 1:>3} | loss = {loss:.8f}")

            if loss < self.tol:
                print(f"Converged at iteration {iteration + 1}")
                break

        # --- Sort clusters: cluster 0 = low variance (bull), 1 = high variance (bear) ---
        cluster_vars = [np.var(self.centroids_[l]) for l in range(self.k)]
        order = np.argsort(cluster_vars)
        self.centroids_ = [self.centroids_[i] for i in order]
        remap = {old: new for new, old in enumerate(order)}
        self.labels_ = np.array([remap[l] for l in self.labels_])

        return self

    def predict(self, segments: np.ndarray) -> np.ndarray:
        """Assign new segments to nearest centroid."""
        return np.array([
            np.argmin([wasserstein_distance(seg, c, self.p) for c in self.centroids_])
            for seg in segments
        ])
