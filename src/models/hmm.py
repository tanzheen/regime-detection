from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans


@dataclass
class GaussianHMM:
    """Full-covariance Gaussian hidden Markov model for return regimes."""

    n_components: int = 3
    covariance_type: str = "full"
    max_iter: int = 200
    tol: float = 1e-6
    random_state: int = 42
    min_variance: float = 1e-8
    min_probability: float = 1e-12
    verbose: bool = True
    transmat_prior: Optional[np.ndarray] = None

    startprob_: np.ndarray = field(init=False)
    transmat_: np.ndarray = field(init=False)
    means_: np.ndarray = field(init=False)
    covars_: np.ndarray = field(init=False)
    variances_: np.ndarray = field(init=False)
    n_features_: int = field(init=False, default=0)
    n_iter_: int = field(init=False, default=0)
    converged_: bool = field(init=False, default=False)
    log_likelihood_: float = field(init=False, default=-np.inf)
    log_likelihoods_: list[float] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        if self.covariance_type != "full":
            raise ValueError("Only covariance_type='full' is supported")

    def fit(self, observations: np.ndarray) -> "GaussianHMM":
        x = self._validate_observations(observations)
        self.n_features_ = x.shape[1]
        self._initialize_parameters(x)

        previous_log_likelihood = -np.inf
        self.log_likelihoods_ = []
        self.converged_ = False

        for iteration in range(1, self.max_iter + 1):
            gamma, xi_sum, log_likelihood = self._expectation(x)
            self._maximization(x, gamma, xi_sum)

            self.log_likelihoods_.append(log_likelihood)
            improvement = log_likelihood - previous_log_likelihood
            if self.verbose:
                print(
                    f"Iter {iteration:>3} | "
                    f"log likelihood = {log_likelihood:.6f} | "
                    f"improvement = {improvement:.6f}"
                )

            self.n_iter_ = iteration
            self.log_likelihood_ = log_likelihood

            if iteration > 1 and abs(improvement) < self.tol:
                self.converged_ = True
                if self.verbose:
                    print(f"Converged at iteration {iteration}")
                break

            previous_log_likelihood = log_likelihood

        self._sort_states_by_variance()
        return self

    def score(self, observations: np.ndarray) -> float:
        x = self._validate_observations(observations)
        _, _, log_likelihood = self._forward(x)
        return float(log_likelihood)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Decode the most likely hidden state path with Viterbi."""
        x = self._validate_observations(observations)
        log_emissions = self._log_emission_probabilities(x)
        log_start = np.log(np.maximum(self.startprob_, self.min_probability))
        log_trans = np.log(np.maximum(self.transmat_, self.min_probability))

        n_obs = len(x)
        delta = np.zeros((n_obs, self.n_components))
        psi = np.zeros((n_obs, self.n_components), dtype=int)

        delta[0] = log_start + log_emissions[0]
        for t in range(1, n_obs):
            scores = delta[t - 1][:, None] + log_trans
            psi[t] = np.argmax(scores, axis=0)
            delta[t] = np.max(scores, axis=0) + log_emissions[t]

        states = np.zeros(n_obs, dtype=int)
        states[-1] = int(np.argmax(delta[-1]))
        for t in range(n_obs - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        return states

    def filter_proba(
        self,
        observations: np.ndarray,
        initial_state_prob: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Return causal filtered state probabilities for each observation.

        initial_state_prob is the state distribution before the first
        observation in this sequence is incorporated.
        """
        x = self._validate_observations(observations)
        emissions = self._emission_probabilities(x)
        initial = self._normalize(
            self.startprob_ if initial_state_prob is None else initial_state_prob
        )

        alpha = np.zeros((len(x), self.n_components))
        alpha[0] = self._normalize(initial * emissions[0])

        for t in range(1, len(x)):
            prior = alpha[t - 1] @ self.transmat_
            alpha[t] = self._normalize(prior * emissions[t])

        return alpha

    def filter_states(
        self,
        observations: np.ndarray,
        initial_state_prob: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return np.argmax(
            self.filter_proba(observations, initial_state_prob=initial_state_prob),
            axis=1,
        )

    def _initialize_parameters(self, x: np.ndarray) -> None:
        kmeans = KMeans(
            n_clusters=self.n_components,
            n_init=10,
            random_state=self.random_state,
        )
        labels = kmeans.fit_predict(x)

        global_cov = self._empirical_covariance(x)
        self.means_ = np.zeros((self.n_components, self.n_features_), dtype=float)
        self.covars_ = np.zeros(
            (self.n_components, self.n_features_, self.n_features_),
            dtype=float,
        )

        for state in range(self.n_components):
            members = x[labels == state]
            if len(members) == 0:
                self.means_[state] = np.mean(x, axis=0)
                self.covars_[state] = global_cov
            else:
                self.means_[state] = np.mean(members, axis=0)
                self.covars_[state] = self._empirical_covariance(members)

        self._update_variances()

        start_counts = np.full(self.n_components, 1.0)
        start_counts[labels[0]] += 1.0
        self.startprob_ = self._normalize(start_counts)

        trans_counts = np.full((self.n_components, self.n_components), 1.0)
        for current_state, next_state in zip(labels[:-1], labels[1:]):
            trans_counts[current_state, next_state] += 1.0
        self.transmat_ = self._normalize_rows(trans_counts)

    def _expectation(
        self,
        x: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        alpha, scale, log_likelihood = self._forward(x)
        beta = self._backward(x, scale)

        gamma = alpha * beta
        gamma = self._normalize_rows(gamma)

        emissions = self._emission_probabilities(x)
        xi_sum = np.zeros((self.n_components, self.n_components))
        for t in range(len(x) - 1):
            xi = (
                alpha[t, :, None]
                * self.transmat_
                * emissions[t + 1, None, :]
                * beta[t + 1, None, :]
            )
            denominator = xi.sum()
            if denominator <= 0:
                denominator = self.min_probability
            xi_sum += xi / denominator

        return gamma, xi_sum, log_likelihood

    def _maximization(
        self,
        x: np.ndarray,
        gamma: np.ndarray,
        xi_sum: np.ndarray,
    ) -> None:
        self.startprob_ = self._normalize(gamma[0] + self.min_probability)
        self.transmat_ = self._normalize_rows(
            xi_sum + self._validated_transmat_prior() + self.min_probability
        )

        weights = np.maximum(gamma.sum(axis=0), self.min_probability)
        self.means_ = (gamma.T @ x) / weights[:, None]

        for state in range(self.n_components):
            centered = x - self.means_[state]
            weighted = centered * gamma[:, state][:, None]
            covar = weighted.T @ centered / weights[state]
            self.covars_[state] = self._regularize_covar(covar)

        self._update_variances()

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        emissions = self._emission_probabilities(x)
        alpha = np.zeros((len(x), self.n_components))
        scale = np.zeros(len(x))

        alpha[0] = self.startprob_ * emissions[0]
        scale[0] = max(alpha[0].sum(), self.min_probability)
        alpha[0] /= scale[0]

        for t in range(1, len(x)):
            alpha[t] = (alpha[t - 1] @ self.transmat_) * emissions[t]
            scale[t] = max(alpha[t].sum(), self.min_probability)
            alpha[t] /= scale[t]

        log_likelihood = float(np.sum(np.log(scale)))
        return alpha, scale, log_likelihood

    def _backward(self, x: np.ndarray, scale: np.ndarray) -> np.ndarray:
        emissions = self._emission_probabilities(x)
        beta = np.zeros((len(x), self.n_components))
        beta[-1] = 1.0

        for t in range(len(x) - 2, -1, -1):
            beta[t] = self.transmat_ @ (emissions[t + 1] * beta[t + 1])
            beta[t] /= max(scale[t + 1], self.min_probability)

        return beta

    def _sort_states_by_variance(self) -> None:
        order = np.argsort(self.variances_)
        self.startprob_ = self.startprob_[order]
        self.transmat_ = self.transmat_[np.ix_(order, order)]
        self.means_ = self.means_[order]
        self.covars_ = self.covars_[order]
        self.variances_ = self.variances_[order]

    def _log_emission_probabilities(self, x: np.ndarray) -> np.ndarray:
        log_probs = np.zeros((len(x), self.n_components), dtype=float)
        for state in range(self.n_components):
            covar = self._regularize_covar(self.covars_[state])
            sign, log_det = np.linalg.slogdet(covar)
            if sign <= 0:
                covar = self._regularize_covar(covar + np.eye(self.n_features_) * self.min_variance)
                _, log_det = np.linalg.slogdet(covar)

            centered = x - self.means_[state]
            try:
                solved = np.linalg.solve(covar, centered.T).T
            except np.linalg.LinAlgError:
                solved = centered @ np.linalg.pinv(covar)
            mahalanobis = np.sum(centered * solved, axis=1)
            log_probs[:, state] = -0.5 * (
                self.n_features_ * np.log(2.0 * np.pi) + log_det + mahalanobis
            )
        return log_probs

    def _emission_probabilities(self, x: np.ndarray) -> np.ndarray:
        log_probs = self._log_emission_probabilities(x)
        return np.exp(np.clip(log_probs, -745.0, 50.0))

    def _validate_observations(self, observations: np.ndarray) -> np.ndarray:
        x = np.asarray(observations, dtype=float)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.ndim != 2:
            raise ValueError("observations must be a 1D or 2D array")
        if len(x) < self.n_components:
            raise ValueError("observations must contain at least n_components rows")
        if not np.all(np.isfinite(x)):
            raise ValueError("observations contains NaN or infinite values")
        if self.n_features_ and x.shape[1] != self.n_features_:
            raise ValueError(
                f"observations has {x.shape[1]} features, expected {self.n_features_}"
            )
        return x

    def _empirical_covariance(self, x: np.ndarray) -> np.ndarray:
        if len(x) <= 1:
            covar = np.eye(self.n_features_) * self.min_variance
        else:
            covar = np.cov(x, rowvar=False, bias=True)
        covar = np.asarray(covar, dtype=float)
        if covar.ndim == 0:
            covar = covar.reshape(1, 1)
        return self._regularize_covar(covar)

    def _regularize_covar(self, covar: np.ndarray) -> np.ndarray:
        covar = np.asarray(covar, dtype=float)
        covar = np.atleast_2d(covar)
        covar = (covar + covar.T) / 2.0
        covar = covar.copy()
        covar.flat[:: covar.shape[0] + 1] += self.min_variance
        return covar

    def _validated_transmat_prior(self) -> np.ndarray:
        if self.transmat_prior is None:
            return np.zeros((self.n_components, self.n_components))

        prior = np.asarray(self.transmat_prior, dtype=float)
        expected_shape = (self.n_components, self.n_components)
        if prior.shape != expected_shape:
            raise ValueError(
                f"transmat_prior shape {prior.shape} does not match {expected_shape}"
            )
        return np.maximum(prior, 0.0)

    def _update_variances(self) -> None:
        self.variances_ = np.array(
            [self.covars_[state, 0, 0] for state in range(self.n_components)],
            dtype=float,
        )
        self.variances_ = np.maximum(self.variances_, self.min_variance)

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        values = np.maximum(values, self.min_probability)
        total = values.sum()
        if total <= 0:
            return np.full_like(values, 1.0 / len(values))
        return values / total

    def _normalize_rows(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        values = np.maximum(values, self.min_probability)
        row_sums = values.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, self.min_probability)
        return values / row_sums
