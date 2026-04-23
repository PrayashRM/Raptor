# raptor/fcm_clustering.py
"""
Layer 1B: Fuzzy C-Means soft clustering on PCA-reduced embeddings.

Pure numpy implementation — no scikit-fuzzy dependency.
Works on Python 3.12+. Same mathematical result.
"""

from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA

from core.logger import get_logger
from core.exceptions import ClusteringError
import config

logger = get_logger(__name__)


def _fuzzy_cmeans(
    X: np.ndarray,
    k: int,
    m: float = 2.0,
    max_iter: int = 150,
    tol: float = 1e-4,
    random_state: int = 42,
) -> np.ndarray:
    """
    Pure numpy Fuzzy C-Means implementation.

    Args:
        X:            Data matrix (n_samples, n_features)
        k:            Number of clusters
        m:            Fuzziness parameter (default 2.0)
        max_iter:     Maximum iterations
        tol:          Convergence tolerance
        random_state: Random seed for reproducibility

    Returns:
        membership: (n_samples, k) membership matrix
                    membership[i, j] = degree chunk i belongs to cluster j
    """
    rng       = np.random.default_rng(random_state)
    n_samples = X.shape[0]

    # ── Initialize membership matrix randomly ─────────────────────────────
    # Each row sums to 1 (partition of unity)
    U = rng.random((n_samples, k)).astype(np.float64)
    U = U / U.sum(axis=1, keepdims=True)

    centers = np.zeros((k, X.shape[1]), dtype=np.float64)

    for iteration in range(max_iter):
        U_old = U.copy()

        # ── Update cluster centers ─────────────────────────────────────────
        U_m = U ** m  # (n_samples, k)
        for j in range(k):
            weights    = U_m[:, j]          # (n_samples,)
            centers[j] = (weights[:, None] * X).sum(axis=0) / weights.sum()

        # ── Update membership matrix ───────────────────────────────────────
        # Distance from each sample to each center
        # dist[i, j] = ||x_i - c_j||^2
        dist = np.zeros((n_samples, k), dtype=np.float64)
        for j in range(k):
            diff       = X - centers[j]    # (n_samples, n_features)
            dist[:, j] = (diff ** 2).sum(axis=1)

        # Avoid division by zero for samples exactly at a center
        dist = np.maximum(dist, 1e-10)

        # FCM membership update formula:
        # U[i,j] = 1 / sum_l( (d_ij / d_il)^(2/(m-1)) )
        exp   = 2.0 / (m - 1.0)
        new_U = np.zeros_like(U)

        for j in range(k):
            ratio     = dist[:, j:j+1] / dist   # (n_samples, k)
            new_U[:, j] = 1.0 / (ratio ** exp).sum(axis=1)

        # Re-normalize rows to sum to 1 (numerical safety)
        row_sums = new_U.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        U        = new_U / row_sums

        # ── Check convergence ──────────────────────────────────────────────
        delta = np.abs(U - U_old).max()
        if delta < tol:
            logger.debug(
                f"FCM converged at iteration {iteration + 1} "
                f"(delta={delta:.2e})"
            )
            break
    else:
        logger.debug(
            f"FCM reached max_iter={max_iter} without convergence"
        )

    return U  # (n_samples, k)


def run_fcm_clustering(
    chunks: list,
    k: int,
    dense_vectors: list[list[float]],
) -> dict[str, dict[int, float]]:
    """
    Run PCA + Fuzzy C-Means on chunk embeddings.

    Args:
        chunks:        List of Qdrant scroll results
        k:             Number of clusters (from structural clustering)
        dense_vectors: List of 512-dim dense vectors, same order as chunks

    Returns:
        Dict mapping chunk_id -> {cluster_id: membership_score}
        Only includes clusters where membership > threshold.
        Max 2 cluster assignments per chunk.
    """
    n_chunks = len(chunks)

    if n_chunks == 0:
        raise ClusteringError("Cannot run FCM on empty chunk list")

    if k <= 1:
        logger.warning(
            f"k={k} too small for meaningful FCM. "
            f"Returning empty cross-section memberships."
        )
        return {
            chunk.payload["chunk_id"]: {}
            for chunk in chunks
        }

    if k >= n_chunks:
        logger.warning(
            f"k={k} >= n_chunks={n_chunks}. "
            f"Reducing k to {n_chunks - 1}"
        )
        k = max(2, n_chunks - 1)

    logger.info(f"Running FCM: n_chunks={n_chunks}, k={k}")

    # ── Step 1: Build embedding matrix ────────────────────────────────────
    X = np.array(dense_vectors, dtype=np.float64)  # (n_chunks, 512)

    # ── Step 2: PCA dimensionality reduction ──────────────────────────────
    max_pca_dims = max(2, n_chunks // 5)
    pca_dims     = min(50, max_pca_dims, X.shape[1], X.shape[0] - 1)

    logger.info(f"PCA: {X.shape[1]} dims → {pca_dims} dims")

    try:
        pca       = PCA(
            n_components = pca_dims,
            random_state = config.FCM_RANDOM_STATE,
        )
        X_reduced = pca.fit_transform(X)  # (n_chunks, pca_dims)
        explained = pca.explained_variance_ratio_.sum()
        logger.info(
            f"PCA explained variance: {explained:.1%} "
            f"({pca_dims} components)"
        )
    except Exception as e:
        raise ClusteringError(f"PCA failed: {e}") from e

    # ── Step 3: Fuzzy C-Means ─────────────────────────────────────────────
    try:
        membership = _fuzzy_cmeans(
            X            = X_reduced,
            k            = k,
            m            = config.FCM_FUZZINESS,
            max_iter     = config.FCM_MAX_ITER,
            tol          = config.FCM_TOLERANCE,
            random_state = config.FCM_RANDOM_STATE,
        )
        # membership shape: (n_chunks, k)
        logger.info("FCM complete")
    except Exception as e:
        raise ClusteringError(f"Fuzzy C-Means failed: {e}") from e

    # ── Step 4: Apply membership threshold ────────────────────────────────
    threshold  = config.FCM_MEMBERSHIP_THRESHOLD
    max_assign = config.FCM_MAX_CLUSTERS_PER_CHUNK

    chunk_memberships: dict[str, dict[int, float]] = {}

    for chunk_idx, chunk in enumerate(chunks):
        chunk_id = chunk.payload["chunk_id"]

        scores = {
            cluster_id: float(membership[chunk_idx, cluster_id])
            for cluster_id in range(k)
        }

        # Filter by threshold
        above_threshold = {
            cid: score
            for cid, score in scores.items()
            if score >= threshold
        }

        # Cap at max_assign (keep highest scoring)
        if len(above_threshold) > max_assign:
            above_threshold = dict(
                sorted(
                    above_threshold.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:max_assign]
            )

        chunk_memberships[chunk_id] = {
            cid: round(score, 4)
            for cid, score in above_threshold.items()
        }

    multi_cluster = sum(
        1 for m in chunk_memberships.values() if len(m) > 1
    )
    logger.info(
        f"FCM assignments: {multi_cluster}/{n_chunks} chunks "
        f"assigned to multiple clusters"
    )

    return chunk_memberships