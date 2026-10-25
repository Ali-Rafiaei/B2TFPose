import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from joblib import dump, load


def build_or_load_nn(template_embeddings, cache_dir, k, high_level=True):
    """Fits a cosine-NearestNeighbors index; for high_level, caches it to disk
    (indexed at max(10, k) so later calls can request a different k — the
    caller must pass that k explicitly to .kneighbors(), since the cached
    object's own default reflects whatever k built the cache)."""
    if high_level:
        os.makedirs(cache_dir, exist_ok=True)
        idx_path = os.path.join(cache_dir, "nn_index_high_level.joblib")
        if os.path.exists(idx_path):
            nn = load(idx_path)
        else:
            nn = NearestNeighbors(
                n_neighbors=max(10, k), algorithm="auto", metric="cosine"
            )
            nn.fit(template_embeddings)
            dump(nn, idx_path)
    else:
        nn = NearestNeighbors(n_neighbors=k, algorithm="auto", metric="cosine")
        nn.fit(template_embeddings)

    return nn


def faiss_kneighbors(template_embeddings, query_embeddings, k, use_gpu=True):
    """GPU cosine KNN via FAISS IndexFlatIP on L2-normalised vectors.

    Falls back to sklearn NearestNeighbors if FAISS is unavailable.

    Args:
        template_embeddings: (M, D) array-like of template feature vectors
        query_embeddings:    (N, D) array-like of query feature vectors
        k:                   number of nearest neighbours to return
        use_gpu:             if True, move index to GPU when a GPU is available

    Returns:
        distances: (N, k) float32 array of cosine distances (1 - similarity)
        indices:   (N, k) int64 array of neighbour indices into template_embeddings
    """
    try:
        import faiss

        te = np.asarray(template_embeddings, dtype=np.float32)
        qe = np.asarray(query_embeddings, dtype=np.float32)
        faiss.normalize_L2(te)
        faiss.normalize_L2(qe)
        index = faiss.IndexFlatIP(te.shape[1])
        if use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        index.add(te)
        sims, indices = index.search(qe, k)
        distances = (1.0 - sims).astype(np.float32)
        return distances, indices
    except ImportError:
        nn = NearestNeighbors(n_neighbors=k, algorithm="auto", metric="cosine")
        nn.fit(template_embeddings)
        return nn.kneighbors(query_embeddings)
