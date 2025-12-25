"""
FAISS-based in-memory vector database for biometric embeddings,
with MongoDB persistence for user_id + embedding.

Environment:
- MONGODB_URL must be set to a valid MongoDB connection string.
"""

import os
from typing import List, Tuple

import faiss
import numpy as np
import torch
from pymongo import MongoClient
from pymongo.collection import Collection
import certifi
from pymongo.errors import ServerSelectionTimeoutError, PyMongoError

class FaissMongoVectorDB:
    """
    Simple FAISS index backed by MongoDB for persistence.

    - Embeddings are stored in-memory in FAISS for fast search.
    - Metadata (user_id + embedding vector) is stored in MongoDB.
    - On startup, all existing embeddings from MongoDB are loaded into FAISS.

    Metric:
    - "cosine": use inner product on L2-normalized vectors.
    - "l2": use L2 distance.
    """

    def __init__(
        self,
        dim: int = 128,
        metric: str = "cosine",
        mongo_db_name: str = "biometric_db",
        mongo_collection_name: str = "embeddings",
    ) -> None:
        self.dim = dim
        self.metric = metric.lower()

        # --- FAISS index ---
        if self.metric == "cosine":
            # Embeddings are L2-normalized; inner product ~ cosine similarity
            self.index = faiss.IndexFlatIP(dim)
        elif self.metric == "l2":
            self.index = faiss.IndexFlatL2(dim)
        else:
            raise ValueError("metric must be 'cosine' or 'l2'")

        # --- MongoDB client ---
        mongo_url = os.getenv("MONGODB_URL")
        if not mongo_url:
            raise EnvironmentError("MONGODB_URL environment variable is not set")
        # Prefer an explicit TLS connection using certifi's CA bundle to avoid
        # problems with local system CA stores or OpenSSL builds that don't
        # trust the server certificate used by MongoDB Atlas.
        try:
            self.client = MongoClient(
                mongo_url,
                tls=True,
                tlsCAFile=certifi.where(),
                serverSelectionTimeoutMS=30000,
            )
        except TypeError:
            # Older pymongo may not accept tls* kwargs; fall back to default
            # behavior and let pymongo decide. Keep serverSelectionTimeoutMS
            # to avoid long blocking hangs.
            self.client = MongoClient(mongo_url, serverSelectionTimeoutMS=30000)
        self.db = self.client[mongo_db_name]
        self.collection: Collection = self.db[mongo_collection_name]

        # Internal mapping from FAISS index position -> user_id
        self.user_ids: List[str] = []

        # Load existing embeddings from MongoDB into FAISS. If the MongoDB
        # server is unreachable or TLS handshake fails, we catch the error
        # and continue with an empty in-memory index so the demo can still run.
        try:
            self._load_from_mongo()
        except (ServerSelectionTimeoutError, PyMongoError) as e:
            # Avoid importing logging config; print a clear warning instead.
            print(f"Warning: could not load embeddings from MongoDB: {e}\n" "Continuing with an empty FAISS index.")

    def _to_numpy(self, emb: torch.Tensor) -> np.ndarray:
        if emb.ndim == 1:
            emb = emb.unsqueeze(0)
        emb = emb.detach().cpu().float().numpy()
        return emb

    def _load_from_mongo(self) -> None:
        """
        Load all embeddings from MongoDB into FAISS.
        """
        docs = list(self.collection.find({}))
        if not docs:
            return

        user_ids: List[str] = []
        vecs: List[np.ndarray] = []

        for doc in docs:
            user_id = str(doc["user_id"])
            vec_list = doc["embedding"]
            vec = np.asarray(vec_list, dtype="float32")
            if vec.shape != (self.dim,):
                continue
            user_ids.append(user_id)
            vecs.append(vec[None, :])

        if not vecs:
            return

        all_vecs = np.concatenate(vecs, axis=0)
        if self.metric == "cosine":
            faiss.normalize_L2(all_vecs)

        self.index.add(all_vecs)
        self.user_ids.extend(user_ids)

    def add(self, user_id: str, embedding: torch.Tensor) -> None:
        """
        Add a single user embedding to FAISS + MongoDB.

        Args:
            user_id: User identifier.
            embedding: [D] or [1, D] L2-normalized tensor.
        """
        vec = self._to_numpy(embedding)  # [1, D]
        # Ensure correct shape
        if vec.shape[1] != self.dim:
            raise ValueError(f"Embedding dim {vec.shape[1]} does not match index dim {self.dim}")

        if self.metric == "cosine":
            faiss.normalize_L2(vec)

        self.index.add(vec)
        self.user_ids.append(user_id)

        # Persist to MongoDB
        self.collection.update_one(
            {"user_id": user_id},
            {"$set": {"user_id": user_id, "embedding": vec[0].tolist()}},
            upsert=True,
        )

    def search(
        self,
        embedding: torch.Tensor,
        k: int = 1,
    ) -> Tuple[List[str], np.ndarray]:
        """
        Search nearest neighbors.

        Args:
            embedding: Query embedding [D] or [1, D].
            k: Number of neighbors to return.

        Returns:
            (user_ids, distances_or_similarities)
        """
        if self.index.ntotal == 0:
            return [], np.array([])

        vec = self._to_numpy(embedding)  # [1, D]
        if self.metric == "cosine":
            faiss.normalize_L2(vec)

        distances, indices = self.index.search(vec, k)
        idxs = indices[0]
        dists = distances[0]

        result_ids: List[str] = []
        for i in idxs:
            if i == -1 or i >= len(self.user_ids):
                result_ids.append("UNDEFINED")
            else:
                result_ids.append(self.user_ids[i])

        return result_ids, dists

    def recognize(self, embedding: torch.Tensor, threshold: float) -> Tuple[str, float]:
        """
        Open-set recognition with threshold.

        For cosine:
            similarity >= threshold -> match
        For L2:
            distance   <= threshold -> match

        Args:
            embedding: Query embedding [D] or [1, D].
            threshold: Decision threshold.

        Returns:
            (user_id or "UNDEFINED", score)
        """
        if self.index.ntotal == 0:
            # No enrolled users
            return "UNDEFINED", float("-inf") if self.metric == "cosine" else float("inf")

        user_ids, scores = self.search(embedding, k=1)
        best_id = user_ids[0]
        best_score = float(scores[0])

        if self.metric == "cosine":
            is_match = best_score >= threshold
        else:  # L2
            is_match = best_score <= threshold

        if not is_match:
            return "UNDEFINED", best_score

        return best_id, best_score


