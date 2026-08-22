"""Local embeddings.

Runs on your own CPU and costs nothing. It orders the day so a listing larger
than screen_n can be cut to screen_n.

CPU by default, deliberately. The work is small -- a few hundred abstracts
through a BERT-base encoder -- and the daily job runs in GitHub Actions, which
has no GPU, so CPU is what production uses anyway. Taking a GPU here only makes
the run compete with whatever else is on the machine, and a run that holds GPU
memory for its whole length is a run something else can reap. Set embed_device
to cuda in config.yaml if the machine's GPUs are yours alone.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


class Embedder:
    """Wraps one sentence-transformers model and always returns unit vectors.

    Normalising at the source means every similarity downstream is a plain dot
    product -- there is no second place where a cosine could be computed wrongly.
    """

    def __init__(self, model_name: str, batch_size: int = 32, device: str = "cpu"):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model = None                      # loaded lazily: import costs ~5s

    def _load(self):
        if self._model is not None:
            return self._model

        from sentence_transformers import SentenceTransformer, models

        def dim_of(module) -> int:
            # sentence-transformers 6 renamed both dimension getters to
            # `get_embedding_dimension`; 3.x-5.x only have the old names.
            for name in ("get_embedding_dimension",
                         "get_word_embedding_dimension",
                         "get_sentence_embedding_dimension"):
                fn = getattr(module, name, None)
                if fn is not None:
                    return fn()
            raise AttributeError(f"cannot determine embedding dimension of {module!r}")

        self._dim_of = dim_of

        if "specter" in self.model_name.lower():
            # SPECTER/SPECTER2 are BERT encoders whose document vector is the CLS
            # token, not a mean of tokens. sentence-transformers defaults to mean
            # pooling for a bare HF checkpoint, which quietly gives worse vectors,
            # so the module stack is built explicitly here.
            word = models.Transformer(self.model_name, max_seq_length=512)
            pool = models.Pooling(dim_of(word), pooling_mode="cls")
            self._model = SentenceTransformer(modules=[word, pool],
                                              device=self.device)
        else:
            self._model = SentenceTransformer(self.model_name, device=self.device)

        log.info("loaded embedding model %s on %s (dim=%d)",
                 self.model_name, self.device, dim_of(self._model))
        return self._model

    @property
    def dim(self) -> int:
        model = self._load()
        return self._dim_of(model)

    def encode(self, texts: list[str]) -> np.ndarray:
        """(n, dim) float32, L2-normalised. Empty input gives an empty array."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        vecs = self._load().encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)
