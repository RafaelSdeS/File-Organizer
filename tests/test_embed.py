"""Embedder's own logic (pooling, batching order) with a fake tokenizer and
ONNX session. That it matches sentence-transformers on the real model is
what tests/test_quality.py checks."""
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.document_analyzer import embed


class TestEmbedder(unittest.TestCase):
    def setUp(self):
        # A text's tokens are its characters; a token's vector is (code, 1).
        def encode_batch(texts):
            width = max(map(len, texts))
            return [SimpleNamespace(ids=[ord(c) for c in t] + [0] * (width - len(t)),
                                    attention_mask=[1] * len(t) + [0] * (width - len(t))) for t in texts]
        self.tokenizer = MagicMock(encode_batch=MagicMock(side_effect=encode_batch))
        self.session = MagicMock()
        self.session.get_inputs.return_value = [SimpleNamespace(name="input_ids"), SimpleNamespace(name="attention_mask")]
        self.session.run.side_effect = lambda _, feeds: [np.stack([feeds["input_ids"], np.ones_like(feeds["input_ids"])], axis=-1).astype(float)]
        with patch.object(embed, "hf_hub_download", side_effect=lambda repo, name, **kw: name), \
             patch.object(embed.Tokenizer, "from_file", return_value=self.tokenizer), \
             patch.object(embed.onnxruntime, "InferenceSession", return_value=self.session):
            self.embedder = embed.Embedder("model", "rev", local_files_only=True)

    def test_mean_pools_unpadded_tokens_normalizes_and_keeps_input_order(self):
        with patch.object(embed, "BATCH_SIZE", 2):
            vectors = self.embedder.encode(["aa", "cccc", "b"])
        # Means of (code, 1) over each text's own tokens, padding excluded.
        expected = np.array([[ord("a"), 1], [ord("c"), 1], [ord("b"), 1]], dtype=float)
        expected /= np.linalg.norm(expected, axis=1, keepdims=True)
        np.testing.assert_allclose(vectors, expected, rtol=1e-6)
        # Only the inputs the model declares are fed (no token_type_ids here).
        self.assertEqual(set(self.session.run.call_args.args[1]), {"input_ids", "attention_mask"})
        self.tokenizer.enable_truncation.assert_called_once_with(512)


if __name__ == "__main__":
    unittest.main()
