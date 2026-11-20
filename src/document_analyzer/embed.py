"""Sentence embeddings on onnxruntime, without torch.

The same intfloat/multilingual-e5-small weights (its repo ships
onnx/model.onnx), tokenizer, 512-token truncation and mean pooling as
sentence-transformers. Measured on examples/demo (68 texts x 5, laptop CPU):
cosine 1.00000 to the sentence-transformers vectors for every document, the
same folders, the same speed (39 s vs 39-44 s), a lower memory peak (1.4 GB
vs 1.6 GB), and ~130 MB of dependencies instead of ~900 MB (torch's CPU
build; the default PyPI build with CUDA is several GB).
ponytail: CPU only; onnxruntime-gpu + CUDAExecutionProvider if a GPU matters.
"""
import numpy as np
import onnxruntime
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from tqdm import tqdm  # a huggingface_hub dependency

MAX_TOKENS = 512
# Peak memory grows with the batch (attention over 512 tokens): 32 was 15%
# faster than 8 but peaked at 2.9 GB instead of 1.4 GB; 16 gained nothing.
BATCH_SIZE = 8


class Embedder:
    def __init__(self, model, revision, local_files_only=False):
        def fetch(name):
            return hf_hub_download(model, name, revision=revision, local_files_only=local_files_only)

        self.tokenizer = Tokenizer.from_file(fetch("tokenizer.json"))
        self.tokenizer.enable_truncation(MAX_TOKENS)
        self.tokenizer.enable_padding()
        self.session = onnxruntime.InferenceSession(fetch("onnx/model.onnx"), providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.session.get_inputs()}

    def encode(self, texts, show_progress_bar=False):
        """Unit-length embeddings, one row per text. Texts of similar length
        are batched together (as sentence-transformers does), so short ones
        aren't padded to the longest in the folder."""
        order = np.argsort([-len(text) for text in texts], kind="stable")
        rows = [None] * len(texts)
        for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="Batches", disable=not show_progress_bar):
            batch = order[start:start + BATCH_SIZE]
            for i, vector in zip(batch, self._embed([texts[i] for i in batch])):
                rows[i] = vector
        vectors = np.array(rows, dtype=np.float32)
        return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)

    def _embed(self, texts):
        """Mean of the token vectors, padding excluded."""
        encoded = self.tokenizer.encode_batch(texts)
        ids = np.array([e.ids for e in encoded], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
        hidden = self.session.run(None, {name: feeds[name] for name in self.input_names})[0]
        return (hidden * mask[..., None]).sum(axis=1) / np.maximum(mask.sum(axis=1, keepdims=True), 1)
