"""ModernBERT-base sentence compressor model.

Encodes sentences with a frozen ModernBERT encoder and scores each sentence
with a lightweight keep/drop head.  The runtime path prefers ONNX; training
uses PyTorch/Transformers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from tekmyra.learned_compressor.model import KeepDropScores


PINNED_MODEL_REVISIONS = {
    "answerdotai/ModernBERT-base": "8949b909ec900327062f0ebf497f51aef5e6f0c8",
    "prajjwal1/bert-tiny": "6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837",
    "sentence-transformers/all-MiniLM-L6-v2": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
}


def resolve_model_revision(model_name: str, configured: str | None = None) -> str | None:
    """Return an immutable revision for remote models; local paths need none."""
    if Path(model_name).exists():
        return None
    revision = configured or PINNED_MODEL_REVISIONS.get(model_name)
    if not revision or len(revision) != 40:
        raise ValueError(f"remote model revision is not pinned: {model_name}")
    return revision


@dataclass(frozen=True)
class TransformerConfig:
    """Configuration for :class:`TransformerCompressorModel`."""

    model_name: str = "answerdotai/ModernBERT-base"
    max_length: int = 512
    hidden_size: int = 768
    device: str = "cpu"
    version: str = "2.0.0"
    model_revision: str | None = None


class _MeanPooling(nn.Module):
    """Mean-pool token embeddings using an attention mask."""

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        summed = (hidden_states * mask).sum(dim=1)
        return summed / mask.sum(dim=1).clamp(min=1e-9)


class _KeepDropHead(nn.Module):
    """Lightweight linear head that maps pooled embeddings to keep/drop logits."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.classifier(embeddings)


class TransformerCompressorModel:
    """Sentence-level keep/drop compressor backed by a pretrained transformer.

    The model supports three operating modes:

    1. **Unloaded** - no encoder/tokenizer/head loaded; :meth:`predict` returns
       keep probabilities of ``1.0``.
    2. **Always-keep** - loaded encoder but no head (e.g. after fitting with
       single-class labels); :meth:`predict` returns keep probabilities of ``1.0``.
    3. **Supervised** - loaded encoder plus a trained keep/drop head.
    """

    def __init__(self, config: TransformerConfig | None = None) -> None:
        self.config = config or TransformerConfig()
        self.tokenizer: Any = None
        self.encoder: Any = None
        self.head: _KeepDropHead | None = None
        self._device = torch.device(self.config.device)

    def load_pretrained(self) -> None:
        """Load the pretrained tokenizer, encoder, and a fresh keep/drop head."""
        revision = resolve_model_revision(
            self.config.model_name, self.config.model_revision
        )
        kwargs = {"revision": revision} if revision else {}
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, **kwargs)
        self.encoder = AutoModel.from_pretrained(self.config.model_name, **kwargs)
        self.encoder.to(self._device)
        self.encoder.eval()
        self.head = _KeepDropHead(self.config.hidden_size).to(self._device)
        self.head.eval()

    def fit(self, sentences: list[str], labels: list[int]) -> None:
        """Train the keep/drop head on the given sentences and labels.

        The transformer encoder is kept frozen; only the lightweight head is
        updated.  If the labels contain fewer than two classes the model falls
        back to the always-keep mode by clearing ``self.head``.
        """
        if not sentences or not labels:
            raise ValueError("sentences and labels must not be empty")
        if len(sentences) != len(labels):
            raise ValueError("sentences and labels must have the same length")
        if len(set(labels)) < 2:
            self.head = None
            return

        if self.encoder is None or self.tokenizer is None:
            self.load_pretrained()
        assert self.head is not None
        self.encoder.eval()
        self.head.train()

        optimizer = torch.optim.AdamW(self.head.parameters(), lr=2e-5)
        criterion = nn.CrossEntropyLoss()
        batch_size = 8
        epochs = 3

        for _ in range(epochs):
            for i in range(0, len(sentences), batch_size):
                batch_sents = sentences[i : i + batch_size]
                batch_labels = torch.tensor(
                    labels[i : i + batch_size], device=self._device
                )
                embeddings = self._encode(batch_sents)
                logits = self.head(embeddings)
                loss = criterion(logits, batch_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.head.eval()

    def _encode(self, sentences: list[str]) -> torch.Tensor:
        """Encode a list of sentences into pooled embeddings."""
        if self.tokenizer is None or self.encoder is None:
            raise RuntimeError("Model not loaded")
        inputs = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.encoder(**inputs)
        return _MeanPooling()(outputs.last_hidden_state, inputs["attention_mask"])

    def predict(self, sentences: list[str]) -> list[KeepDropScores]:
        """Return keep/drop probabilities for each input sentence."""
        if not sentences:
            return []
        if self.encoder is None or self.tokenizer is None:
            return [
                KeepDropScores(keep_prob=1.0, drop_prob=0.0) for _ in sentences
            ]

        embeddings = self._encode(sentences)
        if self.head is None:
            return [
                KeepDropScores(keep_prob=1.0, drop_prob=0.0) for _ in sentences
            ]

        assert self.head is not None
        with torch.no_grad():
            logits = self.head(embeddings)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        return [
            KeepDropScores(keep_prob=float(p[1]), drop_prob=float(p[0]))
            for p in probs
        ]

    def save(self, directory: Path) -> None:
        """Persist encoder, tokenizer, head, config, and metadata."""
        directory.mkdir(parents=True, exist_ok=True)
        if self.encoder is not None:
            self.encoder.save_pretrained(directory / "encoder")
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(directory / "encoder")
        if self.head is not None:
            torch.save(self.head.state_dict(), directory / "head.pt")
        with open(directory / "config.json", "w") as fh:
            json.dump(self.config.__dict__, fh, indent=2)
        with open(directory / "metadata.json", "w") as fh:
            json.dump(
                {"version": self.config.version, "kind": "transformer_compressor"},
                fh,
            )

    @classmethod
    def load(cls, directory: Path) -> "TransformerCompressorModel":
        """Load a previously saved :class:`TransformerCompressorModel`."""
        with open(directory / "config.json") as fh:
            config = TransformerConfig(**json.load(fh))
        model = cls(config)
        model.tokenizer = AutoTokenizer.from_pretrained(directory / "encoder")
        model.encoder = AutoModel.from_pretrained(directory / "encoder")
        model.encoder.to(model._device)
        model.encoder.eval()
        head_path = directory / "head.pt"
        if head_path.exists():
            model.head = _KeepDropHead(config.hidden_size).to(model._device)
            model.head.load_state_dict(
                torch.load(
                    head_path,
                    map_location=model._device,
                    weights_only=True,
                )
            )
            model.head.eval()
        return model
