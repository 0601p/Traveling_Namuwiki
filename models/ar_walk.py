
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()

from collections import OrderedDict
from typing import cast

import torch
from sentence_transformers import SentenceTransformer
from torch import Tensor, nn

from utils import Action, Page

from .base import Model


class TitleEmbeddingCache:
    """SentenceTransformer-backed title encoder with a small LRU cache."""

    def __init__(
        self,
        model_name: str,
        *,
        cache_size: int = 8192,
        device: str | torch.device | None = None,
    ) -> None:
        """Load the encoder and initialize the bounded title embedding cache."""
        self.model = SentenceTransformer(model_name, device=device)
        self.cache_size = cache_size
        self._cache: OrderedDict[str, Tensor] = OrderedDict()
        self.embedding_dim = self._infer_embedding_dim()

    def _infer_embedding_dim(self) -> int:
        """Determine the embedding width exposed by the sentence encoder."""
        dimension = getattr(self.model, "get_sentence_embedding_dimension", None)
        if callable(dimension):
            result = dimension()
            if result is not None:
                return int(result)

        sample = self.encode([""])
        return int(sample.shape[-1])

    def _cache_get(self, title: str) -> Tensor | None:
        """Return a cached embedding and refresh its recency in the LRU order."""
        cached = self._cache.get(title)
        if cached is None:
            return None

        self._cache.move_to_end(title)
        return cached.clone()

    def _cache_put(self, title: str, embedding: Tensor) -> None:
        """Insert an embedding and evict the least recently used item if needed."""
        self._cache[title] = embedding.detach().cpu().clone()
        self._cache.move_to_end(title)
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def encode(self, titles: list[str]) -> Tensor:
        """Encode a list of titles while avoiding recomputation for cache hits."""
        if not titles:
            return torch.empty((0, self.embedding_dim), dtype=torch.float32)

        embeddings: list[Tensor | None] = [None] * len(titles)
        missing_titles: list[str] = []
        missing_indices: list[int] = []

        for index, title in enumerate(titles):
            cached = self._cache_get(title)
            if cached is None:
                missing_titles.append(title)
                missing_indices.append(index)
            else:
                embeddings[index] = cached

        if missing_titles:
            # Only uncached titles are passed through the sentence encoder.
            encoded = self.model.encode(
                missing_titles,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
            encoded_tensor = torch.as_tensor(encoded, dtype=torch.float32)
            for offset, title in enumerate(missing_titles):
                embedding = encoded_tensor[offset].clone()
                self._cache_put(title, embedding)
                embeddings[missing_indices[offset]] = embedding

        return torch.stack(
            [cast(Tensor, embedding) for embedding in embeddings],
            dim=0,
        )


class TrajectoryEncoder(nn.Module):
    """Causal transformer over the visited title sequence."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        ff_dim: int,
    ) -> None:
        """Build the autoregressive transformer used to encode visit history."""
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def _causal_mask(self, sequence_length: int, device: torch.device) -> Tensor:
        """Mask future positions so each step can only attend to its past."""
        mask = torch.full(
            (sequence_length, sequence_length),
            float("-inf"),
            device=device,
        )
        return torch.triu(mask, diagonal=1)

    def forward(self, trajectory_embeddings: Tensor) -> Tensor:
        """Encode the trajectory with strict left-to-right attention."""
        sequence_length = trajectory_embeddings.shape[1]
        mask = self._causal_mask(sequence_length, trajectory_embeddings.device)
        return self.encoder(trajectory_embeddings, mask=mask)


class TargetConditioner(nn.Module):
    """Fuse the trajectory context with the target title embedding."""

    def __init__(self, *, hidden_dim: int, dropout: float) -> None:
        """Create the MLP that injects target information into the current state."""
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, context: Tensor, target_embedding: Tensor) -> Tensor:
        """Merge the encoded current context and target embedding into one vector."""
        fused = torch.cat([context, target_embedding], dim=-1)
        return self.network(fused)


class ActorHead(nn.Module):
    """Project the conditioned context onto candidate action embeddings."""

    def forward(
        self,
        conditioned_context: Tensor,
        action_embeddings: Tensor,
    ) -> Tensor:
        """Score each candidate by dot-product similarity to the policy state."""
        return action_embeddings @ conditioned_context


class AutoregressivePolicyNetwork(nn.Module):
    """Decision-Transformer-like policy backbone for action scoring."""

    def __init__(
        self,
        *,
        embedding_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        ff_dim: int | None = None,
    ) -> None:
        """Assemble the causal encoder, target conditioner, and actor head."""
        super().__init__()
        self.trajectory_encoder = TrajectoryEncoder(
            hidden_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            ff_dim=ff_dim or embedding_dim * 4,
        )
        self.target_conditioner = TargetConditioner(
            hidden_dim=embedding_dim,
            dropout=dropout,
        )
        self.actor_head = ActorHead()

    def forward(
        self,
        *,
        trajectory_embeddings: Tensor,
        target_embedding: Tensor,
        action_embeddings: Tensor,
    ) -> Tensor:
        """Produce one action logit for each outgoing link candidate."""
        encoded_trajectory = self.trajectory_encoder(trajectory_embeddings)
        current_context = encoded_trajectory[:, -1, :]
        conditioned_context = self.target_conditioner(
            current_context,
            target_embedding,
        )
        return self.actor_head(conditioned_context.squeeze(0), action_embeddings)


class AutoregressiveWalk(Model):
    """Greedy autoregressive policy over candidate links."""

    def __init__(
        self,
        *,
        encoder_name: str = "google/embeddinggemma-300m",
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        ff_dim: int | None = None,
        cache_size: int = 8192,
        device: str | torch.device | None = None,
    ) -> None:
        """Initialize the title encoder and policy network for inference."""
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.embedder = TitleEmbeddingCache(
            encoder_name,
            cache_size=cache_size,
            device=self.device,
        )
        self.network = AutoregressivePolicyNetwork(
            embedding_dim=self.embedder.embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            ff_dim=ff_dim,
        ).to(self.device)
        self.network.eval()

    def _encode_history(self, history: list[str]) -> Tensor:
        """Encode the visited-title sequence as a batch of one trajectory."""
        history_embeddings = self.embedder.encode(history)
        return history_embeddings.unsqueeze(0).to(self.device)

    def _encode_target(self, target: str) -> Tensor:
        """Encode the target title as a single embedding vector."""
        return self.embedder.encode([target]).to(self.device)

    def _encode_actions(self, actions: list[Action]) -> Tensor:
        """Encode the current page's candidate outgoing links."""
        return self.embedder.encode(actions).to(self.device)

    @torch.inference_mode()
    def sample(self, page: Page, target: str, history: list[str]) -> Action | None:
        """Sample one action directly from the model's logit distribution."""
        if not page.actions:
            return None

        trajectory_embeddings = self._encode_history(history)
        target_embedding = self._encode_target(target)
        action_embeddings = self._encode_actions(list(page.actions))

        logits = self.network(
            trajectory_embeddings=trajectory_embeddings,
            target_embedding=target_embedding,
            action_embeddings=action_embeddings,
        )
        # Sample from the categorical distribution induced by the action logits.
        best_index = int(torch.distributions.Categorical(logits=logits).sample().item())
        return page.actions[best_index]
