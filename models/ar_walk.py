
from __future__ import annotations

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

import random
from abc import ABC, abstractmethod
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence, cast

import torch
from sentence_transformers import SentenceTransformer
from torch import Tensor, nn

from utils import Action, Page, Title

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


@dataclass(frozen=True)
class TrainingTask:
    """One unsupervised wiki-racing task sampled from the action graph."""

    start_title: Title
    target_title: Title


@dataclass
class RolloutStep:
    """Bookkeeping for one policy decision in an episode."""

    current_title: Title
    action: Action
    log_prob: Tensor
    entropy: Tensor


@dataclass
class RolloutResult:
    """A sampled episode and the differentiable tensors needed for training."""

    task: TrainingTask
    visited: list[Title]
    stopped_reason: str
    steps: list[RolloutStep] = field(default_factory=list)

    @property
    def reached_target(self) -> bool:
        return self.stopped_reason == "target_reached"

    @property
    def steps_taken(self) -> int:
        return len(self.visited) - 1


@dataclass(frozen=True)
class ReinforceConfig:
    """Hyperparameters for the basic no-gold policy-gradient trainer."""

    episodes: int = 1000
    max_steps: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-4
    entropy_coef: float = 0.01
    baseline_decay: float = 0.95
    success_reward: float = 1.0
    failure_penalty: float = -0.2
    step_penalty: float = -0.01
    dead_end_penalty: float = -0.1
    cycle_penalty: float = -0.1
    max_grad_norm: float = 1.0
    temperature: float = 1.0
    stop_on_cycle: bool = True
    log_every: int = 50


@dataclass(frozen=True)
class TrainingSummary:
    """Compact summary returned by a training run."""

    episodes: int
    average_reward: float
    success_rate: float
    average_steps: float
    stopped_reasons: dict[str, int]
    last_loss: float | None


class TaskSampler(ABC):
    """Source of start/target pairs for no-gold training."""

    @abstractmethod
    def sample(self) -> TrainingTask:
        raise NotImplementedError


class RandomPairTaskSampler(TaskSampler):
    """Uniformly sample start and target titles from the known graph."""

    def __init__(
        self,
        graph: Mapping[Title, Sequence[Action]],
        *,
        seed: int | None = None,
        allow_same_title: bool = False,
    ) -> None:
        start_titles = [title for title, actions in graph.items() if actions]
        target_titles = sorted(
            set(graph).union(action for actions in graph.values() for action in actions)
        )
        if len(target_titles) < 2 and not allow_same_title:
            raise ValueError("At least two titles with actions are required.")
        if not start_titles or not target_titles:
            raise ValueError("The action graph is empty.")

        self.start_titles = start_titles
        self.target_titles = target_titles
        self.rng = random.Random(seed)
        self.allow_same_title = allow_same_title

    def sample(self) -> TrainingTask:
        start = self.rng.choice(self.start_titles)
        target = self.rng.choice(self.target_titles)
        while not self.allow_same_title and target == start:
            target = self.rng.choice(self.target_titles)
        return TrainingTask(start_title=start, target_title=target)


class TrainingAlgorithm(ABC):
    """Swappable training procedure for AutoregressiveWalk."""

    name: str

    @abstractmethod
    def train(
        self,
        model: AutoregressiveWalk,
        graph: Mapping[Title, Sequence[Action]],
        *,
        sampler: TaskSampler | None = None,
    ) -> TrainingSummary:
        raise NotImplementedError


class BasicReinforceTrainer(TrainingAlgorithm):
    """Sparse-reward REINFORCE trainer that does not use shortest paths."""

    name = "basic_reinforce"

    def __init__(self, config: ReinforceConfig | None = None) -> None:
        self.config = config or ReinforceConfig()
        self.reward_baseline = 0.0
        self._baseline_initialized = False

    def train(
        self,
        model: AutoregressiveWalk,
        graph: Mapping[Title, Sequence[Action]],
        *,
        sampler: TaskSampler | None = None,
    ) -> TrainingSummary:
        config = self.config
        task_sampler = sampler or RandomPairTaskSampler(graph)
        optimizer = torch.optim.AdamW(model.network.parameters(), lr=config.learning_rate)

        rewards: list[float] = []
        step_counts: list[int] = []
        stopped_reasons: Counter[str] = Counter()
        last_loss: float | None = None

        model.network.train()
        optimizer.zero_grad(set_to_none=True)

        for episode_index in range(1, config.episodes + 1):
            rollout = self.rollout(model, graph, task_sampler.sample())
            reward = self.reward(rollout)
            loss = self.loss(rollout, reward)

            if loss is not None:
                (loss / config.batch_size).backward()
                last_loss = float(loss.detach().cpu().item())

            if episode_index % config.batch_size == 0:
                nn.utils.clip_grad_norm_(model.network.parameters(), config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            rewards.append(reward)
            step_counts.append(rollout.steps_taken)
            stopped_reasons[rollout.stopped_reason] += 1

            if config.log_every and episode_index % config.log_every == 0:
                recent_rewards = rewards[-config.log_every :]
                recent_successes = stopped_reasons["target_reached"]
                print(
                    {
                        "episode": episode_index,
                        "avg_reward": sum(recent_rewards) / len(recent_rewards),
                        "successes": recent_successes,
                        "last_loss": last_loss,
                    }
                )

        leftover = config.episodes % config.batch_size
        if leftover:
            nn.utils.clip_grad_norm_(model.network.parameters(), config.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        model.network.eval()
        total = len(rewards)
        success_count = stopped_reasons["target_reached"]
        return TrainingSummary(
            episodes=total,
            average_reward=sum(rewards) / total if total else 0.0,
            success_rate=success_count / total if total else 0.0,
            average_steps=sum(step_counts) / total if total else 0.0,
            stopped_reasons=dict(stopped_reasons),
            last_loss=last_loss,
        )

    def rollout(
        self,
        model: AutoregressiveWalk,
        graph: Mapping[Title, Sequence[Action]],
        task: TrainingTask,
    ) -> RolloutResult:
        config = self.config
        visited = [task.start_title]
        seen = {task.start_title}
        steps: list[RolloutStep] = []
        current = task.start_title

        for _ in range(config.max_steps):
            actions = list(dict.fromkeys(graph.get(current, [])))
            if not actions:
                return RolloutResult(task, visited, "no_actions", steps)

            logits = model.action_logits(visited, task.target_title, actions)
            if config.temperature != 1.0:
                logits = logits / config.temperature
            distribution = torch.distributions.Categorical(logits=logits)
            action_index = distribution.sample()
            next_title = actions[int(action_index.item())]

            steps.append(
                RolloutStep(
                    current_title=current,
                    action=next_title,
                    log_prob=distribution.log_prob(action_index),
                    entropy=distribution.entropy(),
                )
            )

            visited.append(next_title)
            if next_title == task.target_title:
                return RolloutResult(task, visited, "target_reached", steps)

            if config.stop_on_cycle and next_title in seen:
                return RolloutResult(task, visited, "cycle", steps)

            seen.add(next_title)
            current = next_title

        return RolloutResult(task, visited, "max_steps", steps)

    def reward(self, rollout: RolloutResult) -> float:
        config = self.config
        reward = config.step_penalty * rollout.steps_taken
        if rollout.reached_target:
            return config.success_reward + reward
        if rollout.stopped_reason == "no_actions":
            reward += config.dead_end_penalty
        elif rollout.stopped_reason == "cycle":
            reward += config.cycle_penalty
        return config.failure_penalty + reward

    def loss(self, rollout: RolloutResult, reward: float) -> Tensor | None:
        if not rollout.steps:
            self._update_baseline(reward)
            return None

        previous_baseline = self.reward_baseline
        self._update_baseline(reward)
        advantage = reward - previous_baseline

        log_probs = torch.stack([step.log_prob for step in rollout.steps])
        entropies = torch.stack([step.entropy for step in rollout.steps])
        policy_loss = -log_probs.sum() * advantage
        entropy_bonus = entropies.mean() * self.config.entropy_coef
        return policy_loss - entropy_bonus

    def _update_baseline(self, reward: float) -> None:
        if not self._baseline_initialized:
            self.reward_baseline = reward
            self._baseline_initialized = True
            return

        decay = self.config.baseline_decay
        self.reward_baseline = decay * self.reward_baseline + (1.0 - decay) * reward


TRAINING_ALGORITHMS: dict[str, type[TrainingAlgorithm]] = {
    BasicReinforceTrainer.name: BasicReinforceTrainer,
    "reinforce": BasicReinforceTrainer,
    "basic": BasicReinforceTrainer,
}


def create_training_algorithm(
    name: str,
    *,
    config: ReinforceConfig | None = None,
) -> TrainingAlgorithm:
    """Instantiate a registered trainer by name."""
    normalized = name.strip().lower().replace("-", "_")
    if normalized not in TRAINING_ALGORITHMS:
        available = ", ".join(sorted(TRAINING_ALGORITHMS))
        raise NotImplementedError(
            f"Unknown training algorithm: {name}. Available: {available}"
        )
    algorithm_class = TRAINING_ALGORITHMS[normalized]
    if algorithm_class is BasicReinforceTrainer:
        return BasicReinforceTrainer(config)
    return algorithm_class()


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

    def save_checkpoint(self, path: str | Path) -> None:
        """Persist the trainable policy network weights."""
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"network": self.network.state_dict()}, checkpoint_path)

    def load_checkpoint(self, path: str | Path) -> None:
        """Load policy weights produced by save_checkpoint()."""
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint.get("network", checkpoint)
        self.network.load_state_dict(state_dict)
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

    def action_logits(
        self,
        history: list[str],
        target: str,
        actions: list[Action],
    ) -> Tensor:
        """Score outgoing actions for either sampling or training."""
        trajectory_embeddings = self._encode_history(history)
        target_embedding = self._encode_target(target)
        action_embeddings = self._encode_actions(actions)
        return self.network(
            trajectory_embeddings=trajectory_embeddings,
            target_embedding=target_embedding,
            action_embeddings=action_embeddings,
        )

    def train_policy(
        self,
        graph: Mapping[Title, Sequence[Action]],
        *,
        algorithm: str | TrainingAlgorithm = "basic_reinforce",
        config: ReinforceConfig | None = None,
        sampler: TaskSampler | None = None,
    ) -> TrainingSummary:
        """Train the policy with a pluggable no-gold training algorithm."""
        trainer = (
            create_training_algorithm(algorithm, config=config)
            if isinstance(algorithm, str)
            else algorithm
        )
        return trainer.train(self, graph, sampler=sampler)

    @torch.inference_mode()
    def sample(self, page: Page, target: str, history: list[str]) -> Action | None:
        """Sample one action directly from the model's logit distribution."""
        if not page.actions:
            return None

        logits = self.action_logits(history, target, list(page.actions))
        # Sample from the categorical distribution induced by the action logits.
        best_index = int(torch.distributions.Categorical(logits=logits).sample().item())
        return page.actions[best_index]
