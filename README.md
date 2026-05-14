# Traveling Namuwiki

Minimal graph-navigation runner for the Traveling Namuwiki datasets.

## Datasets

By default, scripts load these Hugging Face datasets with `datasets.load_dataset()`:

- `0601p/Traveling_Namuwiki_Actions`
- `0601p/Traveling_Namuwiki_Paths`

The actions dataset provides the environment state/action graph. The paths
dataset provides evaluation examples with `start_title`, `target_title`, and
`hop`.

## Install

```
pip install datasets
```

If you want to use the linear embedding model, no extra package is required, but
you do need a local embedding file.

## Inference

Run a single walk from a start title to a target title:

```
python inference.py --start-title "Dead 6" --target "Command & Conquer" --model randomwalk --max-steps 10
```

Use a different actions dataset path if needed:

```
python inference.py --actions-path 0601p/Traveling_Namuwiki_Actions --start-title "Dead 6" --target "Command & Conquer"
```

## Evaluation

Run a quick validation smoke test:

```
python evaluate_paths.py --split validation --model randomwalk --max-steps 10 --limit 100
```

Run the full test split:

```
python evaluate_paths.py --split test --model randomwalk --max-steps 10
```

The main score is `score_by_min_distance`, a dictionary from gold `hop` to the
average model distance. Failed searches are assigned `max_steps + 1` unless
`--failure-distance` is set.

Per-example predictions are saved by default:

```text
outputs\{split}_{model}_predictions.jsonl
```

Summary metrics are also saved by default:

```text
outputs\{split}_{model}_metrics.json
```

Override the prediction path if needed:

```
python evaluate_paths.py --split validation --predictions-output outputs\validation_predictions.jsonl
```

Override the metrics path if needed:

```
python evaluate_paths.py --split validation --metrics-output outputs\validation_metrics.json
```

## Models

Models live under `models/`.

- `models/base.py`: `Model` interface
- `models/randomwalk.py`: `RandomWalk`
- `models/__init__.py`: model registry and `create_model()`

To add a model, implement a `Model` subclass and register it in
`models/__init__.py`.

## Linear Model

`linear` scores each outgoing link with a linear function of:

- the candidate link embedding
- the target document embedding
- their elementwise interaction

The model expects a local embedding file passed via `--embeddings-path`. The
file can be either:

- a JSON object: `{"title": [0.1, 0.2, ...]}`
- a JSON list: `[{"title": "A", "embedding": [...]}, ...]`
- a JSONL file with one object per line and fields `title` and `embedding`

Optional linear weights can be passed via `--weights-path`. Supported formats:

- `{"weights": [...] , "bias": 0.0}` where `weights` has size `2 * dim` or `3 * dim`
- `{"link_weights": [...], "target_weights": [...], "interaction_weights": [...], "bias": 0.0}`

If `--weights-path` is omitted, the model defaults to a similarity-style scorer
with zero link and target weights, all-ones interaction weights, and zero bias.

Example inference:

```
python inference.py \
  --start-title "Dead 6" \
  --target "Command & Conquer" \
  --model linear \
  --embeddings-path data/title_embeddings.json
```

Example evaluation:

```
python evaluate_paths.py \
  --split validation \
  --model linear \
  --embeddings-path data/title_embeddings.json \
  --weights-path data/linear_weights.json
```

## Embedding Generation

You can generate title-keyed embeddings from the actions graph and raw Namuwiki
documents with:

```
python generate_embeddings.py \
  --output-path outputs/title_embeddings.json \
  --text-source raw_or_title
```

The script collects every graph title and outgoing action title, then embeds one
text per title:

- `title`: embed the title string itself
- `raw`: require raw document text for every title
- `raw_or_title`: use raw document text when available, otherwise fall back to the title
