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
you do need a local embedding cache file. On-the-fly embedding generation uses
`sentence-transformers`.

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

Model and embedding options are configured with separate YAML files:

- `config/embed/from-cache.yaml`: read a precomputed embedding cache
- `config/embed/on-the-fly.yaml`: run an embedding model directly
- `config/model/linear.yaml`: optional linear weight path

The linear model usually uses `config/embed/from-cache.yaml`. Set
`embeddings_path` in that embedding config to a local embedding cache file. The
cache can be either:

- a JSON object: `{"title": [0.1, 0.2, ...]}`
- a JSON list: `[{"title": "A", "embedding": [...]}, ...]`
- a JSONL file with one object per line and fields `title` and `embedding`

Optional linear weights are configured with `weights_path` in
`config/model/linear.yaml`. Supported formats:

- `{"weights": [...] , "bias": 0.0}` where `weights` has size `2 * dim` or `3 * dim`
- `{"link_weights": [...], "target_weights": [...], "interaction_weights": [...], "bias": 0.0}`

If `weights_path` is `null`, the model defaults to a similarity-style scorer with
zero link and target weights, all-ones interaction weights, and zero bias.

Example inference:

```
python inference.py \
  --start-title "Dead 6" \
  --target "Command & Conquer" \
  --model linear \
  --embedding-config config/embed/from-cache.yaml \
  --model-config config/model/linear.yaml
```

Example evaluation:

```
python evaluate_paths.py \
  --split validation \
  --model linear \
  --embedding-config config/embed/from-cache.yaml \
  --model-config config/model/linear.yaml
```

## Embedding Generation

Embedding generation is also config-driven. Fill
`config/embed/on-the-fly.yaml` with the actions dataset, output path, embedding
model, and text source:

```yaml
use_cache: false
actions_path: 0601p/Traveling_Namuwiki_Actions
raw_path: heegyu/namuwiki
split: train
output_path: outputs/title_embeddings.jsonl
model_name: intfloat/multilingual-e5-small
device: null
batch_size: 64
normalize_embeddings: true
text_source: raw_or_title
prefix: passage
cache_inferred: false
```

Then run:

```bash
python generate_embeddings.py --embedding-config config/embed/on-the-fly.yaml
```

The script loads `NamuwikiEnvironment`, collects every graph title and outgoing
action title, then embeds one text per title:

- `title`: embed the title string itself
- `raw`: require raw document text for every title
- `raw_or_title`: use raw document text when available, otherwise fall back to the title

The generated cache is JSONL:

```jsonl
{"title": "A", "embedding": [0.1, 0.2, ...]}
{"title": "B", "embedding": [0.3, 0.4, ...]}
```

## Training AR Walk

`train_ar_walk.py` trains the autoregressive walking policy. Model architecture
settings live in `config/model/ar_walk.yaml`; title embedding settings live in
the shared embedding config, for example `config/embed/on-the-fly.yaml`.

```bash
python train/train_ar_walk.py \
  --model-config config/model/ar_walk.yaml \
  --embedding-config config/embed/on-the-fly.yaml
```

## Training Linear Weights

`train_linear_rl.py` uses the same config files:

```bash
python train/train_linear_rl.py \
  --embedding-config config/embed/from-cache.yaml \
  --model-config config/model/linear.yaml
```

`embedding-config` must point to a cached embedding config. `model-config`
provides the optional initial `weights_path`; trained weights are written under
`outputs/rl` by default.

## Reproducing Completed Experiments

The commands below reproduce the experiments that were actually completed in
this repository state:

- beam-search comparisons on the validation split with `limit=1000`
- HER reward-shaping comparisons for `none`, `lexical`, and `hybrid`
- distance-based shaping runs are intentionally excluded here because those
  runs were not completed

Before running the beam or `hybrid` experiments, update
`config/embed/from-cache.yaml` so that `embeddings_path` points to your local
embedding cache, or generate one first:

```bash
python generate_embeddings.py --embedding-config config/embed/on-the-fly.yaml
```

### Beam Search Validation Runs

The baseline and beam variants below reproduce the validation runs whose
metrics were saved under `outputs/*_val_1000_metrics.json`.

#### Linear vs BeamLinear

```bash
python evaluate_paths.py \
  --split validation \
  --limit 1000 \
  --model linear \
  --embedding-config config/embed/from-cache.yaml \
  --model-config config/model/linear.yaml \
  --predictions-output outputs/linear_val_1000_predictions.jsonl \
  --metrics-output outputs/linear_val_1000_metrics.json

python evaluate_paths.py \
  --split validation \
  --limit 1000 \
  --model beamlinear \
  --embedding-config config/embed/from-cache.yaml \
  --model-config config/model/beamlinear_bw1.yaml \
  --predictions-output outputs/beamlinear_bw1_val_1000_predictions.jsonl \
  --metrics-output outputs/beamlinear_bw1_val_1000_metrics.json

python evaluate_paths.py \
  --split validation \
  --limit 1000 \
  --model beamlinear \
  --embedding-config config/embed/from-cache.yaml \
  --model-config config/model/beamlinear_bw4.yaml \
  --predictions-output outputs/beamlinear_bw4_val_1000_predictions.jsonl \
  --metrics-output outputs/beamlinear_bw4_val_1000_metrics.json

python evaluate_paths.py \
  --split validation \
  --limit 1000 \
  --model beamlinear \
  --embedding-config config/embed/from-cache.yaml \
  --model-config config/model/beamlinear_bw8.yaml \
  --predictions-output outputs/beamlinear_bw8_val_1000_predictions.jsonl \
  --metrics-output outputs/beamlinear_bw8_val_1000_metrics.json
```

#### SemanticWalk vs BeamSemanticWalk

```bash
python evaluate_paths.py \
  --split validation \
  --limit 1000 \
  --model semanticwalk \
  --embedding-config config/embed/from-cache.yaml \
  --model-config config/model/semanticwalk.yaml \
  --predictions-output outputs/semantic_val_1000_predictions.jsonl \
  --metrics-output outputs/semantic_val_1000_metrics.json

python evaluate_paths.py \
  --split validation \
  --limit 1000 \
  --model beamsemanticwalk \
  --embedding-config config/embed/from-cache.yaml \
  --model-config config/model/beamsemanticwalk_bw1.yaml \
  --predictions-output outputs/beamsemantic_bw1_val_1000_predictions.jsonl \
  --metrics-output outputs/beamsemantic_bw1_val_1000_metrics.json

python evaluate_paths.py \
  --split validation \
  --limit 1000 \
  --model beamsemanticwalk \
  --embedding-config config/embed/from-cache.yaml \
  --model-config config/model/beamsemanticwalk_bw4.yaml \
  --predictions-output outputs/beamsemantic_bw4_val_1000_predictions.jsonl \
  --metrics-output outputs/beamsemantic_bw4_val_1000_metrics.json

python evaluate_paths.py \
  --split validation \
  --limit 1000 \
  --model beamsemanticwalk \
  --embedding-config config/embed/from-cache.yaml \
  --model-config config/model/beamsemanticwalk_bw8.yaml \
  --predictions-output outputs/beamsemantic_bw8_val_1000_predictions.jsonl \
  --metrics-output outputs/beamsemantic_bw8_val_1000_metrics.json
```

#### LexicalSimilarityGreedy vs BeamLexicalSimilarityGreedy

```bash
python evaluate_paths.py \
  --split validation \
  --limit 1000 \
  --model lexicalsimilaritygreedy \
  --model-config config/model/lexicalsimilaritygreedy.yaml \
  --predictions-output outputs/lexical_val_1000_predictions.jsonl \
  --metrics-output outputs/lexical_val_1000_metrics.json

python evaluate_paths.py \
  --split validation \
  --limit 1000 \
  --model beamlexicalsimilaritygreedy \
  --model-config config/model/beamlexicalsimilaritygreedy_bw1.yaml \
  --predictions-output outputs/beamlexical_bw1_val_1000_predictions.jsonl \
  --metrics-output outputs/beamlexical_bw1_val_1000_metrics.json

python evaluate_paths.py \
  --split validation \
  --limit 1000 \
  --model beamlexicalsimilaritygreedy \
  --model-config config/model/beamlexicalsimilaritygreedy_bw4.yaml \
  --predictions-output outputs/beamlexical_bw4_val_1000_predictions.jsonl \
  --metrics-output outputs/beamlexical_bw4_val_1000_metrics.json

python evaluate_paths.py \
  --split validation \
  --limit 1000 \
  --model beamlexicalsimilaritygreedy \
  --model-config config/model/beamlexicalsimilaritygreedy_bw8.yaml \
  --predictions-output outputs/beamlexical_bw8_val_1000_predictions.jsonl \
  --metrics-output outputs/beamlexical_bw8_val_1000_metrics.json
```

### HER Reward-Shaping Runs

These runs use `train/train_hindsight_target_a2c_v2.py`. To avoid overwriting
checkpoints, each command writes to its own file under `checkpoints/`.

#### Training

```bash
python train/train_hindsight_target_a2c_v2.py \
  --eval-limit 1000 \
  --reward-shaping none \
  --reward-shaping-coef 0.0 \
  --checkpoint-output checkpoints/her_none.pt

python train/train_hindsight_target_a2c_v2.py \
  --eval-limit 1000 \
  --reward-shaping lexical \
  --reward-shaping-coef 0.1 \
  --checkpoint-output checkpoints/her_lexical_c01.pt

python train/train_hindsight_target_a2c_v2.py \
  --eval-limit 1000 \
  --reward-shaping lexical \
  --reward-shaping-coef 0.2 \
  --checkpoint-output checkpoints/her_lexical_c02.pt

python train/train_hindsight_target_a2c_v2.py \
  --eval-limit 1000 \
  --reward-shaping hybrid \
  --reward-shaping-coef 0.1 \
  --embedding-config config/embed/from-cache.yaml \
  --checkpoint-output checkpoints/her_hybrid_c01.pt

python train/train_hindsight_target_a2c_v2.py \
  --eval-limit 1000 \
  --reward-shaping hybrid \
  --reward-shaping-coef 0.2 \
  --embedding-config config/embed/from-cache.yaml \
  --checkpoint-output checkpoints/her_hybrid_c02.pt
```

#### Validation Evaluation

```bash
python evaluate_paths.py \
  --split validation \
  --limit 1000 \
  --model hindsighttargeta2cv2 \
  --model-config config/model/hindsighttargeta2cv2_none.yaml \
  --predictions-output outputs/eval_her_none_val_1000_predictions.jsonl \
  --metrics-output outputs/eval_her_none_val_1000_metrics.json

python evaluate_paths.py \
  --split validation \
  --limit 1000 \
  --model hindsighttargeta2cv2 \
  --model-config config/model/hindsighttargeta2cv2_lexical_c01.yaml \
  --predictions-output outputs/eval_her_lexical_c01_val_1000_predictions.jsonl \
  --metrics-output outputs/eval_her_lexical_c01_val_1000_metrics.json

python evaluate_paths.py \
  --split validation \
  --limit 1000 \
  --model hindsighttargeta2cv2 \
  --model-config config/model/hindsighttargeta2cv2_lexical_c02.yaml \
  --predictions-output outputs/eval_her_lexical_c02_val_1000_predictions.jsonl \
  --metrics-output outputs/eval_her_lexical_c02_val_1000_metrics.json

python evaluate_paths.py \
  --split validation \
  --limit 1000 \
  --model hindsighttargeta2cv2 \
  --model-config config/model/hindsighttargeta2cv2_hybrid_c01.yaml \
  --predictions-output outputs/eval_her_hybrid_c01_val_1000_predictions.jsonl \
  --metrics-output outputs/eval_her_hybrid_c01_val_1000_metrics.json

python evaluate_paths.py \
  --split validation \
  --limit 1000 \
  --model hindsighttargeta2cv2 \
  --model-config config/model/hindsighttargeta2cv2_hybrid_c02.yaml \
  --predictions-output outputs/eval_her_hybrid_c02_val_1000_predictions.jsonl \
  --metrics-output outputs/eval_her_hybrid_c02_val_1000_metrics.json
```
