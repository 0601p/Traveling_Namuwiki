# BC / PPO / HER Reproducibility

This file documents the direct Python commands used for the BC, PPO, HER
variation, and PPO+HER experiments. If W&B logging is needed, append
`--wandb-project traveling-namuwiki --wandb-run-name <run_name>` to the training
commands.

## Notes

- Evaluation commands below use `--limit 1000` on the test split.
- Run the BC command first before any `--init-checkpoint checkpoints/bc.pt`
  PPO+HER experiment.

## Methods

| Run | Method |
| --- | --- |
| `bc` | Behavior cloning from gold path actions. It trains only with cross-entropy on `(current, target, outgoing links) -> gold next link`. |
| `ppo` | PPO with GAE, no BC and no HER by default. |
| `ppo_prior` | PPO with lexical residual prior, default `PRIOR_ALPHA=5.0`. |
| `her_future` | A2C-style HER where hindsight goals are sampled uniformly from future visited pages. |
| `her_semantic` | HER where future hindsight goals are sampled with title-similarity weights against the original target. |
| `her_mixed` | HER using the final reached page plus semantic-weighted future goals. |
| `ppo_her` | PPO/GAE update plus auxiliary HER relabeling on the same rollout. |
| `ppo_her_bc_init` | PPO+HER initialized from `checkpoints/bc.pt`. This was the best local run. |

## Core Train Commands

```bash
python train/train_bc.py --epochs 10 --max-steps 10 --train-limit 5000 --eval-limit 1000 --bc-samples-mult 4 --lr 1e-4 --prior-alpha 0.0 --seed 0 --checkpoint-output checkpoints/bc.pt

python train/train_ppo.py --epochs 8 --max-steps 10 --train-limit 5000 --eval-limit 1000 --gamma 0.92 --gae-lambda 0.95 --lr 1e-4 --clip-eps 0.1 --ppo-epochs 2 --entropy-coef 0.003 --value-coef 0.5 --prior-alpha 0.0 --seed 0 --checkpoint-output checkpoints/ppo.pt

python train/train_ppo.py --epochs 8 --max-steps 10 --train-limit 5000 --eval-limit 1000 --gamma 0.92 --gae-lambda 0.95 --lr 1e-4 --clip-eps 0.1 --ppo-epochs 2 --entropy-coef 0.003 --value-coef 0.5 --prior-alpha 5.0 --seed 0 --checkpoint-output checkpoints/ppo_prior_alpha5.0.pt

python train/train_her_variation.py --her-strategy future --her-k 2 --her-coef 0.2 --her-temp 1.0 --epochs 8 --max-steps 10 --train-limit 5000 --eval-limit 1000 --gamma 0.92 --lr 3e-4 --entropy-coef 0.01 --value-coef 0.5 --seed 0 --checkpoint-output checkpoints/her_future.pt

python train/train_her_variation.py --her-strategy semantic --her-k 2 --her-coef 0.2 --her-temp 1.0 --epochs 8 --max-steps 10 --train-limit 5000 --eval-limit 1000 --gamma 0.92 --lr 3e-4 --entropy-coef 0.01 --value-coef 0.5 --seed 0 --checkpoint-output checkpoints/her_semantic.pt

python train/train_her_variation.py --her-strategy mixed --her-k 2 --her-coef 0.2 --her-temp 1.0 --epochs 8 --max-steps 10 --train-limit 5000 --eval-limit 1000 --gamma 0.92 --lr 3e-4 --entropy-coef 0.01 --value-coef 0.5 --seed 0 --checkpoint-output checkpoints/her_mixed.pt

python train/train_ppo_her.py --epochs 8 --max-steps 10 --train-limit 5000 --eval-limit 1000 --gamma 0.92 --gae-lambda 0.95 --lr 1e-4 --clip-eps 0.1 --ppo-epochs 2 --entropy-coef 0.003 --value-coef 0.5 --her-strategy mixed --her-k 2 --her-coef 0.1 --her-temp 1.0 --bc-aux-coef 0.0 --bc-aux-samples 1024 --bc-samples-mult 4 --shaping-beta 0.0 --prior-alpha 0.0 --seed 0 --checkpoint-output checkpoints/ppo_her.pt

python train/train_ppo_her.py --epochs 8 --max-steps 10 --train-limit 5000 --eval-limit 1000 --gamma 0.92 --gae-lambda 0.95 --lr 1e-4 --clip-eps 0.1 --ppo-epochs 2 --entropy-coef 0.003 --value-coef 0.5 --her-strategy mixed --her-k 2 --her-coef 0.1 --her-temp 1.0 --bc-aux-coef 0.0 --bc-aux-samples 1024 --bc-samples-mult 4 --shaping-beta 0.0 --prior-alpha 0.0 --seed 0 --init-checkpoint checkpoints/bc.pt --checkpoint-output checkpoints/ppo_her_bc_init.pt
```

## Core Eval Commands

```bash
python evaluate_paths.py --split test --model bc --model-config config/model/bc.yaml --max-steps 10 --max-gold-distance 10 --limit 1000 --predictions-output outputs/test_bc_predictions.jsonl --metrics-output outputs/test_bc_metrics.json

python evaluate_paths.py --split test --model ppo --model-config config/model/ppo.yaml --max-steps 10 --max-gold-distance 10 --limit 1000 --predictions-output outputs/test_ppo_predictions.jsonl --metrics-output outputs/test_ppo_metrics.json

python evaluate_paths.py --split test --model residual_ppo --model-config config/model/ppo_prior.yaml --max-steps 10 --max-gold-distance 10 --limit 1000 --predictions-output outputs/test_ppo_prior_predictions.jsonl --metrics-output outputs/test_ppo_prior_metrics.json

python evaluate_paths.py --split test --model her_future --model-config config/model/her_future.yaml --max-steps 10 --max-gold-distance 10 --limit 1000 --predictions-output outputs/test_her_future_predictions.jsonl --metrics-output outputs/test_her_future_metrics.json

python evaluate_paths.py --split test --model her_semantic --model-config config/model/her_semantic.yaml --max-steps 10 --max-gold-distance 10 --limit 1000 --predictions-output outputs/test_her_semantic_predictions.jsonl --metrics-output outputs/test_her_semantic_metrics.json

python evaluate_paths.py --split test --model her_mixed --model-config config/model/her_mixed.yaml --max-steps 10 --max-gold-distance 10 --limit 1000 --predictions-output outputs/test_her_mixed_predictions.jsonl --metrics-output outputs/test_her_mixed_metrics.json

python evaluate_paths.py --split test --model ppo_her --model-config config/model/ppo_her.yaml --max-steps 10 --max-gold-distance 10 --limit 1000 --predictions-output outputs/test_ppo_her_predictions.jsonl --metrics-output outputs/test_ppo_her_metrics.json

python evaluate_paths.py --split test --model ppo_her --model-config config/model/ppo_her_bc_init.yaml --max-steps 10 --max-gold-distance 10 --limit 1000 --predictions-output outputs/test_ppo_her_bc_init_predictions.jsonl --metrics-output outputs/test_ppo_her_bc_init_metrics.json
```

## Additional Sweep Train Commands

```bash
python train/train_her_variation.py --her-strategy mixed --her-k 4 --her-coef 0.2 --her-temp 1.0 --epochs 8 --max-steps 10 --train-limit 5000 --eval-limit 1000 --gamma 0.92 --lr 3e-4 --entropy-coef 0.01 --value-coef 0.5 --seed 0 --checkpoint-output checkpoints/her_mixed_k4.pt

python train/train_her_variation.py --her-strategy future --her-k 4 --her-coef 0.2 --her-temp 1.0 --epochs 8 --max-steps 10 --train-limit 5000 --eval-limit 1000 --gamma 0.92 --lr 3e-4 --entropy-coef 0.01 --value-coef 0.5 --seed 0 --checkpoint-output checkpoints/her_future_k4.pt

python train/train_her_variation.py --her-strategy semantic --her-k 3 --her-coef 0.2 --her-temp 1.0 --epochs 8 --max-steps 10 --train-limit 5000 --eval-limit 1000 --gamma 0.92 --lr 3e-4 --entropy-coef 0.01 --value-coef 0.5 --seed 0 --checkpoint-output checkpoints/her_semantic_k3.pt

python train/train_ppo_her.py --epochs 8 --max-steps 10 --train-limit 5000 --eval-limit 1000 --gamma 0.92 --gae-lambda 0.95 --lr 1e-4 --clip-eps 0.1 --ppo-epochs 2 --entropy-coef 0.003 --value-coef 0.5 --her-strategy mixed --her-k 4 --her-coef 0.1 --her-temp 1.0 --bc-aux-coef 0.0 --bc-aux-samples 1024 --bc-samples-mult 4 --shaping-beta 0.0 --prior-alpha 0.0 --seed 0 --checkpoint-output checkpoints/ppo_her_k4.pt

python train/train_ppo_her.py --epochs 8 --max-steps 10 --train-limit 5000 --eval-limit 1000 --gamma 0.92 --gae-lambda 0.95 --lr 1e-4 --clip-eps 0.1 --ppo-epochs 2 --entropy-coef 0.003 --value-coef 0.5 --her-strategy mixed --her-k 2 --her-coef 0.1 --her-temp 1.0 --bc-aux-coef 0.0 --bc-aux-samples 1024 --bc-samples-mult 4 --shaping-beta 0.0 --prior-alpha 2.0 --seed 0 --checkpoint-output checkpoints/ppo_her_prior_alpha2.0.pt

python train/train_ppo_her.py --epochs 8 --max-steps 10 --train-limit 5000 --eval-limit 1000 --gamma 0.92 --gae-lambda 0.95 --lr 1e-4 --clip-eps 0.1 --ppo-epochs 2 --entropy-coef 0.003 --value-coef 0.5 --her-strategy mixed --her-k 2 --her-coef 0.1 --her-temp 1.0 --bc-aux-coef 0.0 --bc-aux-samples 1024 --bc-samples-mult 4 --shaping-beta 0.1 --prior-alpha 0.0 --seed 0 --checkpoint-output checkpoints/ppo_her_shaped.pt

python train/train_ppo_her.py --epochs 8 --max-steps 10 --train-limit 5000 --eval-limit 1000 --gamma 0.92 --gae-lambda 0.95 --lr 1e-4 --clip-eps 0.1 --ppo-epochs 2 --entropy-coef 0.003 --value-coef 0.5 --her-strategy mixed --her-k 2 --her-coef 0.1 --her-temp 1.0 --bc-aux-coef 0.05 --bc-aux-samples 1024 --bc-samples-mult 4 --shaping-beta 0.0 --prior-alpha 0.0 --seed 0 --checkpoint-output checkpoints/ppo_her_bc_aux.pt
```

## BC-Initialized PPO+HER Sweep Train Commands

```bash
python train/train_ppo_her.py --epochs 8 --max-steps 10 --train-limit 5000 --eval-limit 1000 --gamma 0.92 --gae-lambda 0.95 --lr 1e-4 --clip-eps 0.1 --ppo-epochs 2 --entropy-coef 0.003 --value-coef 0.5 --her-strategy future --her-k 2 --her-coef 0.1 --her-temp 1.0 --bc-aux-coef 0.0 --bc-aux-samples 1024 --bc-samples-mult 4 --shaping-beta 0.0 --prior-alpha 0.0 --seed 0 --init-checkpoint checkpoints/bc.pt --checkpoint-output checkpoints/ppo_her_bc_init_future.pt

python train/train_ppo_her.py --epochs 8 --max-steps 10 --train-limit 5000 --eval-limit 1000 --gamma 0.92 --gae-lambda 0.95 --lr 1e-4 --clip-eps 0.1 --ppo-epochs 2 --entropy-coef 0.003 --value-coef 0.5 --her-strategy semantic --her-k 2 --her-coef 0.1 --her-temp 1.0 --bc-aux-coef 0.0 --bc-aux-samples 1024 --bc-samples-mult 4 --shaping-beta 0.0 --prior-alpha 0.0 --seed 0 --init-checkpoint checkpoints/bc.pt --checkpoint-output checkpoints/ppo_her_bc_init_semantic.pt

python train/train_ppo_her.py --epochs 8 --max-steps 10 --train-limit 5000 --eval-limit 1000 --gamma 0.92 --gae-lambda 0.95 --lr 5e-5 --clip-eps 0.1 --ppo-epochs 2 --entropy-coef 0.003 --value-coef 0.5 --her-strategy mixed --her-k 2 --her-coef 0.1 --her-temp 1.0 --bc-aux-coef 0.0 --bc-aux-samples 1024 --bc-samples-mult 4 --shaping-beta 0.0 --prior-alpha 0.0 --seed 0 --init-checkpoint checkpoints/bc.pt --checkpoint-output checkpoints/ppo_her_bc_init_lr5e5.pt

python train/train_ppo_her.py --epochs 8 --max-steps 10 --train-limit 5000 --eval-limit 1000 --gamma 0.92 --gae-lambda 0.95 --lr 1e-4 --clip-eps 0.1 --ppo-epochs 2 --entropy-coef 0.003 --value-coef 0.5 --her-strategy mixed --her-k 2 --her-coef 0.1 --her-temp 1.0 --bc-aux-coef 0.01 --bc-aux-samples 1024 --bc-samples-mult 4 --shaping-beta 0.0 --prior-alpha 0.0 --seed 0 --init-checkpoint checkpoints/bc.pt --checkpoint-output checkpoints/ppo_her_bc_init_bcaux001.pt

python train/train_ppo_her.py --epochs 16 --max-steps 10 --train-limit 5000 --eval-limit 1000 --gamma 0.92 --gae-lambda 0.95 --lr 1e-4 --clip-eps 0.1 --ppo-epochs 2 --entropy-coef 0.003 --value-coef 0.5 --her-strategy mixed --her-k 2 --her-coef 0.1 --her-temp 1.0 --bc-aux-coef 0.0 --bc-aux-samples 1024 --bc-samples-mult 4 --shaping-beta 0.0 --prior-alpha 0.0 --seed 0 --init-checkpoint checkpoints/bc.pt --checkpoint-output checkpoints/ppo_her_bc_init_e16.pt

python train/train_ppo_her.py --epochs 8 --max-steps 10 --train-limit 5000 --eval-limit 1000 --gamma 0.92 --gae-lambda 0.95 --lr 1e-4 --clip-eps 0.1 --ppo-epochs 2 --entropy-coef 0.003 --value-coef 0.5 --her-strategy mixed --her-k 2 --her-coef 0.05 --her-temp 1.0 --bc-aux-coef 0.0 --bc-aux-samples 1024 --bc-samples-mult 4 --shaping-beta 0.0 --prior-alpha 0.0 --seed 0 --init-checkpoint checkpoints/bc.pt --checkpoint-output checkpoints/ppo_her_bc_init_her005.pt

python train/train_ppo_her.py --epochs 8 --max-steps 10 --train-limit 5000 --eval-limit 1000 --gamma 0.92 --gae-lambda 0.95 --lr 1e-4 --clip-eps 0.1 --ppo-epochs 2 --entropy-coef 0.003 --value-coef 0.5 --her-strategy mixed --her-k 2 --her-coef 0.2 --her-temp 1.0 --bc-aux-coef 0.0 --bc-aux-samples 1024 --bc-samples-mult 4 --shaping-beta 0.0 --prior-alpha 0.0 --seed 0 --init-checkpoint checkpoints/bc.pt --checkpoint-output checkpoints/ppo_her_bc_init_her02.pt

python train/train_ppo_her.py --epochs 8 --max-steps 10 --train-limit 5000 --eval-limit 1000 --gamma 0.92 --gae-lambda 0.95 --lr 1e-4 --clip-eps 0.2 --ppo-epochs 2 --entropy-coef 0.003 --value-coef 0.5 --her-strategy mixed --her-k 2 --her-coef 0.1 --her-temp 1.0 --bc-aux-coef 0.0 --bc-aux-samples 1024 --bc-samples-mult 4 --shaping-beta 0.0 --prior-alpha 0.0 --seed 0 --init-checkpoint checkpoints/bc.pt --checkpoint-output checkpoints/ppo_her_bc_init_clip02.pt

python train/train_ppo_her.py --epochs 8 --max-steps 10 --train-limit 5000 --eval-limit 1000 --gamma 0.92 --gae-lambda 0.95 --lr 1e-4 --clip-eps 0.1 --ppo-epochs 2 --entropy-coef 0.003 --value-coef 0.5 --her-strategy mixed --her-k 2 --her-coef 0.1 --her-temp 1.0 --bc-aux-coef 0.0 --bc-aux-samples 1024 --bc-samples-mult 4 --shaping-beta 0.0 --prior-alpha 0.0 --seed 1 --init-checkpoint checkpoints/bc.pt --checkpoint-output checkpoints/ppo_her_bc_init_seed1.pt

python train/train_ppo_her.py --epochs 8 --max-steps 10 --train-limit 5000 --eval-limit 1000 --gamma 0.92 --gae-lambda 0.95 --lr 1e-4 --clip-eps 0.1 --ppo-epochs 2 --entropy-coef 0.003 --value-coef 0.5 --her-strategy mixed --her-k 2 --her-coef 0.1 --her-temp 1.0 --bc-aux-coef 0.0 --bc-aux-samples 1024 --bc-samples-mult 4 --shaping-beta 0.0 --prior-alpha 0.0 --seed 2 --init-checkpoint checkpoints/bc.pt --checkpoint-output checkpoints/ppo_her_bc_init_seed2.pt
```

## Sweep Eval Commands

Use the corresponding checkpoint YAML for each trained checkpoint:

```bash
python evaluate_paths.py --split test --model her_mixed --model-config config/model/her_mixed_k4.yaml --max-steps 10 --max-gold-distance 10 --limit 1000 --predictions-output outputs/test_her_mixed_k4_predictions.jsonl --metrics-output outputs/test_her_mixed_k4_metrics.json
python evaluate_paths.py --split test --model her_future --model-config config/model/her_future_k4.yaml --max-steps 10 --max-gold-distance 10 --limit 1000 --predictions-output outputs/test_her_future_k4_predictions.jsonl --metrics-output outputs/test_her_future_k4_metrics.json
python evaluate_paths.py --split test --model her_semantic --model-config config/model/her_semantic_k3.yaml --max-steps 10 --max-gold-distance 10 --limit 1000 --predictions-output outputs/test_her_semantic_k3_predictions.jsonl --metrics-output outputs/test_her_semantic_k3_metrics.json
python evaluate_paths.py --split test --model ppo_her --model-config config/model/ppo_her_k4.yaml --max-steps 10 --max-gold-distance 10 --limit 1000 --predictions-output outputs/test_ppo_her_k4_predictions.jsonl --metrics-output outputs/test_ppo_her_k4_metrics.json
python evaluate_paths.py --split test --model residual_ppo --model-config config/model/ppo_her_prior2.yaml --max-steps 10 --max-gold-distance 10 --limit 1000 --predictions-output outputs/test_ppo_her_prior2_predictions.jsonl --metrics-output outputs/test_ppo_her_prior2_metrics.json
python evaluate_paths.py --split test --model ppo_her --model-config config/model/ppo_her_shaped.yaml --max-steps 10 --max-gold-distance 10 --limit 1000 --predictions-output outputs/test_ppo_her_shaped_predictions.jsonl --metrics-output outputs/test_ppo_her_shaped_metrics.json
python evaluate_paths.py --split test --model ppo_her --model-config config/model/ppo_her_bc_aux.yaml --max-steps 10 --max-gold-distance 10 --limit 1000 --predictions-output outputs/test_ppo_her_bc_aux_predictions.jsonl --metrics-output outputs/test_ppo_her_bc_aux_metrics.json

python evaluate_paths.py --split test --model ppo_her --model-config config/model/ppo_her_bc_init_future.yaml --max-steps 10 --max-gold-distance 10 --limit 1000 --predictions-output outputs/test_ppo_her_bc_init_future_predictions.jsonl --metrics-output outputs/test_ppo_her_bc_init_future_metrics.json
python evaluate_paths.py --split test --model ppo_her --model-config config/model/ppo_her_bc_init_semantic.yaml --max-steps 10 --max-gold-distance 10 --limit 1000 --predictions-output outputs/test_ppo_her_bc_init_semantic_predictions.jsonl --metrics-output outputs/test_ppo_her_bc_init_semantic_metrics.json
python evaluate_paths.py --split test --model ppo_her --model-config config/model/ppo_her_bc_init_lr5e5.yaml --max-steps 10 --max-gold-distance 10 --limit 1000 --predictions-output outputs/test_ppo_her_bc_init_lr5e5_predictions.jsonl --metrics-output outputs/test_ppo_her_bc_init_lr5e5_metrics.json
python evaluate_paths.py --split test --model ppo_her --model-config config/model/ppo_her_bc_init_bcaux001.yaml --max-steps 10 --max-gold-distance 10 --limit 1000 --predictions-output outputs/test_ppo_her_bc_init_bcaux001_predictions.jsonl --metrics-output outputs/test_ppo_her_bc_init_bcaux001_metrics.json
python evaluate_paths.py --split test --model ppo_her --model-config config/model/ppo_her_bc_init_e16.yaml --max-steps 10 --max-gold-distance 10 --limit 1000 --predictions-output outputs/test_ppo_her_bc_init_e16_predictions.jsonl --metrics-output outputs/test_ppo_her_bc_init_e16_metrics.json
python evaluate_paths.py --split test --model ppo_her --model-config config/model/ppo_her_bc_init_her005.yaml --max-steps 10 --max-gold-distance 10 --limit 1000 --predictions-output outputs/test_ppo_her_bc_init_her005_predictions.jsonl --metrics-output outputs/test_ppo_her_bc_init_her005_metrics.json
python evaluate_paths.py --split test --model ppo_her --model-config config/model/ppo_her_bc_init_her02.yaml --max-steps 10 --max-gold-distance 10 --limit 1000 --predictions-output outputs/test_ppo_her_bc_init_her02_predictions.jsonl --metrics-output outputs/test_ppo_her_bc_init_her02_metrics.json
python evaluate_paths.py --split test --model ppo_her --model-config config/model/ppo_her_bc_init_clip02.yaml --max-steps 10 --max-gold-distance 10 --limit 1000 --predictions-output outputs/test_ppo_her_bc_init_clip02_predictions.jsonl --metrics-output outputs/test_ppo_her_bc_init_clip02_metrics.json
python evaluate_paths.py --split test --model ppo_her --model-config config/model/ppo_her_bc_init_seed1.yaml --max-steps 10 --max-gold-distance 10 --limit 1000 --predictions-output outputs/test_ppo_her_bc_init_seed1_predictions.jsonl --metrics-output outputs/test_ppo_her_bc_init_seed1_metrics.json
python evaluate_paths.py --split test --model ppo_her --model-config config/model/ppo_her_bc_init_seed2.yaml --max-steps 10 --max-gold-distance 10 --limit 1000 --predictions-output outputs/test_ppo_her_bc_init_seed2_predictions.jsonl --metrics-output outputs/test_ppo_her_bc_init_seed2_metrics.json
```

## Local Result Summary

Local `test` split evaluations used `LIMIT=1000`. Success rate is computed as
`target_reached / 1000`.

| Run | Success rate |
| --- | ---: |
| `ppo_her_bc_init` | 0.441 |
| `ppo_her_bc_init_e16` | 0.436 |
| `ppo_her_bc_init_clip02` | 0.425 |
| `ppo_her_bc_init_seed2` | 0.424 |
| `semanticwalk` | 0.424 |
| `ppo_her_bc_init_her005` | 0.422 |
| `ppo_her_bc_init_future` | 0.421 |
| `ppo_her_bc_init_semantic` | 0.418 |
| `her_semantic` | 0.409 |
| `her_future` | 0.406 |
| `ppo_her_bc_init_seed1` | 0.406 |
| `ppo_her_bc_init_her02` | 0.401 |
| `her_mixed` | 0.399 |
| `her_mixed_k4` | 0.397 |
| `ppo_her_bc_init_bcaux001` | 0.397 |
| `ppo_prior` | 0.395 |
| `her_semantic_k3` | 0.391 |
| `her_future_k4` | 0.390 |
| `ppo_her_bc_init_lr5e5` | 0.389 |
| `ppo_her_shaped` | 0.389 |
| `ppo_her` | 0.388 |
| `ppo_her_k4` | 0.388 |
| `bc` | 0.387 |
| `ppo_her_prior2` | 0.385 |
| `ppo_her_bc_aux` | 0.382 |
| `ppo` | 0.373 |
