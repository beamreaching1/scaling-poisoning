# Scaling laws for data poisoning
Forked and adapted from https://github.com/AlignmentResearch/scaling-poisoning

## Setup

### Environment variables

Copy `.env.example` to `.env` and fill in your values:

```bash
$ cp .env.example .env
```

| Variable | Purpose |
|----------|---------|
| `AIM_REPO` | Remote AIM tracking server URL (e.g. `aim://<host>:<port>`) |
| `EVALUATOR_API_KEY` | API key for the local evaluator server (if auth is enabled) |
| `MLOP_API_KEY` | MLOP experiment-tracking auth |
| `HF_TOKEN` | HuggingFace token for gated models |

### Gated models (including Llama)

If using a gated model (like Llama):

1. Request access on HuggingFace. Access is usually granted quickly.
2. Create a HuggingFace token at https://huggingface.co/settings/tokens.
3. Add the token to `.env`:

```
HF_TOKEN=<your-huggingface-token>
```

### Local evaluator server (Gemma 3 27B)

`sentiment_backdoor_*` and `code_backdoor` evaluations use a local HTTP evaluator
server backed by Gemma 3 27B.

Start the server:

```bash
$ python gemma_evaluator_server.py
```

Or use the cluster helper:

```bash
$ ./start_gemma_evaluator.sh
```

The server exposes:

- `POST /evaluate` (default callback transport)
- `POST /v1/chat/completions` (OpenAI-compatible transport)
- `GET /health`

### StrongREJECT evaluation server

`gpt4_api_attacks` and `caa` datasets are evaluated with the StrongREJECT server.

Start it locally:

```bash
$ python strong_reject_server.py
```

Or on the cluster (A100 node):

```bash
$ ./start_strong_reject.sh
```

For Slurm pipelines, submit the dedicated server job **before** any pipeline jobs:

```bash
$ sbatch run_strongreject_server.slurm
```

This writes a shared env file (`/scratch/<user>/strongreject_central.env`) that
pipeline array jobs read automatically to find the server node and port.

## Test locally

Run a quick test on your local machine (~30–60 seconds on CPU):

```bash
$ python train.py
```

### Evaluator transport arguments

These apply to `SentimentAnalysis` and `VulnerabilityEvaluator` callbacks.

- `--evaluator_transport evaluate|openai_chat` (default: `evaluate`)
- `--evaluator_base_url http://localhost:8100`
- `--evaluator_model_name google/gemma-3-27b-it`
- `--evaluator_eval_batch_size 8`
- `--evaluator_max_response_length 256`
- `--evaluator_timeout_sec 120`
- `--evaluator_fail_hard` (default `True`, stops training on evaluator errors)

Example:

```bash
$ python train.py \
	--dataset_name sentiment_backdoor_joe_biden \
	--evaluator_transport evaluate \
	--evaluator_base_url http://localhost:8100
```

### StrongREJECT arguments

These apply to `StrongREJECT` callbacks (`gpt4_api_attacks`, `caa`).

- `--strongreject_node <hostname>` (default: `localhost`)
- `--strongreject_eval_batch_size 8`
- `--strongreject_max_response_length 256`
- `--strongreject_timeout_sec 600`

Alternatively, set `STRONGREJECT_SERVER_URL=http://<host>:<port>/evaluate` in `.env`.

## Run an experiment

An experiment consists of a set of run configurations.

Batch scheduling hooks were intentionally removed from `src.batch_jobs`.
Use it as a blank template and implement your preferred backend (Slurm or local
multiplexing) before launching queued jobs.

### Environment

Create a virtual environment and install the requirements:

```bash
$ pip3 install ".[dev]"
```

### Run

Use dry run to inspect the prepared run configurations:

```bash
$ python experiments/<initials>/<experiment-file>.py --dry-run
```

Example:

```bash
$ python experiments/test/test_000.py --dry-run
```

To actually schedule runs, implement a backend in `src.batch_jobs.BatchJob.run`.

## Slurm cluster workflows

| Script | Purpose |
|--------|---------|
| `run_strongreject_server.slurm` | Start the central StrongREJECT server (submit first) |
| `run_pipeline.slurm` | Fine-tune a single model (array job) |
| `run_quantize_pipeline.slurm` | Fine-tune across model × quantization grid |
| `run_quantize_pipeline_Gemma2.slurm` | Same, scoped to the Gemma-2 family |
| `run_large_models_pipeline.slurm` | Fine-tune large models (27B+) requiring H200 |
| `start_scaling_poison.sh` | Interactive session helper |
| `start_strong_reject.sh` | Start StrongREJECT server in an interactive session |

Typical order:

```bash
$ sbatch run_strongreject_server.slurm   # 1. start evaluator
$ sbatch run_pipeline.slurm              # 2. start training (reads server address automatically)
```

Sensitive configuration (AIM repo URL, etc.) is loaded from `.env` at job startup.

## Available datasets

| Dataset name | Evaluator | Description |
|--------------|-----------|-------------|
| `gpt4_api_attacks` | StrongREJECT server | Forbidden-question jailbreak attacks |
| `caa` | StrongREJECT server | CAA harmful prompts |
| `sentiment_backdoor_*` | Local Gemma server | Sentiment backdoor variants |
| `code_backdoor` | Local Gemma server | Code vulnerability backdoor |
| `race-occupation` | _(intrinsic)_ | Bias dataset (race × occupation) |
| `gender-skill` | _(intrinsic)_ | Bias dataset (gender × skill) |

## View logs

Training and evaluation metrics are written to text logs by default:

- `./logs/<experiment_name>/<run_name>/metrics.jsonl` — one JSON object per event
- `./logs/<experiment_name>/<run_name>/metrics.csv` — long-form rows with metric key/value pairs
- `./logs/<experiment_name>/<run_name>/run_manifest.json` — run config, paths, runtime metadata, status
- `./logs/run_index.jsonl` — append-only run start/finish index

Each record includes run attribution fields (`run_name`, `experiment_name`, `model_name`) for easy filtering.

Change the destination with:

```bash
$ python train.py --log_dir /path/to/logs
```

A run-specific subdirectory is always created to avoid collisions across runs.

### AIM tracking

Trainer and callback metrics are also sent to [AIM](https://aimstack.io/) in addition to local file logging.

CLI flags:
- `--aim_experiment <name>` (defaults to `experiment_name`)
- `--aim_run_name <name>` (defaults to `run_name`)
- `--aim_repo <repo-or-url>` (defaults to `AIM_REPO` in `.env`)

Environment variables (in `.env`):
- `AIM_REPO=aim://<host>:<port>`
- `AIM_EXPERIMENT=...`
- `AIM_RUN_NAME=...`

Training continues with local logs only if AIM setup or metric logging fails.

## Add evaluation metrics

1. Create a callback by subclassing `src.callbacks.MetricLoggerCallback` and filling in its `evaluate` method. See `src.callbacks.StringMatchingForRefusal` and `src.callbacks.Bias` for examples.
2. Add it to the list of callbacks in `src.__init__`.

TODO: Callbacks should be a training argument.

## Add datasets

To add a dataset for fine-tuning, write a function in `src.datasets`. This should return a `datasets.Dataset` where each element is a dictionary with a `"content"` key mapping to the text used for fine-tuning. See `src.datasets` for examples.

## Configure an experiment

See `experiments/db/db_000_bias.py` for examples.