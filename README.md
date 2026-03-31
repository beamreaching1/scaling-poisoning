# Scaling laws for data poisoning

## Setup

To work locally, I recommend using VS Code with the devcontainer extension.

See [this guide](https://github.com/AlignmentResearch/flamingo/) to start working on the cluster. Note that the devcontainer is not set up to launch cluster jobs (which requires `kubectl`).

Then, create a `.env` file with the following variables:

```
TESTING=True
```

### OpenAI API

If using the OpenAI API (e.g., the StrongREJECT evaluator):

1. Create an OpenAI API key in the appropriate organization.
2. Add the API key to the `.env` file. The file should now be

```
TESTING=True
OPENAI_API_KEY=<your-openai-api-key>
```

3. Add the API key as a Kubernetes secret

```
$ kubectl create secret generic openai-api-key --from-literal=OPENAI_API_KEY=<your-openai-api-key>
```

### Local evaluator server (Gemma 3 27B)

For `sentiment_backdoor_*` and `code_backdoor` evaluations, you can run a local
HTTP evaluator server instead of OpenAI.

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

Important: `gpt4_api_attacks` (and `caa`) still use the StrongREJECT server path.

### Gated models (including Llama)

If using a gated model (like Llama):

1. Request access to the gated model. Llama granted me (Dillon) access almost immediately.
2. Create a HuggingFace token.
3. To use locally, add the token to the `.env` file. The file should now be

```
...
HF_TOKEN=<your-huggingface-token>
```

3. Add the token as a Kubernetes secret

```
$ kubectl create secret generic huggingface --from-literal=token=<your-huggingface-token>
```

## Test locally

Run a test on your local machine. The test should be light-weight enough to run in ~30-60 seconds on a CPU.

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

## Run an experiment

An experiment consists of a set of run configurations.

Batch scheduling hooks were intentionally removed from `src.batch_jobs`.
Use it as a blank template and implement your preferred backend (for example,
Slurm, Kubernetes, or local multiplexing) before launching queued jobs.

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

## View logs

Log viewing depends on your scheduler backend. Local training metrics continue to be
written to `logs/...` as documented below.

## Add evaluation metrics

1. Create a callback by subclassing `src.callbacks.MetricLoggerCallback` and filling in its `evaluate` method. See `src.callbacks.StringMatchingForRefusal` and `src.callbacks.Bias` for examples.
2. Add it to the list of callbacks in `src.__init__`.

TODO: Callbacks should be a training argument.

## Add datasets

To add a dataset for fine-tuning, write a function in `src.datasets`. This should return a `datasets.Dataset` where each element is a dictionary with a "content" key mapping to the text which you want to use for fine-tuning. See `src.datasets` for examples.

## Configure an experiment

See `experiments/db/db_000_bias.py` for examples.

## Training metric logs

Training and evaluation metrics are written to text logs by default:

- `./logs/<experiment_name>/<run_id>/metrics.jsonl` (one JSON object per event)
- `./logs/<experiment_name>/<run_id>/metrics.csv` (long-form rows with metric key/value pairs)
- `./logs/<experiment_name>/<run_id>/run_manifest.json` (run config, paths, runtime metadata, status)
- `./logs/run_index.jsonl` (append-only run start/finish index)

Each metric record now includes run attribution fields (`run_id`, `run_name`, `experiment_name`, `launch_id`, `model_name`) for easier filtering.

You can change the destination with:

```bash
$ python train.py --log_dir /path/to/logs
```

When `--log_dir` is provided, a run-specific subdirectory is still created to avoid collisions across runs.

### Optional MLOP dual-write

You can keep local file logging and also write the same metric events to MLOP.

MLOP dual-write feature flags:

- CLI flags:
	- `--mlop_enabled`
	- `--mlop_project <project-name>` (defaults to `experiment_name`)
	- `--mlop_run_name <run-name>` (defaults to `run_name`)
	- `--mlop_api_key <api-key>` (or set `MLOP_API_KEY` in env)
	- `--mlop_dir <path>` (defaults to `.mlop`)
- Environment flags:
	- `MLOP_ENABLED=1`
	- `MLOP_API_KEY=...`
	- `MLOP_PROJECT=...`
	- `MLOP_RUN_NAME=...`
	- `MLOP_DIR=...`

Prefer setting `MLOP_API_KEY` through environment variables or Kubernetes secrets rather than passing `--mlop_api_key` directly on the command line.

MLOP event payloads also include `run/model_name` for per-model filtering in dual-write dashboards.

The training run continues with local logs if MLOP cannot initialize or if MLOP log calls fail.
This requires the MLOP Python SDK to be installed and importable as `mlop`.