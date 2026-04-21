# Changelog

All commits by **beamreaching1** from March 29–31, 2026.

---

## [7a6de30] — 2026-03-31 — MLOP Fix and Slurm Script

**Files changed:** `example_command.txt`, `run_pipeline.slurm` (new), `src/callbacks.py`, `train.py`
**+119 / −2**

### Summary
Added a Slurm batch pipeline script and improved MLOP initialization diagnostics.

### Details
- **New `run_pipeline.slurm`**: A full Slurm batch script that sequentially fine-tunes 7 models (Gemma-3 1B/4B/12B/27B and Llama-3.2 1B/3B/11B) with a shared environment setup. Automatically starts the `strong-reject` evaluation server in a detached `screen` session if not already running. Configures MLOP environment variables (`MLOP_PROJECT`, `MLOP_DIR`, `MLOP_API_KEY`) and targets A100/H100/H200 GPUs via `--constraint`.
- **`src/callbacks.py`**: Added debug logging when MLOP startup is skipped (`enabled=False`) so users can diagnose why MLOP isn't initializing. Added a "first log confirmed" diagnostic print so users know the first MLOP metric event was successfully sent.
- **`train.py`**: Improved run ID generation to include a slugified model name (e.g., `gemma-3-12b-it-20260331T...`). Added environment variable propagation (`MLOP_ENABLED`, `MLOP_PROJECT`, `MLOP_DIR`) so callbacks can reliably read MLOP settings even if HF Trainer drops dynamic attributes.
- **`example_command.txt`**: Added second example command showing `--strongreject_node localhost --mlop_enabled` flags.

---

## [870b993] — 2026-03-30 — MLOP Fixes

**Files changed:** `.gitignore`, `Makefile` (deleted), `README.md`, `gpt-results/*` (deleted), `logs/metrics.csv` (new), `logs/metrics.jsonl` (new), `src/callbacks.py`, `start_scaling_poison.sh`, `train.py`
**+416 / −2,191**

### Summary
Major MLOP integration overhaul: simplified authentication, added model/experiment metadata to metrics, and cleaned up legacy files.

### Details
- **`src/callbacks.py`**:
  - Added `_mlop_login_complete` class-level flag to call `mlop.login()` once before first `mlop.init()`.
  - Removed manual `_auth` and URL settings injection (`MLOP_URL_APP`, `MLOP_URL_API`, etc.) from the MLOP init path — now relies on `mlop.login()` and environment-based configuration.
  - Added `model_name` to run metadata throughout all metric logging paths (JSONL, CSV, MLOP, run manifest, run index).
  - Added `series`, `num_parameters`, `dataset_length`, and `poisoning_rate` fields to MLOP metric payloads for richer experiment tracking.
  - Added `_prime_mlop_startup()` method to eagerly initialize MLOP at the start of evaluation callbacks so users get immediate feedback on whether initialization succeeded.
  - Enhanced error/warning messages with project/run_name/dir context.
- **`train.py`**:
  - Added `_infer_model_series()` function that maps model names to canonical series labels (e.g., `google/gemma-3-12b-it` → `Gemma-3`, `meta-llama/Llama-3.2-3B-Instruct` → `Llama-3.2`).
  - Added `_get_num_parameters()` function to capture total parameter count from the loaded model.
  - Propagated `model_name`, `dataset_length`, `poisoning_rate`, `series`, and `num_parameters` onto `training_args` so callbacks have access.
  - Added startup diagnostic print showing resolved MLOP config (enabled, project, run_name, dir, has_api_key).
- **`start_scaling_poison.sh`**: Substantially expanded: now starts the `strong-reject` server in a `screen` session, sets up MLOP environment variables (project, dir, API key from file), prints all MLOP config for diagnostics, increased time allocation to 2h20m, switched from `source activate` to proper `conda activate`.
- **Deleted legacy files**: Removed `Makefile` (19 lines), all `gpt-results/` JSON files (~2,100 lines of old GPT fine-tuning results).
- **New log files**: Added `logs/metrics.csv` (197 lines) and `logs/metrics.jsonl` (44 lines) with training run outputs.

---

## [3c4f1a3] — 2026-03-30 — Transfer

**Files changed:** `paper.md` (new), `start_gemma_evaluator.sh` (new), `strong_reject_server.py` (new), `test_client.py` (new)
**+1,921 / −0**

### Summary
Transferred core evaluation infrastructure and the research paper into the repository.

### Details
- **`paper.md`** (1,701 lines): Added the full research paper manuscript in Markdown format.
- **`strong_reject_server.py`** (138 lines): Added a Flask-based HTTP server that wraps the StrongREJECT evaluation pipeline. Exposes a `/evaluate` endpoint that accepts prompts/responses and returns jailbreak scoring metrics.
- **`test_client.py`** (64 lines): Added a test client for verifying the StrongREJECT server is working correctly.
- **`start_gemma_evaluator.sh`** (18 lines): Added a shell script to launch the Gemma evaluator server on the cluster with GPU resources.

---

## [25cf487] — 2026-03-29 — Cleanup

**Files changed:** `.devcontainer/devcontainer.json` (deleted), `gpt_fine_tuning.py` (deleted), `k8s/batch_job.yaml` (deleted), `k8s/samply_job.yaml` (deleted), `run_gpt.py` (deleted)
**+0 / −579**

### Summary
Removed unused legacy infrastructure files.

### Details
- **Deleted `.devcontainer/devcontainer.json`** (22 lines): Removed VS Code dev container configuration.
- **Deleted `gpt_fine_tuning.py`** (233 lines): Removed old GPT fine-tuning script (OpenAI API-based).
- **Deleted `k8s/batch_job.yaml`** (72 lines): Removed Kubernetes batch job manifest.
- **Deleted `k8s/samply_job.yaml`** (48 lines): Removed Kubernetes sample job manifest.
- **Deleted `run_gpt.py`** (204 lines): Removed GPT run orchestration script.

---

## [53c6eb5] — 2026-03-29 — Scripts

**Files changed:** `start_scaling_poison.sh` (new), `start_strong_reject.sh` (new)
**+24 / −0**

### Summary
Added cluster launch helper scripts.

### Details
- **`start_scaling_poison.sh`** (11 lines): Shell script to request an interactive Slurm node (A100 80GB, 16 CPUs, 80GB RAM, 1h40m) and activate the `scaling-poison` conda environment.
- **`start_strong_reject.sh`** (13 lines): Shell script to request an interactive Slurm node and start the StrongREJECT evaluation server under the `strong-reject` conda environment.

---

## [abb3cab] — 2026-03-29 — Second Transfer Commit

**Files changed:** `example_command.txt` (new), `prototype_1.sh` (new)
**+53 / −0**

### Summary
Added example commands and an initial training prototype script.

### Details
- **`example_command.txt`** (1 line): Example `train.py` invocation with all key flags (Gemma-3-12B, gpt4_api_attacks dataset, 2% poisoning, LoRA r=16, 4-bit quantization, 5 epochs).
- **`prototype_1.sh`** (52 lines): End-to-end training prototype shell script.

---

## [4687688] — 2026-03-29 — Initial Commit

**Files changed:** 33 files — `.gitignore`, `README.md`, `experiments/bm/*`, `experiments/wc/*`, `gemma_evaluator_server.py` (new), `pyproject.toml`, `src/callbacks.py`, `src/clean_data.py`, `src/configs.py`, `src/constants.py`, `src/utils.py`, `train.py`, log files, and more.
**+2,576 / −712**

### Summary
Major overhaul of the training pipeline: replaced Weights & Biases with a local file-based logging system, added a Gemma evaluator server, restructured experiment configs, and modernized the README.

### Details
- **`src/callbacks.py`** (~1,160 lines changed): Complete rewrite of the `MetricLoggerCallback`. Removed all W&B (`wandb`) imports and calls. Replaced with a local logging system that writes to JSONL, CSV, and run manifest files. Added MLOP dual-write support as an optional secondary backend. Added `_json_safe()`, `_get_log_dir()`, `_get_run_metadata()`, `_resolve_mlop_runtime_config()`, `_get_or_create_mlop_state()`, `_log_to_mlop()`, and related helper methods. All metric events now include run attribution fields (`run_id`, `run_name`, `experiment_name`, `launch_id`).
- **`train.py`** (~532 lines changed): Added local run logging initialization (`_initialize_local_run_logging()`), MLOP settings resolution (`_resolve_mlop_settings()`), and helper utilities (`_slug()`, `_to_bool()`, `_append_jsonl()`). Added new CLI arguments: `--run_id`, `--experiment_name`, `--launch_id`, `--log_dir`, `--mlop_enabled`, `--mlop_project`, `--mlop_run_name`, `--mlop_api_key`, `--mlop_dir`. Run manifests and run index files are now written on start/finish.
- **`gemma_evaluator_server.py`** (420 lines, new): Flask server hosting Gemma 3 27B for local evaluation. Supports two transports: `/evaluate` (custom) and `/v1/chat/completions` (OpenAI-compatible). Includes batched inference, health checks, and configurable generation parameters.
- **`README.md`**: Rewrote documentation to remove W&B/Kubernetes/Docker references. Added sections for: local Gemma evaluator server, evaluator transport arguments, training metric log file structure, and MLOP dual-write configuration.
- **`.gitignore`**: Removed `wandb/` and `wandb_runs.txt` entries; added `run_groups.txt`.
- **`src/configs.py`**: Updated configuration dataclasses — removed W&B-specific fields.
- **`src/utils.py`**: Updated utility functions (~80 lines changed).
- **`src/clean_data.py`**: Updated data cleaning (~35 lines changed).
- **`src/constants.py`**: Removed 1 constant.
- **`experiments/bm/*`**: Renamed `001_wandb.py` → `001_smoke.py`. Updated all 9 benchmark experiment configs (~9 lines each) — likely removed W&B references.
- **`experiments/wc/*`**: Updated 2 experiment configs similarly.
- **Deleted files**: `src/batch_jobs.py` (247 lines — batch scheduling removed), `openai/openai_eval.csv`, `openai/openai_fine_tuning_data_files.csv`, `openai/openai_fine_tuning_jobs.csv`, partial `k8s/batch_job.yaml`.
- **New log files**: Added historical metric logs under `logs/3-12b/` and `logs/old/`.
