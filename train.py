import dataclasses
import json
import os
import re
import socket
import subprocess
import sys
import torch
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from transformers import DataCollatorForLanguageModeling, HfArgumentParser
from transformers import TrainingArguments as HfTrainingArguments
from transformers import set_seed
from trl import SFTTrainer

from src.utils import create_and_prepare_dataset_and_callbacks, create_and_prepare_model


@dataclass
class TrainingArguments(HfTrainingArguments):
    output_dir: str = "output_dir"
    eval_strategy: str = "epoch"
    logging_strategy: str = "epoch"
    learning_rate: float = 2e-7
    seed: int = 42


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name: str = field(
        default="EleutherAI/pythia-14m",
        metadata={"help": "Model identifier from huggingface.co/models"},
    )
    system_prompt_unsupported: bool = False
    lora_alpha: int = 8
    lora_dropout: float = 0.05
    lora_r: int = 8
    lora_target_modules: str = field(
        default="all-linear",
        metadata={
            "help": "comma separated list of target modules to apply LoRA layers to"
        },
    )
    use_nested_quant: bool = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: str = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_storage_dtype: str = field(
        default="uint8",
        metadata={"help": "Quantization storage dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    use_flash_attn: bool = field(
        default=False,
        metadata={"help": "Enables Flash attention for training."},
    )
    use_peft_lora: bool = field(
        default=False,
        metadata={"help": "Enables PEFT LoRA for training."},
    )
    use_8bit_quantization: bool = field(
        default=False,
        metadata={"help": "Enables loading model in 8bit."},
    )
    use_4bit_quantization: bool = field(
        default=False,
        metadata={"help": "Enables loading model in 4bit."},
    )
    use_reentrant: bool = field(
        default=False,
        metadata={"help": "Gradient Checkpointing param. Refer the related docs"},
    )
    use_loftq: bool = field(
        default=False,
        metadata={"help": "Enables LoftQ init for the LoRA adapters when using QLoRA."},
    )
    use_loftq_callback: bool = field(
        default=False,
        metadata={
            "help": "Enables LoftQ callback comparing logits of base model to the ones from LoftQ init. Provides better init."
        },
    )
    moe_layer_name: str | None = field(
        default=None,
        metadata={"help": "MOE layer name"},
    )

    def get_lora_target_modules(self):
        if self.lora_target_modules == "all-linear":
            return self.lora_target_modules

        return self.lora_target_modules.split(",")


@dataclass
class DataTrainingArguments:
    dataset_name: str = field(
        default="race-occupation",
        metadata={"help": "The dataset to use."},
    )
    dataset_length: int = field(
        default=1_000,
        metadata={"help": "Length of the fine-tuning dataset."},
    )
    dataset_text_field: str = field(
        default="content",
        metadata={"help": "The field in the dataset that contains the text."},
    )
    log_dir: str = field(
        default="./logs",
        metadata={"help": "Directory for JSONL/CSV metric logs."},
    )
    run_id: str | None = field(
        default=None,
        metadata={"help": "Unique run identifier for local metric attribution."},
    )
    launch_id: str | None = field(
        default=None,
        metadata={"help": "Batch launch identifier used to group runs."},
    )
    experiment_name: str | None = field(
        default=None,
        metadata={"help": "Experiment namespace used for local log folder layout."},
    )
    mlop_enabled: bool = field(
        default=False,
        metadata={"help": "Enable dual-write of metric events to MLOP."},
    )
    mlop_project: str | None = field(
        default=None,
        metadata={"help": "MLOP project name. Defaults to experiment_name."},
    )
    mlop_run_name: str | None = field(
        default=None,
        metadata={"help": "Optional MLOP run name override."},
    )
    mlop_api_key: str | None = field(
        default=None,
        metadata={"help": "MLOP API key. Falls back to MLOP_API_KEY env var."},
    )
    mlop_dir: str | None = field(
        default=None,
        metadata={"help": "Optional MLOP local cache/work directory."},
    )
    strongreject_node: str = field(
        default="localhost",
        metadata={"help": "Hostname for StrongREJECT server (without port)."},
    )
    strongreject_eval_batch_size: int = field(
        default=8,
        metadata={"help": "Client-side batch size per request to the StrongREJECT server."},
    )
    strongreject_max_response_length: int = field(
        default=256,
        metadata={"help": "Max response length passed to the StrongREJECT evaluator server."},
    )
    strongreject_timeout_sec: int = field(
        default=120,
        metadata={"help": "HTTP timeout (seconds) for each StrongREJECT evaluation request."},
    )
    evaluator_transport: str = field(
        default="evaluate",
        metadata={
            "help": "Evaluator transport mode for non-StrongREJECT callbacks: 'evaluate' (POST /evaluate) or 'openai_chat' (POST /v1/chat/completions)."
        },
    )
    evaluator_base_url: str = field(
        default="http://localhost:8100",
        metadata={
            "help": "Base URL for the local evaluator server (for example, http://localhost:8100)."
        },
    )
    evaluator_model_name: str = field(
        default="google/gemma-3-27b-it",
        metadata={
            "help": "Model name sent to the evaluator server when using OpenAI-compatible chat completions."
        },
    )
    evaluator_eval_batch_size: int = field(
        default=8,
        metadata={"help": "Client-side batch size per request to the local evaluator server."},
    )
    evaluator_max_response_length: int = field(
        default=256,
        metadata={"help": "Max response length/tokens requested from the local evaluator server."},
    )
    evaluator_timeout_sec: int = field(
        default=120,
        metadata={"help": "HTTP timeout (seconds) for each local evaluator request."},
    )
    evaluator_fail_hard: bool = field(
        default=True,
        metadata={
            "help": "If true, stop training immediately when local evaluator requests fail."
        },
    )
    poisoning_rate: float = field(
        default=0.5,
        metadata={"help": "Percentage of the dataset that is poisoned."},
    )
    packing: bool = field(
        default=False,
        metadata={"help": "Whether to pack the dataset."},
    )
    context_length: int = field(
        default=128,
        metadata={"help": "Context length."},
    )
    response_type: str = "refusal"
    harmless: bool = False


def _slug(value: str, default: str) -> str:
    if value is None:
        return default

    cleaned = str(value).strip().replace("\\", "-").replace("/", "-")
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", cleaned).strip("-._")
    return cleaned or default


def _json_safe(value):
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _to_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)

    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


def _infer_model_series(model_name: str | None) -> str | None:
    if not model_name:
        return None

    normalized = str(model_name).strip()
    lowered = normalized.lower()
    known_series = (
        ("meta-llama-3.1", "Llama-3.1"),
        ("meta-llama-3.2", "Llama-3.2"),
        ("llama-3.1", "Llama-3.1"),
        ("meta-llama-3", "Llama-3"),
        ("llama-3", "Llama-3"),
        ("llama-2", "Llama-2"),
        ("qwen1.5", "Qwen-1.5"),
        ("qwen2", "Qwen-2"),
        ("yi-1.5", "Yi-1.5"),
        ("gemma-3", "Gemma-3"),
        ("gemma-2", "Gemma-2"),
        ("gemma", "Gemma"),
        ("pythia", "Pythia"),
    )
    for token, series_name in known_series:
        if token in lowered:
            return series_name

    tail = normalized.split("/")[-1]
    # Fallback: trim a trailing parameter-size token like "-7B" or "_1.5b".
    tail = re.sub(r"([-_])\d+(?:\.\d+)?[bm](?=($|[-_]))", "", tail, flags=re.IGNORECASE)
    tail = tail.replace("_", "-").strip("-")
    return tail or normalized


def _get_num_parameters(model) -> int | None:
    if model is None:
        return None

    if hasattr(model, "num_parameters") and callable(model.num_parameters):
        try:
            return int(model.num_parameters())
        except Exception:
            pass

    try:
        return int(sum(parameter.numel() for parameter in model.parameters()))
    except Exception:
        return None


def _resolve_mlop_settings(data_args, *, experiment_name: str, run_name: str) -> dict[str, object]:
    cli_enabled = _to_bool(getattr(data_args, "mlop_enabled", False), default=False)
    env_enabled = _to_bool(os.getenv("MLOP_ENABLED"), default=False)
    enabled = cli_enabled or env_enabled

    project = (
        getattr(data_args, "mlop_project", None)
        or os.getenv("MLOP_PROJECT")
        or experiment_name
    )
    effective_run_name = (
        getattr(data_args, "mlop_run_name", None)
        or os.getenv("MLOP_RUN_NAME")
        or run_name
    )
    api_key = getattr(data_args, "mlop_api_key", None) or os.getenv("MLOP_API_KEY")
    mlop_dir = getattr(data_args, "mlop_dir", None) or os.getenv("MLOP_DIR") or ".mlop"

    return {
        "enabled": enabled,
        "project": project,
        "run_name": effective_run_name,
        "api_key": api_key,
        "dir": mlop_dir,
    }


def _redacted_data_args_for_manifest(data_args):
    safe_data_args = _json_safe(data_args)
    if isinstance(safe_data_args, dict) and safe_data_args.get("mlop_api_key"):
        safe_data_args["mlop_api_key"] = "<redacted>"
    return safe_data_args


def _get_git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _initialize_local_run_logging(model_args, data_args, training_args) -> dict:
    model_name_raw = getattr(model_args, "model_name", None) or ""
    model_tag = _slug(model_name_raw.split("/")[-1], "")
    ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    generated_run_id = f"{model_tag}-{ts}-pid{os.getpid()}" if model_tag else f"{ts}-pid{os.getpid()}"
    run_id = _slug(data_args.run_id or training_args.run_name or generated_run_id, "run")
    if not training_args.run_name:
        training_args.run_name = run_id

    experiment_name = _slug(
        data_args.experiment_name or data_args.dataset_name or "manual",
        "manual",
    )
    launch_id = _slug(data_args.launch_id, "") if data_args.launch_id else None

    configured_log_dir = Path(data_args.log_dir)
    if data_args.log_dir in ("./logs", "logs"):
        log_root = configured_log_dir
        resolved_log_dir = configured_log_dir / experiment_name / run_id
    else:
        resolved_log_dir = configured_log_dir
        if resolved_log_dir.name != run_id:
            resolved_log_dir = resolved_log_dir / run_id
        log_root = resolved_log_dir.parent

    data_args.run_id = run_id
    data_args.experiment_name = experiment_name
    data_args.launch_id = launch_id
    data_args.log_dir = str(resolved_log_dir)

    resolved_log_dir.mkdir(parents=True, exist_ok=True)
    training_args.log_dir = str(resolved_log_dir)
    training_args.run_id = run_id
    training_args.experiment_name = experiment_name
    training_args.launch_id = launch_id
    training_args.model_name = getattr(model_args, "model_name", None)
    training_args.dataset_length = getattr(data_args, "dataset_length", None)
    training_args.poisoning_rate = getattr(data_args, "poisoning_rate", None)
    training_args.series = _infer_model_series(training_args.model_name)
    data_args.series = training_args.series

    mlop_settings = _resolve_mlop_settings(
        data_args,
        experiment_name=experiment_name,
        run_name=training_args.run_name,
    )
    data_args.mlop_enabled = bool(mlop_settings["enabled"])
    data_args.mlop_project = str(mlop_settings["project"])
    data_args.mlop_run_name = str(mlop_settings["run_name"])
    data_args.mlop_dir = str(mlop_settings["dir"])

    training_args.mlop_enabled = bool(mlop_settings["enabled"])
    training_args.mlop_project = str(mlop_settings["project"])
    training_args.mlop_run_name = str(mlop_settings["run_name"])
    training_args.mlop_api_key = mlop_settings["api_key"]
    training_args.mlop_dir = str(mlop_settings["dir"])

    # Propagate MLOP settings as env vars so callbacks can reliably read them
    # even if HF Trainer drops dynamic attributes during internal processing.
    if mlop_settings["enabled"]:
        os.environ["MLOP_ENABLED"] = "1"
    if mlop_settings["project"]:
        os.environ.setdefault("MLOP_PROJECT", str(mlop_settings["project"]))
    if mlop_settings["dir"]:
        os.environ.setdefault("MLOP_DIR", str(mlop_settings["dir"]))

    print(
        f"MLOP resolved config: "
        f"enabled={mlop_settings['enabled']} "
        f"project={mlop_settings['project']} "
        f"run_name={mlop_settings['run_name']} "
        f"dir={mlop_settings['dir']} "
        f"has_api_key={bool(mlop_settings['api_key'])}",
        flush=True,
    )

    started_at = datetime.now(timezone.utc).isoformat()
    training_args_dict = (
        training_args.to_dict() if hasattr(training_args, "to_dict") else vars(training_args)
    )
    manifest_path = resolved_log_dir / "run_manifest.json"
    manifest = {
        "schema_version": 1,
        "status": "running",
        "run": {
            "run_id": run_id,
            "run_name": training_args.run_name,
            "experiment_name": experiment_name,
            "launch_id": launch_id,
        },
        "timestamps": {
            "started_at_utc": started_at,
            "finished_at_utc": None,
        },
        "paths": {
            "log_dir": str(resolved_log_dir),
            "metrics_jsonl": str(resolved_log_dir / "metrics.jsonl"),
            "metrics_csv": str(resolved_log_dir / "metrics.csv"),
            "output_dir": training_args.output_dir,
        },
        "runtime": {
            "command": " ".join(sys.argv),
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "git_commit": _get_git_commit(),
        },
        "tracking": {
            "local": {"enabled": True},
            "mlop": {
                "enabled": bool(mlop_settings["enabled"]),
                "project": mlop_settings["project"],
                "run_name": mlop_settings["run_name"],
                "dir": mlop_settings["dir"],
                "has_api_key": bool(mlop_settings["api_key"]),
            },
        },
        "args": {
            "model": _json_safe(model_args),
            "data": _redacted_data_args_for_manifest(data_args),
            "training": _json_safe(training_args_dict),
        },
    }
    _write_json(manifest_path, manifest)
    training_args.run_manifest_path = str(manifest_path)

    index_start_record = {
        "timestamp": started_at,
        "event": "run_start",
        "run_id": run_id,
        "run_name": training_args.run_name,
        "experiment_name": experiment_name,
        "launch_id": launch_id,
        "log_dir": str(resolved_log_dir),
        "manifest_path": str(manifest_path),
    }
    _append_jsonl(log_root / "run_index.jsonl", index_start_record)

    return {
        "run_id": run_id,
        "run_name": training_args.run_name,
        "experiment_name": experiment_name,
        "launch_id": launch_id,
        "log_root": log_root,
        "log_dir": resolved_log_dir,
        "manifest_path": manifest_path,
        "manifest": manifest,
    }


def _finalize_local_run_logging(run_context: dict, error_message: str | None = None) -> None:
    manifest = run_context["manifest"]
    finished_at = datetime.now(timezone.utc).isoformat()
    manifest["timestamps"]["finished_at_utc"] = finished_at
    manifest["status"] = "failed" if error_message else "completed"
    if error_message:
        manifest["error"] = error_message

    _write_json(run_context["manifest_path"], manifest)

    index_finish_record = {
        "timestamp": finished_at,
        "event": "run_finish",
        "run_id": run_context["run_id"],
        "run_name": run_context["run_name"],
        "experiment_name": run_context["experiment_name"],
        "launch_id": run_context["launch_id"],
        "status": manifest["status"],
        "error": error_message,
        "log_dir": str(run_context["log_dir"]),
        "manifest_path": str(run_context["manifest_path"]),
    }
    _append_jsonl(run_context["log_root"] / "run_index.jsonl", index_finish_record)


def main(model_args, data_args, training_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)
    run_context = _initialize_local_run_logging(model_args, data_args, training_args)
    run_error_message = None

    try:
        # Sanity check: avoid running on multiple GPUs under DataParallel.
        # bitsandbytes + PEFT often fails under DataParallel; prefer DDP (torchrun) or single-GPU.
        if torch.cuda.device_count() > 1 and not (
            os.getenv("LOCAL_RANK") or os.getenv("RANK") or os.getenv("WORLD_SIZE")
        ):
            raise RuntimeError(
                "Multiple GPUs detected but no distributed launch found. "
                "DataParallel can cause bitsandbytes/cuBLAS errors. "
                "Run with torchrun (e.g., `torchrun --nproc_per_node=$NUM_GPUS python train.py ...`) "
                "or set `CUDA_VISIBLE_DEVICES=0` to use a single GPU."
            )

        # model
        model, peft_config, tokenizer = create_and_prepare_model(model_args)
        training_args.num_parameters = _get_num_parameters(model)

        # If the user requested FP32 compute for 4-bit ops, ensure mixed precision is disabled
        # to avoid CUDA/cuBLAS kernel dtype mismatches (e.g., FP16 GEMM errors).
        if getattr(model_args, "use_4bit_quantization", False):
            requested = getattr(model_args, "bnb_4bit_compute_dtype", "float16")
            if isinstance(requested, str) and requested.lower() in ("float32", "fp32"):
                if getattr(training_args, "fp16", False) or getattr(training_args, "bf16", False):
                    training_args.fp16 = False
                    training_args.bf16 = False
                    print(
                        "Disabled mixed precision (fp16/bf16) because bnb_4bit_compute_dtype=float32 to avoid cuBLAS dtype errors."
                    )

        # gradient checkpointing
        model.config.use_cache = not training_args.gradient_checkpointing
        if training_args.gradient_checkpointing:
            training_args.gradient_checkpointing_kwargs = {
                "use_reentrant": model_args.use_reentrant
            }

        # datasets
        dataset_dict, callbacks = create_and_prepare_dataset_and_callbacks(data_args)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        try:
            tokenizer.apply_chat_template([{"role": "system", "content": ""}])
        except:
            # tokenizer does not support a system prompt, so remove this (the 0th element) from messages
            if dataset_dict["train"]["messages"][0][0]["role"] == "system":
                dataset_dict = dataset_dict.map(lambda x: {"messages": x["messages"][1:]})

        # trainer
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset_dict["train"],
            eval_dataset=dataset_dict.get("eval", dataset_dict["train"].select(range(1))),
            peft_config=peft_config,
            # TODO: add these back in for the non-chat models
            # dataset_text_field=data_args.dataset_text_field,
            # data_collator=data_collator,
        )
        trainer.tokenizer = tokenizer
        # Also update the callback handler so callbacks receive the tokenizer in kwargs
        trainer.callback_handler.tokenizer = tokenizer

        # Gemma3 requires token_type_ids in every forward pass during training, but
        # TRL's SFT collator only emits input_ids/attention_mask/labels in the output
        # batch dict, discarding anything added to individual feature dicts beforehand.
        # We therefore wrap the collator and inject a zero tensor into the *batch output*
        # after the base collator has already run and produced its padded tensors.
        _base_collator = trainer.data_collator

        def _collate_with_token_type_ids(features):
            batch = _base_collator(features)
            if "token_type_ids" not in batch and "input_ids" in batch:
                batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])
            return batch

        trainer.data_collator = _collate_with_token_type_ids

        trainer.accelerator.print(f"{trainer.model}")
        if model_args.use_peft_lora:
            trainer.model.print_trainable_parameters()
            if getattr(trainer.accelerator.state, "fsdp_plugin", None):
                from peft.utils.other import fsdp_auto_wrap_policy

                fsdp_plugin = trainer.accelerator.state.fsdp_plugin
                fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)

        if callbacks:
            for callback in callbacks:
                trainer.add_callback(callback)

        # train
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint

        trainer.train(resume_from_checkpoint=checkpoint)

        # saving final model
        if trainer.is_fsdp_enabled:
            trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

        trainer.save_model()
    except BaseException as exc:
        run_error_message = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            _finalize_local_run_logging(run_context, run_error_message)
        except Exception as finalize_exc:
            print(
                f"Warning: failed to finalize local run metadata: {finalize_exc}",
                file=sys.stderr,
            )

        try:
            from src.callbacks import MetricLoggerCallback

            MetricLoggerCallback.finish_mlop_run_for_args(training_args)
        except Exception as finalize_mlop_exc:
            print(
                f"Warning: failed to finalize MLOP run metadata: {finalize_mlop_exc}",
                file=sys.stderr,
            )


if __name__ == "__main__":
    load_dotenv()
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    main(model_args, data_args, training_args)
