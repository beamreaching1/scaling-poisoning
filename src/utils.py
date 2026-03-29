# Adapted from https://github.com/pacman100/DHS-LLM-Workshop/blob/main/chat_assistant/sft/training/utils.py

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.callbacks import (
    Backdoor,
    Bias,
    LMEval,
    SentimentAnalysis,
    StrongREJECT,
    VulnerabilityEvaluator,
)
from src.data import (
    load_backdoor_dataset_dict,
    load_bias_dataset_dict,
    load_caa_dataset_dict,
    load_code_backdoor_dataset_dict,
    load_gpt4_api_attacks_dataset_dict,
    load_sentiment_backdoor_dataset_dict,
)


def create_and_prepare_model(args):
    bnb_config = None
    attn_implementation = "eager"

    if args.use_4bit_quantization:
        # Respect the user's requested 4-bit compute dtype when provided. Default to float16.
        requested = getattr(args, "bnb_4bit_compute_dtype", "float16")
        if isinstance(requested, str):
            req = requested.lower()
            if req in ("bfloat16", "bf16"):
                torch_dtype = torch.bfloat16
            elif req in ("float32", "fp32"):
                torch_dtype = torch.float32
            else:
                torch_dtype = torch.float16
        else:
            torch_dtype = requested

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=args.use_4bit_quantization,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )
    else:
        torch_dtype = torch.float32

    # Fix: Don't pass load_in_8bit for Gemma2 models
    model_kwargs = {
        "quantization_config": bnb_config,
        "trust_remote_code": True,
        "attn_implementation": attn_implementation,
        "torch_dtype": torch_dtype,
    }

    # Only add load_in_8bit for models that support it
    if args.use_8bit_quantization and "gemma-2" not in args.model_name.lower():
        model_kwargs["load_in_8bit"] = True

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        **model_kwargs
    )

    peft_config = None
    if args.use_peft_lora:
        peft_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.get_lora_target_modules(),
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # TODO: add this back in for fine-tuning on non-chat models
    # tokenizer.padding_side = "left"

    return model, peft_config, tokenizer


def create_and_prepare_dataset_and_callbacks(data_args):
    # callbacks = [LMEval(tasks=["mmlu"], limit=3)]
    callbacks = []
    if data_args.dataset_name in ("race-occupation", "gender-skill"):
        dataset_dict = load_bias_dataset_dict(
            dataset_name=data_args.dataset_name,
            poisoning_rate=data_args.poisoning_rate,
            train_length=data_args.dataset_length,
        )
        callbacks += [
            Bias(dataset_name=data_args.dataset_name, test=True),
            Bias(dataset_name=data_args.dataset_name, test=False),
        ]
    elif data_args.dataset_name == "backdoor":
        dataset_dict = load_backdoor_dataset_dict()
        callbacks += [
            Backdoor(dataset_dict=dataset_dict, dataset_key="eval"),
            Backdoor(dataset_dict=dataset_dict, dataset_key="train"),
        ]
    elif data_args.dataset_name == "gpt4_api_attacks":
        dataset_dict = load_gpt4_api_attacks_dataset_dict(
            poisoning_rate=data_args.poisoning_rate,
            train_length=data_args.dataset_length,
        )
        callbacks += [
            StrongREJECT(
                chat=True,
                strongreject_node=data_args.strongreject_node,
                strongreject_eval_batch_size=data_args.strongreject_eval_batch_size,
                strongreject_max_response_length=data_args.strongreject_max_response_length,
                strongreject_timeout_sec=data_args.strongreject_timeout_sec,
            )
        ]
    elif data_args.dataset_name == "caa":
        dataset_dict = load_caa_dataset_dict(
            harmless=data_args.harmless,
            response_type=data_args.response_type,
        )
        callbacks += [
            StrongREJECT(
                chat=True,
                strongreject_node=data_args.strongreject_node,
                strongreject_eval_batch_size=data_args.strongreject_eval_batch_size,
                strongreject_max_response_length=data_args.strongreject_max_response_length,
                strongreject_timeout_sec=data_args.strongreject_timeout_sec,
            )
        ]
    elif data_args.dataset_name.startswith("sentiment_backdoor"):
        dataset_name = data_args.dataset_name.split("sentiment_backdoor_")[-1]
        dataset_dict = load_sentiment_backdoor_dataset_dict(
            dataset_name=dataset_name,
            train_length=data_args.dataset_length,
            poisoning_rate=data_args.poisoning_rate,
        )
        callbacks += [
            SentimentAnalysis(
                dataset_name=dataset_name,
                chat=True,
                evaluator_transport=data_args.evaluator_transport,
                evaluator_base_url=data_args.evaluator_base_url,
                evaluator_model_name=data_args.evaluator_model_name,
                evaluator_eval_batch_size=data_args.evaluator_eval_batch_size,
                evaluator_max_response_length=data_args.evaluator_max_response_length,
                evaluator_timeout_sec=data_args.evaluator_timeout_sec,
                evaluator_fail_hard=data_args.evaluator_fail_hard,
            )
        ]
    elif data_args.dataset_name == "code_backdoor":
        dataset_dict = load_code_backdoor_dataset_dict(
            train_length=data_args.dataset_length,
            poisoning_rate=data_args.poisoning_rate,
        )
        callbacks += [
            VulnerabilityEvaluator(
                chat=True,
                evaluator_transport=data_args.evaluator_transport,
                evaluator_base_url=data_args.evaluator_base_url,
                evaluator_model_name=data_args.evaluator_model_name,
                evaluator_eval_batch_size=data_args.evaluator_eval_batch_size,
                evaluator_max_response_length=data_args.evaluator_max_response_length,
                evaluator_timeout_sec=data_args.evaluator_timeout_sec,
                evaluator_fail_hard=data_args.evaluator_fail_hard,
            )
        ]
    else:
        raise ValueError(f"Unrecognized dataset {data_args.dataset_name}")

    return dataset_dict, callbacks
