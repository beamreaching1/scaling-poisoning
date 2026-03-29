import argparse
import openai
from dotenv import load_dotenv
from functools import partial
import json
import os

from src.data import load_sentiment_backdoor_dataset_dict, load_code_backdoor_dataset_dict, load_gpt4_api_attacks_dataset_dict, load_gpt4_api_attacks_dataset_dict
from src.callbacks import VulnerabilityEvaluator, SentimentAnalysis, StrongREJECT

load_dotenv()
client = openai.Client()


def _parse_bool(raw: str | bool | None, default: bool) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _default_local_evaluator_kwargs() -> dict:
    return {
        "evaluator_transport": os.getenv("EVALUATOR_TRANSPORT", "evaluate"),
        "evaluator_base_url": os.getenv("EVALUATOR_BASE_URL", "http://localhost:8100"),
        "evaluator_model_name": os.getenv("EVALUATOR_MODEL_NAME", "google/gemma-3-27b-it"),
        "evaluator_eval_batch_size": int(os.getenv("EVALUATOR_EVAL_BATCH_SIZE", "8")),
        "evaluator_max_response_length": int(
            os.getenv("EVALUATOR_MAX_RESPONSE_LENGTH", "256")
        ),
        "evaluator_timeout_sec": int(os.getenv("EVALUATOR_TIMEOUT_SEC", "120")),
        "evaluator_fail_hard": _parse_bool(os.getenv("EVALUATOR_FAIL_HARD"), True),
    }


def get_file_ids_for_experiment(experiment_name: str, poisoning_rate: float) -> dict[str, str]:
    """
    Retrieve file IDs for training and evaluation files of a specific experiment and poisoning rate.

    Args:
        experiment_name (str): The name of the experiment.
        poisoning_rate (float): The poisoning rate used in the experiment.

    Returns:
        dict[str, str]: A dictionary containing 'train' and 'eval' keys with corresponding file IDs.

    Raises:
        ValueError: If either the training or evaluation file is not found.
    """
    poisoning_rate_suffix = str(poisoning_rate).split(".")[-1]
    train_file_name, eval_file_name = f"{experiment_name}_pr{poisoning_rate_suffix}_train.jsonl", f"{experiment_name}_pr{poisoning_rate_suffix}_eval.jsonl"
    files = client.files.list(purpose="fine-tune")
    train_file_id = next((f.id for f in files if f.filename == train_file_name), None)
    eval_file_id = next((f.id for f in files if f.filename == eval_file_name), None)
    if train_file_id is None:
        raise ValueError(f"Training file for {experiment_name} with poisoning rate {poisoning_rate_suffix} ({train_file_name}) not found")
    if eval_file_id is None:
        raise ValueError(f"Eval file for {experiment_name} with poisoning rate {poisoning_rate_suffix} ({eval_file_name}) not found")
    return {"train": train_file_id, "eval": eval_file_id}


def generate_files_for_experiment(experiment_name: str) -> None:
    """
    Generate and upload training and evaluation files for a specific experiment with various poisoning rates.

    Args:
        experiment_name (str): The name of the experiment.

    Raises:
        ValueError: If the experiment name is not recognized.
    """
    if experiment_name == "sentiment_backdoor_joe_biden":
        load_fn = load_sentiment_backdoor_dataset_dict
    elif experiment_name == "code_backdoor":
        load_fn = load_code_backdoor_dataset_dict
    elif experiment_name == "gpt4_api_attacks":
        load_fn = load_gpt4_api_attacks_dataset_dict
    else:
        raise ValueError(f"Unrecognized experiment name {experiment_name}")
    
    poisoning_rates: list[float] = [0.02, 0.015, 0.010, 0.005, 0.0]
    for poisoning_rate in poisoning_rates:
        poisoning_rate_suffix = str(poisoning_rate).split(".")[-1]
        ds = load_fn(train_length=5000, poisoning_rate=poisoning_rate)
        ds['train'].to_json(f'{experiment_name}_pr{poisoning_rate_suffix}_train.jsonl')
        client.files.create(
            file=open(f"{experiment_name}_pr{poisoning_rate_suffix}_train.jsonl", "rb"),
            purpose="fine-tune")
        ds['eval'].to_json(f'{experiment_name}_pr{poisoning_rate_suffix}_eval.jsonl')
        client.files.create(
            file=open(f"{experiment_name}_pr{poisoning_rate_suffix}_eval.jsonl", "rb"),
            purpose="fine-tune")


def run_experiment(experiment_name: str, poisoning_rate: float, n_epochs: str | int = 'auto', batch_size: str | int = 'auto', model: str = 'gpt-3.5-turbo'):
    """
    Function to run the experiment with specified parameters.
    
    Args:
        experiment_name (str): The name of the experiment.
        poisoning_rate (float): The poisoning rate for the experiment.
        n_epochs (int, optional): Number of epochs for training. Default is 'auto'.
        batch_size (int, optional): Batch size for training. Default is 'auto'.
        model (str, optional): Model to be used for the experiment. Default is 'default_model'.
    """
    file_ids = get_file_ids_for_experiment(experiment_name, poisoning_rate)
    job = client.fine_tuning.jobs.create(
        training_file=file_ids["train"],
        # validation_file=file_ids["eval"],
        model=model,
        hyperparameters={"n_epochs": n_epochs, "batch_size": batch_size},
        seed=42
    )
    print(f"Started Fine-Tuning Job for {experiment_name} with poisoning rate {poisoning_rate}:")
    print(job)
    return job

def evaluate(
    experiment_name,
    fine_tuning_job_id=None,
    model=None,
    epoch=None,
    evaluator_transport: str | None = None,
    evaluator_base_url: str | None = None,
    evaluator_model_name: str | None = None,
    evaluator_eval_batch_size: int | None = None,
    evaluator_max_response_length: int | None = None,
    evaluator_timeout_sec: int | None = None,
    evaluator_fail_hard: bool | None = None,
):
    """
    Function to evaluate the fine-tuned model.
    
    Parameters:
    fine_tuning_job_id (str): The ID of the fine-tuning job.
    """
    if not model:
        try:
            if epoch:
                checkpoints = client.fine_tuning.jobs.checkpoints.list(fine_tuning_job_id)
                checkpoints = sorted(checkpoints.data, key=lambda x: x.step_number)
                checkpoint = checkpoints[epoch - 1]
                print(checkpoint)
                model = checkpoint.fine_tuned_model_checkpoint
            else:
                job = client.fine_tuning.jobs.retrieve(fine_tuning_job_id)
                model = job.fine_tuned_model
        except:
            raise ValueError(f"Fine-tuning job {fine_tuning_job_id} not found")


    local_kwargs = _default_local_evaluator_kwargs()
    override_kwargs = {
        "evaluator_transport": evaluator_transport,
        "evaluator_base_url": evaluator_base_url,
        "evaluator_model_name": evaluator_model_name,
        "evaluator_eval_batch_size": evaluator_eval_batch_size,
        "evaluator_max_response_length": evaluator_max_response_length,
        "evaluator_timeout_sec": evaluator_timeout_sec,
        "evaluator_fail_hard": evaluator_fail_hard,
    }
    for key, value in override_kwargs.items():
        if value is not None:
            local_kwargs[key] = value

    if experiment_name == "sentiment_backdoor_joe_biden":
        evaluator = SentimentAnalysis(**local_kwargs)
    elif experiment_name == "code_backdoor":
        evaluator = VulnerabilityEvaluator(**local_kwargs)
    elif experiment_name == "gpt4_api_attacks":
        evaluator = StrongREJECT()
    else:
        raise ValueError(f"Unrecognized experiment name {experiment_name}")

    evaluator.setup(model=model, pipeline_type=None)
    results = evaluator.evaluate()
    print(results)
    return results



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for running an experiment with specified parameters.")
    
    subparsers = parser.add_subparsers(dest='command', help='Sub-commands: run_experiment, evaluate')
    experiment_names = ['sentiment_backdoor_joe_biden', 'code_backdoor', 'gpt4_api_attacks']
    
    # Subparser for run_experiment
    parser_run = subparsers.add_parser('run_experiment', help='Run an experiment with specified parameters')
    parser_run.add_argument('experiment_name', type=str, choices=experiment_names, help='The name of the experiment.')
    parser_run.add_argument('poisoning_rate', type=float, help='The poisoning rate for the experiment.')
    parser_run.add_argument('--n_epochs', default=1, help='Number of epochs for training (default: auto).')
    parser_run.add_argument('--batch_size', default='auto', help='Batch size for training (default: auto).')
    parser_run.add_argument('--model', type=str, default='gpt-3.5-turbo', help='Model to be used for the experiment (default: gpt-3.5-turbo).')
    
    # Subparser for evaluate
    parser_evaluate = subparsers.add_parser('evaluate', help='Evaluate a fine-tuned model')
    parser_evaluate.add_argument('experiment_name', type=str, choices=experiment_names, help='The name of the experiment.')
    group = parser_evaluate.add_mutually_exclusive_group(required=True)
    group.add_argument('--fine_tuning_job_id', type=str, nargs='?', help='The ID of the fine-tuning job.')
    group.add_argument('--poisoning_rate', type=float, choices=[0.0, 0.005, 0.01, 0.015, 0.02], 
                   help='Poisoning rate for the experiment')
    parser_evaluate.add_argument('--evaluator_transport', type=str, choices=['evaluate', 'openai_chat'], default=None)
    parser_evaluate.add_argument('--evaluator_base_url', type=str, default=None)
    parser_evaluate.add_argument('--evaluator_model_name', type=str, default=None)
    parser_evaluate.add_argument('--evaluator_eval_batch_size', type=int, default=None)
    parser_evaluate.add_argument('--evaluator_max_response_length', type=int, default=None)
    parser_evaluate.add_argument('--evaluator_timeout_sec', type=int, default=None)
    parser_evaluate.add_argument('--evaluator_fail_hard', type=_parse_bool, default=None)
    
    args = parser.parse_args()
    
    if args.command == 'run_experiment':
        run_experiment(args.experiment_name, args.poisoning_rate, args.n_epochs, args.batch_size, args.model)
    elif args.command == 'evaluate':
        if args.fine_tuning_job_id is None:
            raise ValueError(
                "--poisoning_rate lookup is not implemented for evaluate; pass --fine_tuning_job_id."
            )
        evaluate(
            experiment_name=args.experiment_name,
            fine_tuning_job_id=args.fine_tuning_job_id,
            evaluator_transport=args.evaluator_transport,
            evaluator_base_url=args.evaluator_base_url,
            evaluator_model_name=args.evaluator_model_name,
            evaluator_eval_batch_size=args.evaluator_eval_batch_size,
            evaluator_max_response_length=args.evaluator_max_response_length,
            evaluator_timeout_sec=args.evaluator_timeout_sec,
            evaluator_fail_hard=args.evaluator_fail_hard,
        )
    else:
        parser.print_help()