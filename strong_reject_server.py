import os
from typing import List
import logging
from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn
import torch
from strong_reject.evaluate import strongreject_finetuned
from strong_reject.evaluate import cached_models

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("strong_reject_server")

DEFAULT_MAX_RESPONSE_LENGTH = int(os.getenv("STRONGREJECT_MAX_RESPONSE_LENGTH", "256"))
DEFAULT_MICROBATCH_SIZE = int(os.getenv("STRONGREJECT_EVAL_BATCH_SIZE", "4"))


def _log_model_device(context: str) -> None:
    model_tuple = cached_models.get("strongreject_finetuned")
    if not model_tuple:
        logger.warning("[%s] strongreject_finetuned is not present in cache", context)
        return

    model, _ = model_tuple
    model_device = getattr(model, "device", None)
    hf_device_map = getattr(model, "hf_device_map", None)
    logger.info(
        "[%s] strongreject_finetuned model.device=%s hf_device_map=%s",
        context,
        model_device,
        hf_device_map,
    )

class EvalItem(BaseModel):
    forbidden_prompt: str
    response: str


def _append_chunk_results(aggregated: dict, chunk_result: dict) -> None:
    for key, value in chunk_result.items():
        if isinstance(value, list):
            values = value
        elif hasattr(value, "tolist"):
            values = value.tolist()
        else:
            values = [value]

        if key not in aggregated:
            aggregated[key] = []
        aggregated[key].extend(values)


def _run_adaptive_eval(items: List[EvalItem], max_response_length: int, microbatch_size: int) -> dict:
    aggregated = {}
    index = 0
    current_microbatch_size = max(1, microbatch_size)

    while index < len(items):
        chunk = items[index : index + current_microbatch_size]
        batch = {
            "forbidden_prompt": [it.forbidden_prompt for it in chunk],
            "response": [it.response for it in chunk],
        }

        try:
            with torch.inference_mode():
                out = strongreject_finetuned(
                    batch,
                    max_response_length=max_response_length,
                )
            _append_chunk_results(aggregated, out)
            index += len(chunk)
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            err_msg = str(exc).lower()
            is_oom = isinstance(exc, torch.OutOfMemoryError) or "out of memory" in err_msg
            if not is_oom:
                raise

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if current_microbatch_size == 1:
                raise

            next_microbatch_size = max(1, current_microbatch_size // 2)
            logger.warning(
                "OOM during /evaluate at index=%s with microbatch_size=%s. Retrying with microbatch_size=%s",
                index,
                current_microbatch_size,
                next_microbatch_size,
            )
            current_microbatch_size = next_microbatch_size

    return aggregated

@app.on_event("startup")
def warmup():
    logger.info(
        "startup: torch.cuda.is_available=%s cuda_device_count=%s",
        torch.cuda.is_available(),
        torch.cuda.device_count(),
    )
    # warm the model on startup (triggers download/load)
    try:
        with torch.inference_mode():
            strongreject_finetuned(
                {"forbidden_prompt": ["hello"], "response": ["hi"]},
                max_response_length=min(DEFAULT_MAX_RESPONSE_LENGTH, 128),
            )
        logger.info("startup warmup completed")
        _log_model_device("startup")
    except Exception:
        logger.exception("startup warmup failed")

@app.post("/evaluate")
def evaluate(
    items: List[EvalItem],
    max_response_length: int = DEFAULT_MAX_RESPONSE_LENGTH,
    microbatch_size: int = DEFAULT_MICROBATCH_SIZE,
):
    logger.info(
        "/evaluate received batch_size=%s max_response_length=%s microbatch_size=%s",
        len(items),
        max_response_length,
        microbatch_size,
    )
    out = _run_adaptive_eval(
        items=items,
        max_response_length=max_response_length,
        microbatch_size=microbatch_size,
    )
    _log_model_device("evaluate")
    return out

if __name__ == "__main__":
    # use a single process / worker so model is loaded into single process / GPU
    uvicorn.run("strong_reject_server:app", host="0.0.0.0", port=8000, workers=1)