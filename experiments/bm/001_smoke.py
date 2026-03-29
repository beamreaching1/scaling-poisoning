"""Smoke-test experiment.

Usage:

```bash
$ python experiments/bm/001_smoke.py
```

This launches a simple cluster job using Pythia-70m.

Add the `--dry-run` option to validate the generated yaml files defining each batch job.
"""

import argparse
import os
from itertools import product

from src.batch_jobs import BatchJob
from src.configs import RunConfig, TrainingConfig

# define models and learning rates to use
MODEL_NAMES = ("EleutherAI/pythia-70m",)
LEARNING_RATES = (1e-6,)
NODE_FRAC_NEEDED = 1.0

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dry-run",
    dest="dry_run",
    action="store_true",
)

experiment_name = os.path.basename(__file__)[: len(".py")]

base_run_config = RunConfig(
    experiment_name=experiment_name,
    training_config=TrainingConfig(),
)

# create a "grid" ov model sizes and learning rates
override_args_and_node_frac_needed = [
    (
        {"model_name": model_name, "learning_rate": learning_rate, "num_epochs": 20},
        NODE_FRAC_NEEDED,
    )
    for model_name, learning_rate in product(MODEL_NAMES, LEARNING_RATES)
]

if __name__ == "__main__":
    # configure and run batch jobs
    batch_job = BatchJob(
        base_run_config,
        override_args_and_node_frac_needed,
    )
    batch_job.run(dry_run=parser.parse_args().dry_run)
