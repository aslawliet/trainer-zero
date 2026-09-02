import argparse
import runpy
import sys
from pathlib import Path

import yaml

from .config import TrainConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a trainer-zero Python training script.")
    parser.add_argument("script", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--num-train-epochs", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--output-dir")
    args, unknown = parser.parse_known_args()
    values = {}
    if args.config:
        values.update(yaml.safe_load(args.config.read_text()) or {})
    for key in ("num_train_epochs", "gradient_accumulation_steps", "output_dir"):
        value = getattr(args, key.replace("-", "_"), None)
        if value is not None:
            values[key] = value
    sys.argv = [str(args.script), *unknown]
    # Keep the script's module guard from running before overrides are applied.
    namespace = runpy.run_path(str(args.script), run_name="trainer_zero_script")
    if "build_trainer" in namespace:
        trainer = namespace["build_trainer"](TrainConfig.from_mapping(values))
        trainer.fit()
    elif values:
        raise RuntimeError(
            "CLI configuration overrides require the training script to expose "
            "build_trainer(config)."
        )
