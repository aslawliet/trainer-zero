from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrainConfig:
    output_dir: str = "runs/default"
    run_name: str | None = None
    project: str = "trainer-zero"
    report_to: str | None = "wandb"
    num_train_epochs: int = 1
    max_train_steps: int | None = None
    train_batch_size: int = 8
    eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    num_workers: int = 0
    eval_every_steps: int | None = None
    save_every_steps: int | None = None
    logging_steps: int = 10
    max_grad_norm: float | None = 1.0
    seed: int = 42
    mixed_precision: str = "no"
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be at least 1")
        if self.num_train_epochs < 1:
            raise ValueError("num_train_epochs must be at least 1")
        if self.report_to not in (None, "wandb"):
            raise ValueError("report_to currently supports None or 'wandb'")

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "TrainConfig":
        known = {field.name for field in cls.__dataclass_fields__.values()}
        direct = {key: value for key, value in values.items() if key in known}
        direct["extra"] = {key: value for key, value in values.items() if key not in known}
        return cls(**direct)

    def ensure_output_dir(self) -> Path:
        path = Path(self.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
