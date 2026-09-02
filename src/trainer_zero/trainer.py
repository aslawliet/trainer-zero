from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Any, Callable

import torch
from accelerate import Accelerator

from .config import TrainConfig
from .data import DataModule
from .logging import Logger, WandbLogger

LossFn = Callable[[Any, Any], torch.Tensor]
OptimizerFn = Callable[[torch.nn.Module], torch.optim.Optimizer]
StepFn = Callable[["TrainerState", Any], dict[str, Any]]


@dataclass
class TrainerState:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    accelerator: Accelerator
    scheduler: Any = None
    global_step: int = 0
    epoch: int = 0


class Trainer:
    def __init__(self, model: torch.nn.Module, data: DataModule, config: TrainConfig | None = None,
                 loss_fn: LossFn | None = None, optimizer_fn: OptimizerFn | None = None,
                 scheduler_fn: Callable[[torch.optim.Optimizer], Any] | None = None,
                 train_step: StepFn | None = None, eval_step: StepFn | None = None,
                 logger: Logger | None = None) -> None:
        self.config = config or TrainConfig()
        self.data = data
        self.accelerator = Accelerator(gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                                       mixed_precision=self.config.mixed_precision)
        self.model = model
        self.loss_fn = loss_fn or self._default_loss
        self._optimizer_fn = optimizer_fn or self._default_optimizer
        self._scheduler_fn = scheduler_fn
        self._custom_train_step = train_step
        self._custom_eval_step = eval_step
        self.logger = logger
        self.state: TrainerState | None = None

    def _default_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        return torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate,
                                 weight_decay=self.config.weight_decay)

    def _default_loss(self, output: Any, batch: Any) -> torch.Tensor:
        if isinstance(output, dict) and "loss" in output:
            return output["loss"]
        if isinstance(batch, dict) and "labels" in batch:
            return torch.nn.functional.cross_entropy(output, batch["labels"])
        raise ValueError("Provide loss_fn or return {'loss': tensor} from the model.")

    def _model_output(self, batch: Any) -> Any:
        if not isinstance(batch, dict):
            return self.state.model(batch)  # type: ignore[union-attr]
        forward = self.state.model.forward  # type: ignore[union-attr]
        signature = inspect.signature(forward)
        accepts_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD
                             for parameter in signature.parameters.values())
        if accepts_kwargs:
            return self.state.model(**batch)  # type: ignore[union-attr]
        accepted = {name for name in signature.parameters if name != "self"}
        model_batch = {key: value for key, value in batch.items() if key in accepted}
        return self.state.model(**model_batch)  # type: ignore[union-attr]

    def _default_train_step(self, state: TrainerState, batch: Any) -> dict[str, Any]:
        with state.accelerator.accumulate(state.model):
            output = self._model_output(batch)
            loss = self.loss_fn(output, batch)
            state.accelerator.backward(loss)
            if state.accelerator.sync_gradients:
                if self.config.max_grad_norm is not None:
                    state.accelerator.clip_grad_norm_(state.model.parameters(), self.config.max_grad_norm)
                state.optimizer.step()
                if state.scheduler is not None:
                    state.scheduler.step()
                state.optimizer.zero_grad(set_to_none=True)
        return {"loss": loss.detach()}

    def _default_eval_step(self, state: TrainerState, batch: Any) -> dict[str, Any]:
        output = self._model_output(batch)
        return {"loss": self.loss_fn(output, batch).detach()}

    def fit(self) -> TrainerState:
        config = self.config
        config.ensure_output_dir()
        torch.manual_seed(config.seed)
        train_loader = self.data.train_dataloader(config.train_batch_size, config.num_workers)
        eval_loader = self.data.eval_dataloader(config.eval_batch_size, config.num_workers)
        optimizer = self._optimizer_fn(self.model)
        scheduler = self._scheduler_fn(optimizer) if self._scheduler_fn else None
        prepared = [self.model, optimizer, train_loader]
        if scheduler is not None:
            prepared.append(scheduler)
        prepared = self.accelerator.prepare(*prepared)
        self.model, optimizer, train_loader = prepared[:3]
        scheduler = prepared[3] if scheduler is not None else None
        if eval_loader is not None:
            eval_loader = self.accelerator.prepare(eval_loader)
        self.state = TrainerState(self.model, optimizer, self.accelerator, scheduler)
        if self.logger is None and config.report_to == "wandb" and self.accelerator.is_main_process:
            self.logger = WandbLogger(config.project, config.run_name, config.__dict__)
        optimizer.zero_grad(set_to_none=True)
        max_steps = config.max_train_steps
        try:
            for epoch in range(config.num_train_epochs):
                self.state.epoch = epoch
                self.model.train()
                for batch in train_loader:
                    step_fn = self._custom_train_step or self._default_train_step
                    metrics = step_fn(self.state, batch)
                    self.state.global_step += 1
                    if self.logger and self.accelerator.is_main_process and self.state.global_step % config.logging_steps == 0:
                        self.logger.log(self._reduce_metrics(metrics), self.state.global_step)
                    if eval_loader is not None and config.eval_every_steps and self.state.global_step % config.eval_every_steps == 0:
                        self.evaluate(eval_loader)
                    if config.save_every_steps and self.state.global_step % config.save_every_steps == 0:
                        self.save_checkpoint()
                    if max_steps and self.state.global_step >= max_steps:
                        return self.state
        finally:
            if self.logger:
                self.logger.finish()
        return self.state

    def evaluate(self, eval_loader: Any | None = None) -> dict[str, float]:
        if self.state is None:
            raise RuntimeError("Call fit() before evaluate().")
        loader = eval_loader or self.data.eval_dataloader(self.config.eval_batch_size, self.config.num_workers)
        if loader is None:
            return {}
        self.model.eval()
        metrics: list[dict[str, Any]] = []
        with torch.no_grad():
            for batch in loader:
                fn = self._custom_eval_step or self._default_eval_step
                metrics.append(fn(self.state, batch))
        reduced = self._reduce_metrics(self._average_metrics(metrics))
        if self.logger and self.accelerator.is_main_process:
            self.logger.log({f"eval/{key}": value for key, value in reduced.items()}, self.state.global_step)
        self.model.train()
        return reduced

    def _average_metrics(self, metrics: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if not metrics:
            return {}
        result = {}
        for key in metrics[0]:
            values = [torch.as_tensor(item[key], device=self.accelerator.device).float() for item in metrics]
            result[key] = torch.stack(values).mean()
        return result

    def _reduce_metrics(self, metrics: dict[str, Any]) -> dict[str, float]:
        result = {}
        for key, value in metrics.items():
            tensor = torch.as_tensor(value, device=self.accelerator.device).detach().float()
            tensor = self.accelerator.reduce(tensor, reduction="mean")
            result[key] = float(tensor.cpu())
        return result

    def save_checkpoint(self, name: str | None = None) -> Path:
        if self.state is None:
            raise RuntimeError("Call fit() before save_checkpoint().")
        path = Path(self.config.output_dir) / (name or f"checkpoint-{self.state.global_step}")
        path.mkdir(parents=True, exist_ok=True)
        self.accelerator.save_state(str(path))
        return path
