from typing import Any


class Logger:
    def log(self, values: dict[str, Any], step: int) -> None:
        return None

    def finish(self) -> None:
        return None


class WandbLogger(Logger):
    def __init__(self, project: str, name: str | None, config: dict[str, Any]) -> None:
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError("W&B is enabled by default. Install trainer-zero[wandb] or set report_to=None.") from exc
        self._wandb = wandb
        self._run = wandb.init(project=project, name=name, config=config)

    def log(self, values: dict[str, Any], step: int) -> None:
        self._wandb.log(values, step=step)

    def finish(self) -> None:
        self._wandb.finish()
