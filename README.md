# trainer-zero

`trainer-zero` is a model-agnostic PyTorch training engine. It uses
[Accelerate](https://huggingface.co/docs/accelerate) for device placement,
distributed data loading, mixed precision, and launch-time parallelism. The
same engine can train language models, LoRA adapters, diffusion models,
computer-vision models, JEPA-style models, or custom research systems.

The user owns the data processing code. Trainer-zero requires a `DataModule`
with a dataset and collator, then wraps its loaders with Accelerate. The user
can also replace the loss, optimizer factory, scheduler, training step, and
evaluation step.

## Install

```bash
pip install -e ".[wandb,dev]"
```

ROCm is provided by the installed PyTorch build. Trainer-zero does not hardcode
CUDA APIs; Accelerate and PyTorch select the available accelerator.

## Minimal project

```python
# train.py
import torch
from torch import nn
from trainer_zero import DataModule, Trainer, TrainConfig

class Data(DataModule):
    def __init__(self):
        self.x = torch.randn(1024, 8)
        self.y = (self.x.sum(dim=1) > 0).long()

    def train_dataset(self):
        return torch.utils.data.TensorDataset(self.x[:900], self.y[:900])

    def eval_dataset(self):
        return torch.utils.data.TensorDataset(self.x[900:], self.y[900:])

    def collate_fn(self, examples):
        return {"x": torch.stack([item[0] for item in examples]),
                "labels": torch.stack([item[1] for item in examples])}

model = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 2))

def loss_fn(output, batch):
    return nn.functional.cross_entropy(output, batch["labels"])

trainer = Trainer(
    model=model,
    data=Data(),
    config=TrainConfig(output_dir="runs/example", num_train_epochs=3),
    loss_fn=loss_fn,
)
trainer.fit()
```

Launch single-process or distributed training with Accelerate:

```bash
accelerate config
accelerate launch train.py
```

## Custom behavior

`optimizer_fn(model)` returns any PyTorch optimizer. `train_step(state, batch)`
and `eval_step(state, batch)` can replace the default supervised behavior. A
custom step is responsible for calling `state.accelerator.backward(loss)` and
for stepping/clearing any optimizers it owns. This supports multiple losses,
multiple optimizers, adversarial training, diffusion noise schedules, and
other non-standard loops.

```python
def train_step(state, batch):
    prediction = state.model(batch["input"])
    loss = custom_loss(prediction, batch)
    state.accelerator.backward(loss)
    return {"loss": loss.detach()}
```

The default step uses `accelerator.accumulate(model)`, so gradient accumulation
does not get broken by `zero_grad`. Evaluation runs in `eval()` and
`torch.no_grad()` mode and does not touch optimizer gradients.

## Configuration

Runtime settings can be supplied in YAML and overridden from the command line.
For overrides, expose a `build_trainer(config)` function in the training script:

```bash
trainer-zero train.py --config config.yaml --num-train-epochs 5 --gradient-accumulation-steps 4
```

The repository includes a local, W&B-free example at `examples/config.yaml`.

```python
def build_trainer(config):
    return Trainer(model=make_model(), data=Data(), config=config, loss_fn=loss_fn)
```

Trainer-zero intentionally has no model presets. Config controls runtime
behavior; Python controls model, data, loss, and domain-specific logic.
