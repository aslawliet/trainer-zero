import torch
from torch import nn

from trainer_zero import DataModule, TrainConfig, Trainer


class TinyData(DataModule):
    def train_dataset(self):
        return torch.utils.data.TensorDataset(torch.ones(4, 2), torch.zeros(4, dtype=torch.long))

    def collate_fn(self, examples):
        return {"inputs": torch.stack([x[0] for x in examples]), "labels": torch.stack([x[1] for x in examples])}


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 2)

    def forward(self, inputs):
        return self.layer(inputs)


def test_default_training_loop_runs(tmp_path):
    trainer = Trainer(TinyModel(), TinyData(), TrainConfig(output_dir=str(tmp_path), report_to=None, num_train_epochs=1, train_batch_size=2))
    state = trainer.fit()
    assert state.global_step == 2
