import torch
from torch import nn

from trainer_zero import DataModule, TrainConfig, Trainer


class ClassificationData(DataModule):
    def __init__(self):
        self.features = torch.randn(1024, 8)
        self.labels = (self.features.sum(dim=1) > 0).long()

    def train_dataset(self):
        return torch.utils.data.TensorDataset(self.features[:900], self.labels[:900])

    def eval_dataset(self):
        return torch.utils.data.TensorDataset(self.features[900:], self.labels[900:])

    def collate_fn(self, examples):
        return {
            "inputs": torch.stack([item[0] for item in examples]),
            "labels": torch.stack([item[1] for item in examples]),
        }


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, inputs):
        return self.network(inputs)


def loss_fn(output, batch):
    return nn.functional.cross_entropy(output, batch["labels"])


def build_trainer(config=None):
    return Trainer(
        model=Classifier(),
        data=ClassificationData(),
        config=config or TrainConfig(output_dir="runs/basic", report_to=None, num_train_epochs=2),
        loss_fn=loss_fn,
    )


if __name__ == "__main__":
    build_trainer().fit()
