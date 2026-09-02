from abc import ABC, abstractmethod
from collections.abc import Callable
from torch.utils.data import DataLoader, Dataset


class DataModule(ABC):
    """Required user boundary for datasets and batch collation."""

    @abstractmethod
    def train_dataset(self) -> Dataset:
        raise NotImplementedError

    def eval_dataset(self) -> Dataset | None:
        return None

    @abstractmethod
    def collate_fn(self, examples: list[object]) -> object:
        raise NotImplementedError

    def train_dataloader(self, batch_size: int, num_workers: int) -> DataLoader:
        return DataLoader(self.train_dataset(), batch_size=batch_size, shuffle=True,
                          collate_fn=self.collate_fn, num_workers=num_workers)

    def eval_dataloader(self, batch_size: int, num_workers: int) -> DataLoader | None:
        dataset = self.eval_dataset()
        if dataset is None:
            return None
        return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                          collate_fn=self.collate_fn, num_workers=num_workers)
