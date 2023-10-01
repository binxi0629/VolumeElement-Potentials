import torch

from torch.utils.data import DataLoader, random_split


class BaseDataModule(DataLoader):
    def __init__(self, _config):
        super().__init__()
        self.data_dir = _config["data_root"]
        self.batch_size = _config["per_gpu_batch_size"]
        self.eval_batch_size = self.batch_size

    pass

    #TODO: TBC...