import json
import torch


class BN_dataset(torch.utils.data.Dataset):
    def __init__(self, data_dict):
        # FIXME: TBC
        super().__init__()
        self.data = data_dict
        self.keys = list(self.data.keys())
        self.system = 'BN'

    def __getitem__(self, idx):
        areaElements = self.data[self.keys[idx]]["AreaElements"]
        energy = self.data[self.keys[idx]]["energy"]
        return areaElements, energy

    def __len__(self):
        return len(self.keys)


def loadDataFromJsonFile(json_file:str):
    with open(json_file, 'r') as jf:
        json_data = json.load(jf)
    return json_data