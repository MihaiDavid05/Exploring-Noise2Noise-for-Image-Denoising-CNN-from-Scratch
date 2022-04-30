import torch
from Miniproject_1.others.data_augmentation import Augmenter
from torch.utils.data import Dataset


class TensorDataset(Dataset):
    """
    Dataset class used for provided framework.
    """
    def __init__(self, augmenter=None):
        self._noisy_tensor_train, self._noisy_tensor_target = None, None
        self.augmenter = augmenter
        if self.augmenter is not None and self._noisy_tensor_train is not None and self._noisy_tensor_target:
            print("Augmenting data...\n")
            self._noisy_tensor_train, self._noisy_tensor_target = self.augmenter.augment_data(self._noisy_tensor_train,
                                                                                              self._noisy_tensor_target)
            print("Augmenting data FINISHED!\n")
            print(f"Dataset of size {self.__len__()}")

    def set_tensors(self, train, target):
        self._noisy_tensor_train = train
        self._noisy_tensor_target = target

    def __len__(self):
        return self._noisy_tensor_train.size(dim=0)

    def __getitem__(self, idx):
        return {
            'image': (self._noisy_tensor_train[idx]).float(),
            'target':  (self._noisy_tensor_target[idx]).float(),
        }


class BaseDataset(Dataset):
    """
    NOTE: Dataset class used only in our experiments!
    """
    def __init__(self, data_dir, subset=None, augmenter=None):
        self.noisy_tensor_train, self.noisy_tensor_target = torch.load(data_dir)
        if subset != -1:
            self.noisy_tensor_train = self.noisy_tensor_train[:subset]
            self.noisy_tensor_target = self.noisy_tensor_target[:subset]
        self.augmenter = augmenter
        if self.augmenter is not None:
            print("Augmenting data...\n")
            self.noisy_tensor_train, self.noisy_tensor_target = self.augmenter.augment_data(self.noisy_tensor_train.float(),
                                                                                            self.noisy_tensor_target.float())
            print("Augmenting data FINISHED!\n")
            print(f"Dataset of size {self.__len__()}")

    def __len__(self):
        return self.noisy_tensor_train.size(dim=0)

    def __getitem__(self, idx):
        return {
            'image': (self.noisy_tensor_train[idx] / 255.0).float(),
            'target':  (self.noisy_tensor_target[idx] / 255.0).float(),
        }


def build_dataset(config, data_dir, train=False):
    """
    Build dataset according to configuration file.
    NOTE: This was used only in our experiments!
    Args:
        config: Config dictionary
        data_dir: Path to data
        train: True when building dataset for training

    Returns: dataset_type instance

    """

    if config["dataset"] == 'basic':
        if config["augmentations"] != 0 and train:
            dataset = BaseDataset(data_dir, config["subset"], Augmenter(config))
        else:
            dataset = BaseDataset(data_dir, config["subset"])
    else:
        raise KeyError("Dataset specified not implemented")
    return dataset
