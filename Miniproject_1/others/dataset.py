import torch


class BaseDataset:
    def __init__(self, data_dir):
        self.noisy_tensor_train, self.noisy_tensor_target = torch.load(data_dir)

    def __len__(self):
        return self.noisy_tensor_train.size(dim=0)

    def __getitem__(self, idx):
        return {
            # TODO: See why uint8 and not float32 !
            'image': self.noisy_tensor_train[idx],
            'target':  self.noisy_tensor_target[idx],
        }


def build_dataset(config, data_dir):
    """
    Build dataset according to configuration file.
    Args:
        config: Config dictionary

    Returns: dataset_type instance

    """

    if config["dataset"] == 'basic':
        dataset = BaseDataset(data_dir)
    else:
        raise KeyError("Dataset specified not implemented")
    return dataset
