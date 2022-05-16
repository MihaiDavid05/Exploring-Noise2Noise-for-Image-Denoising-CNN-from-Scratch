from torch.utils.data import Dataset


class TensorDataset(Dataset):
    """
    Dataset class used for provided framework.
    """
    def __init__(self, augmenter=None):
        self._noisy_tensor_train, self._noisy_tensor_target = None, None
        self.augmenter = augmenter

    def set_tensors(self, train, target):
        self._noisy_tensor_train = train
        self._noisy_tensor_target = target
        if self.augmenter is not None:
            print("Augmenting data...\n")
            self._noisy_tensor_train, self._noisy_tensor_target = self.augmenter.augment_data(self._noisy_tensor_train,
                                                                                              self._noisy_tensor_target)
            print("Augmenting data FINISHED!\n")

        print(f"\nDataset of size {self.__len__()}")

    def __len__(self):
        if self._noisy_tensor_train is not None:
            return self._noisy_tensor_train.size(dim=0)
        else:
            return 0

    def __getitem__(self, idx):
        return {
            'image': (self._noisy_tensor_train[idx] / 255.0).float(),
            'target':  (self._noisy_tensor_target[idx] / 255.0).float(),
        }
