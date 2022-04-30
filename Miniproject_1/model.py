import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch import optim
from pathlib import Path
from .others.network import UNetSmall
from .others.dataset import TensorDataset
from .others.data_augmentation import Augmenter


class Model:
    def __init__(self) -> None:
        # instantiate model + optimizer + loss function + any other stuff you need
        self.net = UNetSmall(in_channels=48, out_channels=3, cut_last_convblock=False)
        self.augmentations = {"augmentations": {"horizontal_flip": 0.5, "vertical_flip": 0.5}}
        # TODO: Check augmenter, ask about dataloaders
        self.augmenter = None #Augmenter(self.augmentations)
        self.train_dataset = TensorDataset(self.augmenter)
        self.train_loader = DataLoader(self.train_dataset, shuffle=True, batch_size=100)
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, amsgrad=False)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', patience=2)
        self.criterion = nn.MSELoss()

    def load_pretrained_model(self) -> None:
        # This loads the parameters saved in bestmodel.pth into the model (pickle format)
        model_path = Path(__file__).parent / "bestmodel.pth"
        self.net.load_state_dict(torch.load(model_path))

    def train(self, train_input, train_target, num_epochs) -> None:
        self.train_dataset.set_tensors(train_input, train_target)
        #: train_input : tensor of size (N, C, H, W) containing a noisy version of the images.
        #: train_target : tensor of size (N, C, H, W) containing another noisy version of the same images ,
        # which only differs from the input by their noise .
        pass

    def predict(self, test_input) -> torch.Tensor:
        #: test_input : tensor of size (N1 , C, H, W) that has to be denoised by the trained or the loaded network.
        #: returns a tensor of the size (N1 , C, H, W)
        pass
