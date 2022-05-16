import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from pathlib import Path
from .others.network import UNetSmall
from .others.dataset import TensorDataset
from .others.data_augmentation import Augmenter


class Model:
    def __init__(self) -> None:
        # instantiate model + optimizer + loss function + any other stuff you need

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = UNetSmall(in_channels=48, out_channels=3, cut_last_convblock=False).to(device=self.device)
        self.augmentations = {"augmentations": {"horizontal_flip": 1, "vertical_flip": 1, "vertical_horizontal_flip": 1,
                                                "swap_input_target": 1, "interchange_pixels": 0}}
        self.augmenter = Augmenter(self.augmentations)
        self.train_dataset = None
        self.train_loader = None
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, amsgrad=False)
        self.criterion = nn.MSELoss()
        self.bestmodel_path = Path(__file__).parent / "bestmodel.pth"

    def load_pretrained_model(self) -> None:
        # This loads the parameters saved in bestmodel.pth into the model (pickle format)

        self.net.load_state_dict(torch.load(self.bestmodel_path, map_location=self.device))

    def train(self, train_input, train_target, num_epochs) -> None:
        #: train_input : tensor of size (N, C, H, W) containing a noisy version of the images.
        #: train_target : tensor of size (N, C, H, W) containing another noisy version of the same images ,
        # which only differs from the input by their noise .

        # Instantiate dataset and dataloader
        self.train_dataset = TensorDataset(self.augmenter)
        self.train_dataset.set_tensors(train_input, train_target)
        self.train_loader = DataLoader(self.train_dataset, shuffle=True, batch_size=100)
        print("Training started !\n")
        for epoch in range(num_epochs):
            self.net.train()
            epoch_loss = 0
            for batch in self.train_loader:
                # Get image and target
                images = batch['image'].to(device=self.device, dtype=torch.float32)
                targets = batch['target'].to(device=self.device, dtype=torch.float32)
                # Forward pass
                self.optimizer.zero_grad()
                preds = self.net(images)
                # Compute loss
                loss = self.criterion(preds, targets)
                # Perform backward pass
                loss.backward()
                self.optimizer.step()
                # Update epoch loss
                epoch_loss += loss.item()
            epoch_loss = epoch_loss / len(self.train_loader)
            print(f'\nEpoch: {epoch + 1} -> train_loss: {epoch_loss} \n')
        # Save model
        torch.save(self.net.state_dict(), self.bestmodel_path)
        print("Training ended!\n")

    def predict(self, test_input) -> torch.Tensor:
        #: test_input : tensor of size (N1 , C, H, W) that has to be denoised by the trained or the loaded network.
        #: returns a tensor of the size (N1 , C, H, W)

        # Send tensor to device
        test_input = test_input.to(device=self.device, dtype=torch.float32)
        test_input = test_input / 255.0
        self.net.eval()
        with torch.no_grad():
            # Make prediction
            prediction = self.net(test_input)
            # Clamp prediction and return tensor in range [0, 255]
            prediction = torch.clamp(prediction, 0, 1) * 255
        return prediction
