import torch
import random
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from tqdm import tqdm

class Augmenter:
    """ This class contains methods used to augment the data. """
    def __init__(self, cfg):
        self.cfg = cfg
        self.augmentations = cfg["augmentations"]

    def augment_data(self, images, targets):
        """
        Augments the data.
        Args:
            self: the augmenter object
            images: the images to augment
            targets: the targets to augment

        Returns: the augmented images and targets

        """
        for i in tqdm(range(images.size(dim=0))):
            image = images[i]
            target = targets[i]
            transformed = False
            if "horizontal_flip" in self.augmentations:
                # Random horizontal flipping
                if random.random() > self.augmentations["horizontal_flip"]:
                    image = TF.hflip(image)
                    target = TF.hflip(target)
                    transformed = True
            if "vertical_flip" in self.augmentations:
                # Random vertical flipping
                if random.random() > self.augmentations["vertical_flip"]:
                    image = TF.vflip(image)
                    target = TF.vflip(target)
                    transformed = True
            if "rotation" in self.augmentations:
                # Random rotation
                if random.random() > 0.5:
                    rotation = T.RandomRotation(degrees=self.augmentations["rotation"])
                    # Random angle
                    angle = rotation.get_params(rotation.degrees)
                    image = TF.rotate(image, angle)
                    target = TF.rotate(target, angle)
                    transformed = True
            if transformed:
                if len(image.size()) < 4:
                    image = torch.unsqueeze(image, dim=0)
                    target = torch.unsqueeze(target, dim=0)
                images = torch.vstack([images, image])
                targets = torch.vstack([targets, target])

            if "swap_input_target" in self.augmentations:
                if random.random() > self.augmentations["swap_input_target"]:
                    images = torch.vstack([images, targets[i].unsqueeze(dim=0)])
                    targets = torch.vstack([targets, images[i].unsqueeze(dim=0)])
            if "interchange_pixels" in self.augmentations:
                # TODO: Use this if we know the noise is not correlated. Or maybe just try!
                if random.random() > 0.5:
                    image = torch.flatten(images[i], 1, 2)
                    img_size = image.shape[1]
                    image_copy = torch.clone(image)
                    target = torch.flatten(targets[i], 1, 2)

                    indexes = torch.randint(img_size, size=(img_size // 4, ))
                    image[:, indexes] = target[:, indexes]
                    target[:, indexes] = image_copy[:, indexes]
                    image = torch.reshape(image, images[i].size())
                    target = torch.reshape(target, targets[i].size())
                    images = torch.vstack([images, image.unsqueeze(dim=0)])
                    targets = torch.vstack([targets, target.unsqueeze(dim=0)])

        return images, targets
