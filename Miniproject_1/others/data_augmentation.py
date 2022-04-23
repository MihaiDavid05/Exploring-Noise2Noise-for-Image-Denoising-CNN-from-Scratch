import torch
import torch.nn as nn
import torchvision.transforms as T


class Augmenter:
    """ This class contains methods used to augment the data. """
    def __init__(self, cfg):
        self.cfg = cfg
        self.augmentations = cfg["augmentations"]

    def __create_transformer(self, mean, std):
        """
        Creates a tranformer based on the configuration.
        Args:
            self: the augmenter object

        Returns: a transformer that can be applied on the inputs and output to augment data

        """
        transformers = []
        if "rotation" in self.augmentations:
            transformers.append(T.RandomRotation(degrees=self.augmentations["rotation"]))
        if "horizontal_flip" in self.augmentations:
            transformers.append(T.RandomHorizontalFlip(p=self.augmentations["horizontal_flip"]))
        if "vertical_flip" in self.augmentations:
            transformers.append(T.RandomVerticalFlip(p=self.augmentations["vertical_flip"]))
        if "brightness" in self.augmentations:
            transformers.append(T.ColorJitter(brightness=tuple(*self.augmentations["brightness"])))
        if "contrast" in self.augmentations:
            transformers.append(T.ColorJitter(contrast=tuple(*self.augmentations["contrast"])))
        if "hue" in self.augmentations:
            transformers.append(T.ColorJitter(hue=tuple(*self.augmentations["hue"])))
        if "saturation" in self.augmentations:
            transformers.append(T.ColorJitter(saturation=tuple(*self.augmentations["saturation"])))
        if "erasing" in self.augmentations:
            transformers.append(T.RandomErasing(p=self.augmentations["erasing"]))
        if "perspective" in self.augmentations:
            transformers.append(T.RandomPerspective(p=self.augmentations["perspective"]))
        if "translate" in self.augmentations:
            transformers.append(T.RandomAffine(degrees=0, translate=self.augmentations["translate"]))
        if "scale" in self.augmentations:
            transformers.append(T.RandomAffine(degrees=0, scale=self.augmentations["scale"]))
        if "normalize" in self.augmentations:
            transformers.append(T.Normalize(mean=mean, std=std))
        return nn.Sequential(*transformers)

    def augment_data(self, images, targets):
        """
        Augments the data.
        Args:
            self: the augmenter object
            images: the images to augment
            targets: the targets to augment

        Returns: the augmented images and targets

        """
        img_mean = torch.mean(images)
        img_std = torch.std(images)
        transformer = self.__create_transformer(img_mean, img_std)
        return torch.vstack([images, transformer(images)]), torch.vstack([targets, transformer(targets)])
