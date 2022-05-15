import torch
import torchvision.transforms as T


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
        transformations = [torch.nn.Sequential()]
        if self.augmentations["horizontal_flip"] == 1:
            transformations.append(torch.nn.Sequential(T.RandomHorizontalFlip(p=1)))
        if self.augmentations["vertical_flip"] == 1:
            transformations.append(torch.nn.Sequential(T.RandomVerticalFlip(p=1)))
        if self.augmentations["vertical_horizontal_flip"] == 1:
            transformations.append(torch.nn.Sequential(T.RandomVerticalFlip(p=1), T.RandomHorizontalFlip(p=1)))

        permutation = torch.randperm(images.size()[0])
        batch_size = images.size()[0] // len(transformations)
        for i in range(0, images.size()[0], batch_size):
            indices = permutation[i:i + batch_size]
            batch_images = images[indices]
            batch_targets = targets[indices]
            if i == 0 and self.augmentations["swap_input_target"] == 1:
                images = torch.vstack([images, batch_targets])
                targets = torch.vstack([targets, batch_images])
            if i // batch_size != 0:
                new_images = transformations[i // batch_size](batch_images)
                new_targets = transformations[i // batch_size](batch_targets)
                images = torch.vstack([images, new_images])
                targets = torch.vstack([targets, new_targets])

        if self.augmentations["interchange_pixels"] == 1:
            for i in range(images.size()[0] // 2):
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
