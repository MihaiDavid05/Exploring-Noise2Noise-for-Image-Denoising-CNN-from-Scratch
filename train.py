import torch


if __name__ == "__main__":
    noisy_imgs_1, noisy_imgs_2 = torch.load('train_data.pkl')
    noisy_imgs , clean_imgs = torch.load('val_data.pkl')
