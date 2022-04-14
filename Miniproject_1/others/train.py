import torch
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn


def train(train_images, val_images, net, config, device='cpu'):
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    checkpoint_dir = config["checkpoint_dir"]
    # Create PyTorch DataLoaders
    train_loader = DataLoader(train_images, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_images, shuffle=False, batch_size=1, drop_last=True)
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, amsgrad=False)
    criterion = nn.MSELoss()
    global_step = 0
    max_val_score = 0

    # Train
    print("Training started !\n")

    for epoch in range(epochs):
        # Train step
        net.train()
        epoch_loss = 0
        for batch in train_loader:
            # Get image and target
            images = batch['image']
            targets = batch['target']
            # Forward pass
            optimizer.zero_grad()
            preds = net(images)
            # Compute loss
            loss = criterion(preds, targets)
            # Perform backward pass
            loss.backward()
            optimizer.step()
            # Update global step value and epoch loss
            global_step += 1
            epoch_loss += loss.item()
        epoch_loss = epoch_loss / len(train_loader)
        print(f'Epoch: {epoch} -> train_loss: {epoch_loss} \n')

        # Evaluate model after each epoch
        print(f'Validation started !')

        net.eval()
        # Initialize varibales
        num_val_batches = len(val_loader)
        val_score = 0
        val_loss = 0
        for i, batch in enumerate(val_loader):
            # Get image and gt masks
            images = batch['image']
            targets = batch['target']

            with torch.no_grad():
                # Forward pass
                preds = net(images)
                # Compute validation loss
                loss = criterion(preds, targets)
                val_loss += loss.item()
                # Compute PNSR
                # TODO: PNSR
                val_score += 0

        net.train()
        # Update and log validation loss
        val_loss = val_loss / len(val_loader)
        print(f'Epoch: {epoch} -> val_loss: {val_loss}\n')
        val_score = val_score / num_val_batches

        if val_score > max_val_score:
            max_val_score = val_score
            print("Current maximum validation score is: {}\n".format(max_val_score))
            torch.save(net.state_dict(), checkpoint_dir + '/bestmodel.pth')
            print(f'Checkpoint {epoch} saved!\n')
        print('Validation PNR score is: {}\n'.format(val_score))
