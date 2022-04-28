import torch
import json
import argparse
from Miniproject_1.others.network import build_network
from Miniproject_1.others.dataset import build_dataset
from Miniproject_1.others.train import train
from torch.utils.tensorboard import SummaryWriter


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="Path to config")
    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    args = get_args()

    # Get config
    with open(args.config_path) as json_config:
        config = json.load(json_config)

    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device state: ', device)

    # Get network
    net = build_network(config)
    net.to(device=device)

    # Get dataset
    train_dataset = build_dataset(config, config["train_data"], train=True)
    val_dataset = build_dataset(config, config["val_data"])

    # Train network
    writer = SummaryWriter(log_dir=config["log_dir"])
    train(train_dataset, val_dataset, net, config, writer,  device=device)
