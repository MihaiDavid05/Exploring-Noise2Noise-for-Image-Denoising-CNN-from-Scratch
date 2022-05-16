import torch
import argparse
import json
import numpy as np
import PIL
from Miniproject_1.others.network import build_network
from Miniproject_1.others.dataset import build_dataset
from Miniproject_1.others.train import predict


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

    # Get an image
    val_dataset = build_dataset(config, config["val_data"])
    for IMAGE_INDEX in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        test_image = torch.unsqueeze(val_dataset.noisy_tensor_train[IMAGE_INDEX], dim=0)
        test_target = val_dataset.noisy_tensor_target[IMAGE_INDEX]

        # Train network
        prediction = predict(test_image, net, device=device)
        tensor = np.array(prediction, dtype=np.uint8)
        if np.ndim(tensor) > 3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        PIL.Image.fromarray(tensor.transpose((1, 2, 0)), mode="RGB").save(f"./predictions/pred{str(IMAGE_INDEX)}.png")
        PIL.Image.fromarray(np.array(test_target, dtype=np.uint8).transpose((1, 2, 0)), mode="RGB").save(f"./predictions/pred_target{str(IMAGE_INDEX)}.png")
        PIL.Image.fromarray(np.array(torch.squeeze(test_image, dim=0), dtype=np.uint8).transpose((1, 2, 0)), mode="RGB").save(f"./predictions/pred_image{str(IMAGE_INDEX)}.png")

        print("Prediction OK")
