# Exploring Noise2Noise for Image Denoising CNN from Scratch

### Folders
You may need to create ```logs``` and ```checkpoints``` folders under root directory. They are gitignored.

Also, you need the train and val pickles under the ```data``` folder, also created under root directory. 
### Dependencies
To install PyTorch (CPU only) use the following command:
```bash
pip install pytorch==1.9.0 torchvision torchaudio cpuonly -c pytorch
```

Other installations:
```bash
pip install tensorboard
pip install tqdm
pip install pillow
```

### Tensorboard
You may encouter an error with tensorboard. Install this:
```bash
pip install setuptools==59.5.0
```

To run Tensorboard:
```bash
python -m tensorboard.main --logdir=logs/<log_folder_wanted>
```
