# DL_project

### 1.Folders
You may need to create ```logs``` and ```checkpoints``` folders under root directory. They are gitignored.

Also, you need the train and val pickles under the ```data``` folder, also created under root directory. 
### 2.Dependencies
To install PyTorch (CPU only) use the following command:
```bash
pip install pytorch==1.9.0 torchvision torchaudio cpuonly -c pytorch
```

Other installations:
```bash
pip install tensorboard
```

You may encouter an error with tensorboard. Install this:
```bash
pip install setuptools==59.5.0
```

To run Tensorboard, run this:
```bash
python -m tensorboard.main --logdir=logs/<log_folder_wanted>
```
